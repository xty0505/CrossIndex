import os
import time
import operator
from functools import reduce
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from query import Query, Condition
from sort_parallel import Sort
from type_parallel import *

os.environ["MODIN_ENGIN"] = "ray"


class VizCube(object):
    def __init__(self, name, dimensions=[], types=[]):
        self.R = pd.DataFrame()
        self.name = name
        self.dimensions = dimensions
        self.types = [Type.getType(t) for t in types]
        self.dimensionSetLayers = []
        self.ready = False
        self.pbar = None

    def save(self, path):
        self.R.to_csv(os.path.join(path, self.name + '.csv'), index=False, encoding='utf-8')
        with open(os.path.join(path, self.name + '.cube'), 'w', encoding='utf-8') as f:
            # dimensions
            for i in range(len(self.dimensions)):
                if i != len(self.dimensions) - 1:
                    if type(self.dimensions[i]) == list:
                        f.write('geohash' + ',')
                    else:
                        f.write(self.dimensions[i] + ',')
                else:
                    if type(self.dimensions[i]) == list:
                        f.write('geohash' + '\n')
                    else:
                        f.write(self.dimensions[i] + '\n')
            # types
            for i in range(len(self.types)):
                if i != len(self.types) - 1:
                    f.write(str(self.types[i]) + ',')
                else:
                    f.write(str(self.types[i]) + '\n')
            # cube
            for ds in self.dimensionSetLayers[0]:
                i = 0
                self.recursiveSave(f, ds, i)
        f.close()

    def recursiveSave(self, f, ds, i):
        f.write(i * '>' + ds.save() + '\n')
        if len(ds.subSet) == 0:
            return
        for sub in ds.subSet:
            self.recursiveSave(f, sub, i + 1)

    def load(self, path, name):
        self.R = pd.read_csv(os.path.join(path, name + '.csv'), encoding='utf-8', delimiter=',')
        self.name = name
        with open(os.path.join(path, name + '.cube'), 'r', encoding='utf-8') as f:
            line = f.readline().strip('\n')
            self.dimensions = list(line.strip('\n').split(','))
            line = f.readline().strip('\n')
            self.types = list(line.strip('\n').split(','))

            for i in range(len(self.dimensions)):
                self.dimensionSetLayers.append([])
            line = f.readline().strip('\n')
            last_l = 0
            while line:
                l = line.count('>')
                ds = DimensionSet('', '', None)
                ds.load(line[l:])
                if last_l == l:
                    self.dimensionSetLayers[l].append(ds)
                    line = f.readline().strip('\n')
                elif last_l < l:
                    _, line = self.recursiveLoad(f, line, l)
        f.close()
        self.ready = True

    def recursiveLoad(self, f, line, last_l):
        subSet = []
        while line:
            l = line.count('>')
            if last_l == l:
                ds = DimensionSet('', '', None, self.dimensionSetLayers[last_l - 1][-1])
                ds.load(line[l:])
                self.dimensionSetLayers[l].append(ds)
                subSet.append(ds)
                line = f.readline().strip('\n')
            elif last_l > l:
                self.dimensionSetLayers[last_l - 1][-1].subSet = subSet
                return last_l - 1, line
            elif last_l < l:
                last_l, line = self.recursiveLoad(f, line, l)
        self.dimensionSetLayers[last_l - 1][-1].subSet = subSet
        return last_l - 1, ''

    def build(self):
        # process bar
        self.pbar = tqdm(desc='VizCube Build', total=len(self.R) * len(self.dimensions))

        i = 0
        # first dimension
        dimension = self.dimensions[i]
        dimensionType = self.types[i]
        i = i + 1
        begin = 0
        firstLayer = []
        self.R.sort_values(by=dimension, inplace=True)
        self.R.reset_index(drop=True, inplace=True)
        for index, value in self.R[dimension].value_counts().sort_index().items():
            ds = DimensionSet(dimension, index, Interval(begin, begin + value - 1))
            begin = begin + value
            firstLayer.append(ds)
            self.pbar.update(ds.interval.count)
        self.dimensionSetLayers.append(firstLayer)

        # other dimensions
        while i < len(self.dimensions):
            layer = []
            for dS in self.dimensionSetLayers[i - 1]:
                dimension = self.dimensions[i]
                dimensionType = self.types[i]
                begin = dS.interval.begin
                end = dS.interval.end
                partialRelation = self.R.iloc[begin:end + 1]
                partialRelation = partialRelation.sort_values(by=dimension)
                partialRelation.set_index(pd.Index(range(begin, end + 1)), inplace=True)
                for index, value in partialRelation[dimension].value_counts().sort_index().items():
                    ds = DimensionSet(dimension, str(index), Interval(begin, begin + value - 1))
                    begin = begin + value
                    dS.subSet.append(ds)
                    layer.append(ds)
                    self.pbar.update(ds.interval.count)
                self.R.iloc[dS.interval.begin:dS.interval.end + 1, :] = partialRelation[:]
            i = i + 1
            self.dimensionSetLayers.append(layer)
        self.pbar.close()

    def build2(self, path, delimiter):
        start = time.time()

        self.R = pd.read_csv(path, encoding='utf-8', delimiter=delimiter)
        print('pd.read_csv finished.')
        # process bar
        self.pbar = tqdm(desc='VizCube Build', total=len(self.R) * len(self.dimensions))
        for i in range(len(self.dimensions)):
            dimension = self.dimensions[i]
            dimensionType = self.types[i]
            sort = Sort(self.R, 0, -1)
            layer = []
            if i == 0:
                layer = sort.sort(dimension, dimensionType, self.pbar)
                self.dimensionSetLayers.append(layer)
            else:
                for ds in self.dimensionSetLayers[i - 1]:
                    begin = ds.interval.begin
                    end = ds.interval.end
                    sort.setBeginEnd(begin, end)
                    sort.setDs(ds)
                    layer = layer + sort.sort(dimension, dimensionType, self.pbar)
                self.dimensionSetLayers.append(layer)
        self.pbar.close()
        self.ready = True
        end = time.time()
        print('build time:' + str(end - start))

    def build_parallel(self, path, delimiter):
        start = time.time()
        self.R = pd.read_csv(path, encoding='utf-8', delimiter=delimiter)
        print('pd.read_csv finished.')
        # process bar
        self.pbar = tqdm(desc='VizCube Build Parallel', total=len(self.R) * len(self.dimensions))
        for i in range(len(self.dimensions)):
            dimension = self.dimensions[i]
            dimensionType = self.types[i]
            sort = Sort(self.R, 0, len(self.R) - 1)
            if i == 0:
                root = DimensionSet('root', 'all', Interval(0, len(self.R) - 1))
                result = sort.sort(0, len(self.R) - 1, root, dimension, dimensionType, self.pbar)
                # self.R = result[1]
                self.dimensionSetLayers.append(result)
            else:
                result = Parallel(n_jobs=8, backend='threading')(
                    delayed(sort.sort)(ds.interval.begin, ds.interval.end, ds, dimension, dimensionType, self.pbar) for
                    ds in self.dimensionSetLayers[i - 1])
                layer = reduce(operator.add, result)
                # self.R = pd.concat(result[1])
                self.dimensionSetLayers.append(layer)
        self.pbar.close()
        self.ready = True
        end = time.time()
        print('build time:' + str(end - start))

    def query(self, query):
        # 记录当前循环中符合之前 where 条件即有效的 DimensionSet
        validDSs = []
        root = DimensionSet('root', 'all', None)
        root.subSet = self.dimensionSetLayers[0]
        validDSs.append(root)
        xyMap = defaultdict(lambda: [])
        n = query.where_n
        # no wheres
        if n == 0:
            for i in range(len(self.dimensions)):
                if self.dimensions[i] == query.groupby:
                    for ds in self.dimensionSetLayers[i]:
                        xyMap[ds.value].append(ds.interval)
        else:
            for i in range(len(self.dimensions)):
                if n == 0:
                    break
                tmpDS = []
                if query.wheres[i].value is None:
                    if self.dimensions[i] == query.groupby:
                        for ds in validDSs:
                            for sub in ds.subSet:
                                tmpDS.append(sub)
                                xyMap[sub.value].append(sub.interval)
                    else:
                        for ds in validDSs:
                            for sub in ds.subSet:
                                tmpDS.append(sub)
                    validDSs = tmpDS
                    continue
                where = query.wheres[i]
                for ds in validDSs:
                    for sub in ds.subSet:
                        if where.match(sub.value):
                            tmpDS.append(sub)
                # 该 where 条件限定的列刚好是 group by 的列
                if self.dimensions[i] == query.groupby:
                    for ds in tmpDS:
                        xyMap[ds.value].append(ds.interval)
                # group by 的列已经遇到
                elif len(xyMap) > 0:
                    tmpMap = defaultdict(lambda: [])
                    groupby_validDSs = defaultdict(lambda: [])
                    for ds in validDSs:
                        for sub in ds.subSet:
                            if where.match(sub.value):
                                p = sub.find_parent(query.groupby)
                                if p is not None:
                                    groupby_validDSs[p.value].append(sub.interval)
                    for key in xyMap.keys():
                        if len(groupby_validDSs[key]) != 0:
                            tmpMap[key] = groupby_validDSs[key]
                        xyMap = tmpMap
                validDSs = tmpDS
                n = n - 1
            # n = 0 break循环后, 如果 groupby列没有遇到
            if len(xyMap) == 0:
                while validDSs[0].dimension != query.groupby:
                    tmpDS = []
                    for ds in validDSs:
                        for sub in ds.subSet:
                            tmpDS.append(sub)
                    validDSs = tmpDS
                for ds in validDSs:
                    xyMap[ds.value].append(ds.interval)

        query.validDSs = validDSs
        for key in sorted(xyMap.keys()):
            query.result.x_data.append(key)
            query.result.y_intervals.append(xyMap[key])
        query.compute()
        return query.result.output_xy()

    def query2(self, query):
        # 记录当前循环中符合之前 where 条件即有效的 DimensionSet
        validDSs = []
        root = DimensionSet('root', 'all', None)
        root.subSet = self.dimensionSetLayers[0]
        validDSs.append(root)
        xyMap = defaultdict(lambda: [])
        n = query.where_n
        # no wheres
        if n == 0:
            for i in range(len(self.dimensions)):
                if self.dimensions[i] == query.groupby:
                    for ds in self.dimensionSetLayers[i]:
                        xyMap[ds.value].append(ds.interval)
        else:
            for i in range(len(self.dimensions)):
                if n == 0:
                    break
                tmpDS = []
                if query.wheres[i].value is None:
                    for ds in validDSs:
                        for sub in ds.subSet:
                            tmpDS.append(sub)
                    validDSs = tmpDS
                    continue
                where = query.wheres[i]
                for ds in validDSs:
                    for sub in ds.subSet:
                        if where.match(sub.value):
                            tmpDS.append(sub)
                validDSs = tmpDS
                n = n - 1
            # group by
            valid_dimension_i = self.dimensions.index(validDSs[0].dimension)
            groupby_i = self.dimensions.index(query.groupby)
            if valid_dimension_i == groupby_i:
                for ds in validDSs:
                    xyMap[ds.value].append(ds.interval)
            elif valid_dimension_i > groupby_i:
                for ds in validDSs:
                    p = ds.find_parent(query.groupby)
                    if p is not None:
                        xyMap[p.value].append(ds.interval)
            else:
                while validDSs[0].dimension != query.groupby:
                    tmpDS = []
                    for ds in validDSs:
                        for sub in ds.subSet:
                            tmpDS.append(sub)
                    validDSs = tmpDS
                for ds in validDSs:
                    xyMap[ds.value].append(ds.interval)

        query.validDSs = validDSs
        for key in sorted(xyMap.keys()):
            query.result.x_data.append(key)
            query.result.y_intervals.append(xyMap[key])
        query.compute()
        return query.result.output_xy()

    def backward_query(self, query, conditions):
        j = 0
        xyMap = defaultdict(lambda: [])
        validDSs = query.validDSs
        for i in range(len(self.dimensions)):
            if j >= len(conditions):
                break
            if conditions[j].dimension != self.dimensions[i]:
                continue

            tmpDS = []
            new_i = self.dimensions.index(conditions[j].dimension)
            old_i = self.dimensions.index(validDSs[0].dimension)
            # condition 列和 groupby 列为同一列
            if new_i == old_i:
                for ds in validDSs:
                    if conditions[j].match(ds.value):
                        tmpDS.append(ds)
                validDSs = tmpDS
            # condition 列 < groupby 列
            elif new_i < old_i:
                for ds in validDSs:
                    p = ds.find_parent(conditions[j].dimension)
                    if p is not None and conditions[j].match(p.value):
                        tmpDS.append(ds)
                validDSs = tmpDS
            # condition > groupby
            elif new_i > old_i:
                while validDSs[0].dimension != conditions[j].dimension:
                    for ds in validDSs:
                        for sub in ds.subSet:
                            tmpDS.append(sub)
                    validDSs = tmpDS
                    tmpDS = []
                tmpDS = []
                for ds in validDSs:
                    if conditions[j].match(ds.value):
                        tmpDS.append(ds)
                validDSs = tmpDS
            query.add_condition(conditions[j])
            # 最外层循环
            j = j + 1

        # group by
        if validDSs[0].dimension == query.groupby:
            for ds in validDSs:
                xyMap[ds.value].append(ds.interval)
        else:
            for ds in validDSs:
                p = ds.find_parent(query.groupby)
                if p is not None:
                    xyMap[p.value].append(ds.interval)
        query.validDSs = validDSs
        query.clear()
        for key in sorted(xyMap.keys()):
            query.result.x_data.append(key)
            query.result.y_intervals.append(xyMap[key])
        query.compute()
        return query.result.output_xy()

    def output(self):
        for ds in self.dimensionSetLayers[0]:
            ds.output()


if __name__ == '__main__':
    # traffic.csv
    # vizcube = VizCube('traffic', './data/traffic.csv', ['link_id', 'time'], '\t')
    # vizcube.build()
    # vizcube.save()
    # vizcube.load('traffic')
    # vizcube.output()
    # sql = "SELECT COUNT(vehicle_num) from traffic WHERE time = '651'  GROUP BY link_id"

    # trace.csv
    vizcube = VizCube('trace',
                      [['lng', 'lat'], 'link_id', 'vehicle_id'],
                      ['spatial', 'categorical', 'categorical'])
    # vizcube.build2()
    # vizcube.save()
    vizcube.load('../cube/', 'trace')
    sql = "SELECT COUNT(vehicle_length) from trace WHERE geohash='wtw3sm'  GROUP BY geohash"

    # myshop.csv
    # vizcube = VizCube('myshop_temporal', './data/websales_home_myshop.csv',
    #                   ['category', 'itemname', 'gender', 'nationality', 'date'],
    #                   ['categorical', 'categorical', 'categorical', 'categorical', 'temporal'],
    #                   '\t')
    # vizcube.build2()
    # vizcube.save()
    # vizcube.load('myshop_temporal')
    # sql = "SELECT AVG(quantity) from myshop WHERE gender = '女' AND date BETWEEN '2019' and '2020' GROUP BY category"

    q = Query(measure='', agg='', groupby='', cube=vizcube)
    q.parse(sql)

    sql1 = "SELECT COUNT(vehicle_length) from trace WHERE geohash='wtw3sm'  AND link_id IN ['152C909GV90152CL09GVD00', '152D309GVT0152CJ09GVM00']  GROUP BY geohash"
    q1 = Query(measure='', agg='', groupby='', cube=vizcube)
    q1.parse(sql1)

    # 直接query
    start = time.time()
    vizcube.query(q1)
    end = time.time()

    q1.clear()
    start1 = time.time()
    vizcube.query2(q1)
    end1 = time.time()

    print('query time:' + str(end - start))
    print('query2 time:' + str(end1 - start1))
    q1.result.pretty_output()

    # backward_query
    vizcube.query2(q)
    # Condition('link_id', ['152C909GV90152CL09GVD00', '152D309GVT0152CJ09GVM00'], Type.categorical),
    condition = [Condition('link_id', ['152C909GV90152CL09GVD00', '152D309GVT0152CJ09GVM00'], Type.categorical)]
    start = time.time()
    vizcube.backward_query(q, condition)
    end = time.time()

    q.result.pretty_output()
    print('back_query: ' + str(end - start))
