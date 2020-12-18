import os
import time
import operator
import argparse
from functools import reduce
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from query import Query, Condition
from sort_parallel import Sort
from type_parallel import *


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
            self.types = [int(x) for x in list(line.strip('\n').split(','))]
            # self.types = [Type.getType(x) for x in list(line.strip('\n').split(','))]

            for i in range(len(self.dimensions)):
                self.dimensionSetLayers.append([])
            line = f.readline().strip('\n')
            last_l = 0
            while line:
                l = line.count('>')
                ds = DimensionSet('', -1, '', None)
                ds.load(line[l:], self.dimensions, self.types)
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
                ds = DimensionSet('', -1, '', None, self.dimensionSetLayers[last_l - 1][-1])
                ds.load(line[l:], self.dimensions, self.types)
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

    def build(self, path, delimiter):
        start = time.time()
        self.R = pd.read_csv(path, encoding='utf-8', delimiter=delimiter)
        print('pd.read_csv finished.')

        # bin numerical
        for i in range(len(self.dimensions)):
            if self.types[i] == Type.numerical:
                bin_label = self.dimensions[i] + '_bin'
                bin_width = 20
                self.R[bin_label] = pd.cut(self.R[self.dimensions[i]], bin_width).tolist()

        # process bar
        self.pbar = tqdm(desc='VizCube Build Parallel', total=len(self.R) * len(self.dimensions))
        for i in range(len(self.dimensions)):
            dimension = self.dimensions[i]
            dimension_type = self.types[i]
            sorter = Sort(self.R)

            if i == 0:
                root = DimensionSet('root', -1, 'all', Interval(0, len(self.R) - 1))
                result = sorter.sort(0, len(self.R) - 1, root, dimension, dimension_type, self.pbar)
                self.dimensionSetLayers.append(result['layer'])
                self.R = result['r']
                del result
            else:
                results = []
                for ds in self.dimensionSetLayers[i - 1]:
                    results.append(sorter.sort(ds.interval.begin, ds.interval.end, ds, dimension, dimension_type, self.pbar))
                layer = reduce(operator.add, [r['layer'] for r in results])
                self.R = pd.concat([r['r'] for r in results])
                self.dimensionSetLayers.append(layer)
                del results
            if dimension_type == Type.numerical:
                self.R.drop(columns=[dimension + '_bin'], inplace=True)
            del sorter
        self.pbar.close()
        self.ready = True
        end = time.time()
        print('build time:' + str(end - start))

    def build_parallel(self, path, delimiter):
        start = time.time()
        self.R = pd.read_csv(path, encoding='utf-8', delimiter=delimiter)
        print('pd.read_csv finished.')

        # bin numerical
        for i in range(len(self.dimensions)):
            if self.types[i] == Type.numerical:
                bin_label = self.dimensions[i] + '_bin'
                bin_width = 20
                self.R[bin_label] = pd.cut(self.R[self.dimensions[i]], bin_width).tolist()

        # process bar
        self.pbar = tqdm(desc='VizCube Build Parallel', total=len(self.R) * len(self.dimensions))
        for i in range(len(self.dimensions)):
            dimension = self.dimensions[i]
            dimension_type = self.types[i]
            sorter = Sort(self.R)

            if i == 0:
                root = DimensionSet('root', -1, 'all', Interval(0, len(self.R) - 1))
                result = sorter.sort(0, len(self.R) - 1, root, dimension, dimension_type, self.pbar)
                self.dimensionSetLayers.append(result['layer'])
                self.R = result['r']
                del result
            else:
                results = Parallel(n_jobs=8, backend='threading')(
                   delayed(sorter.sort)(ds.interval.begin, ds.interval.end, ds, dimension, dimension_type, self.pbar) for
                   ds in self.dimensionSetLayers[i - 1])
                # result = []
                # for ds in self.dimensionSetLayers[i - 1]:
                #     result.append(sort.sort(ds.interval.begin, ds.interval.end, ds, dimension, dimension_type, self.pbar))
                layer = reduce(operator.add, [r['layer'] for r in results])
                self.R = pd.concat([r['r'] for r in results])
                self.dimensionSetLayers.append(layer)
                del results
            if dimension_type == Type.numerical:
                self.R.drop(columns=[dimension + '_bin'], inplace=True)
            del sorter
        self.pbar.close()
        self.ready = True
        end = time.time()
        print('build time:' + str(end - start))

    def query(self, query):
        # 记录当前循环中符合之前 where 条件即有效的 DimensionSet
        validDSs = []
        root = DimensionSet('root', -1, 'all', None)
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
                tmpDS = []
                where = query.wheres[i]

                # 当前 dimension 无where条件
                if where.value is None:
                    for ds in validDSs:
                        tmpDS.extend(ds.subSet)
                    validDSs = tmpDS
                    continue

                # numerical 可能会出现 condition 落在 bin_boundary 中间的情况
                if self.types[i] == Type.numerical:
                    for ds in validDSs:
                        for sub in ds.subSet:
                            is_match, new_sub = where.match(sub, self.R)
                            if is_match:
                                if new_sub is not None:
                                    tmpDS.append(new_sub)
                                else:
                                    tmpDS.append(sub)
                else:
                    for ds in validDSs:
                        for sub in ds.subSet:
                            if where.match(sub):
                                tmpDS.append(sub)
                validDSs = tmpDS

                # 当前 where 条件筛选完后如果没有后续条件直接break
                n = n - 1
                if n == 0:
                    break

            if len(validDSs) == 0:
                return query.result.output_xy()

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
                        tmpDS.extend(ds.subSet)
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
        if len(validDSs) == 0:
            return query.result.output_xy()
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
                    if conditions[j].match(ds):
                        tmpDS.append(ds)
                validDSs = tmpDS
            # condition 列 < groupby 列
            elif new_i < old_i:
                for ds in validDSs:
                    p = ds.find_parent(conditions[j].dimension)
                    if p is not None and conditions[j].match(p):
                        tmpDS.append(ds)
                validDSs = tmpDS
            # condition > groupby
            elif new_i > old_i:
                while validDSs[0].dimension != conditions[j].dimension:
                    for ds in validDSs:
                        tmpDS.extend(ds.subSet)
                    validDSs = tmpDS
                    tmpDS = []
                tmpDS = []
                for ds in validDSs:
                    if conditions[j].match(ds):
                        tmpDS.append(ds)
                validDSs = tmpDS
            query.add_condition(conditions[j])

            if len(validDSs) == 0:
                return query.result.output_xy()
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

    def backward_query2(self, query, conditions):
        validDSs = []
        xyMap = defaultdict(lambda: [])

        for i in len(conditions):
            tmp = []

            condition = conditions[i]
            next_condition = conditions[i+1] if (i+1) < len(conditions) else None

            idx = self.dimensions.index(condition.dimension)
            for ds in query.validDimensionSetLayers[idx]:
                if condition.match(ds):
                    validDSs.append(ds)
            if next_condition is not None:
                while validDSs[0].dimension != next_condition.dimension:
                    for ds in validDSs:
                        tmp.extend(ds.subSet)
                    validDSs = tmp

        if len(validDSs) == 0:
            return query.result.output_xy()
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
                    tmpDS.extend(ds.subSet)
                validDSs = tmpDS
            for ds in validDSs:
                xyMap[ds.value].append(ds.interval)

        query.validDSs = validDSs
        for key in sorted(xyMap.keys()):
            query.result.x_data.append(key)
            query.result.y_intervals.append(xyMap[key])
        query.compute()
        return query.result.output_xy()

    def output(self):
        for ds in self.dimensionSetLayers[0]:
            ds.output()

    def execute_query(self, sql):
        q = Query(cube=vizcube)
        q.parse(sql)

        start = time.time()
        vizcube.query(q)
        end = time.time()

        q.result.pretty_output()
        print('direct query time:' + str(end - start))

def execute_direct_query(vizcube, sql):
    q = Query(cube=vizcube)
    q.parse(sql)

    start = time.time()
    vizcube.query(q)
    end = time.time()

    q.result.pretty_output()
    print('direct query time:' + str(end - start))

    return q

def execute_backward_query(vizcube, sql, new_conditions):
    q = Query(cube=vizcube)
    q.parse(sql)

    vizcube.query(q)
    start = time.time()
    vizcube.backward_query(q, new_conditions)
    end = time.time()
    q.result.pretty_output()
    print('backward query: ' + str(end - start))

    return q


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='An index for accelerating interactive data exploration.')
    argparser.add_argument('--input-dir', dest='input_dir', help='input csv file directory')
    argparser.add_argument('--cube-dir', dest='cube_dir', help='cube file directory', default='cube/')
    argparser.add_argument('--name', dest='name', help='cube file and csv file name')
    # if dimension type is spatial, it should be like "lng,lat"
    argparser.add_argument('--dimensions', dest='dimensions', nargs='+', type=str, help='dimensions need to be filtered')
    argparser.add_argument('--types', dest='types', nargs='+', type=str, help='types of dimensions')

    argparser.add_argument('--delimiter', dest='delimiter', help='delimiter of csv file', default=',')
    argparser.add_argument('--single', dest='single', action='store_true', help='whether to build in single thread', default=False)

    args = vars(argparser.parse_args())

    # combine dimension args if its tpye is spatial
    for i in range(len(args['types'])):
        if args['types'][i] == 'spatial':
            args['dimensions'][i] = args['dimensions'][i].split(',')

    # initialization
    vizcube =VizCube(args['name'], args['dimensions'], args['types'])
    if os.path.exists(args['cube_dir'] + args['name'] + '.cube'):
        vizcube.load(args['cube_dir'], args['name'])
    else:
        if args['single'] == True:
            vizcube.build(args['input_dir'], args['delimiter'])
        else:
            vizcube.build_parallel(args['input_dir'], args['delimiter'])
        vizcube.save(args['cube_dir'])

    # sql = "SELECT AVG(quantity) from myshop WHERE gender = '女' AND date BETWEEN '2019' and '2020' GROUP BY category"
    # q = execute_direct_query(vizcube, sql)
    # print(q.result.convert_to_filters_IN())
    # print(q.result.convert_to_filters())



''' 
====EXPERIMENT ARGS====
flights_1M_numerical.csv args:
    --input-dir data/dataset_flights_1M.csv --name flights_1M_numerical --dimensions DISTANCE AIR_TIME ARR_TIME DEP_TIME ARR_DELAY DEP_DELAY --types numerical numerical numerical numerical categorical categorical 
    sql = "SELECT FLOOR(ARR_TIME/1) AS bin_ARR_TIME,  COUNT(*) as count FROM flights WHERE (AIR_TIME >= 150 AND AIR_TIME < 500 AND DISTANCE >= 0 AND DISTANCE < 1000) GROUP BY bin_ARR_TIME"
    execute_direct_query(vizcube, sql)

flights_1M_categorical.csv args:
    --input-dir data/dataset_flights_1M.csv --name flights_1M_categorical --dimensions AIR_TIME ARR_DELAY ARR_TIME DEP_DELAY DEP_TIME DISTANCE --types categorical categorical categorical categorical categorical categorical
    sql = "SELECT FLOOR(DEP_TIME/1) AS bin_DEP_TIME,  COUNT(*) as count FROM flights WHERE (DISTANCE >= 985.7142857142858 AND DISTANCE < 1200 AND AIR_TIME >= 122.85714285714286 AND AIR_TIME < 500) GROUP BY bin_DEP_TIME"
    backward_sql = "SELECT FLOOR(DEP_TIME/1) AS bin_DEP_TIME,  COUNT(*) as count FROM flights WHERE (DISTANCE >= 985.7142857142858 AND DISTANCE < 1200) GROUP BY bin_DEP_TIME"
    execute_direct_query(vizcube, sql)
    execute_backward_query(vizcube, backward_sql, [Condition('AIR_TIME', [122.85714285714286, 500], Type.categorical)])

traffic_categorical.csv args:
    --input-dir data/traffic.csv --name traffic_categorical --dimensions link_id vehicle_num velocity_ave time --types categorical categorical categorical categorical --delimiter \t
    
    
traffic_numerical.csv args:
    --input-dir data/traffic.csv --name traffic_numerical --dimensions link_id vehicle_num velocity_ave time --types categorical numerical numerical categorical --delimiter \t
    sql = "SELECT COUNT(vehicle_num) from traffic WHERE velocity_ave >= 4 AND velocity_ave < 8 GROUP BY time"

traffic.csv args:
    --name traffic --dimensions link_id time --types categorical categorical --delimiter \t
    sql = "SELECT COUNT(vehicle_num) from traffic WHERE time = '651'  GROUP BY link_id"

trace.csv args:
    --name trace --dimensions "lng,lat" link_id vehicle_id timestep --types spatial categorical categorical categorical
    direct_sql = "SELECT COUNT(vehicle_length) from trace WHERE geohash='wtw3sm'  AND link_id IN ['152C909GV90152CL09GVD00', '152D309GVT0152CJ09GVM00'] AND timestep >= 39654.4 AND timestep < 39800.4 GROUP BY geohash"
    backward_sql = "SELECT COUNT(vehicle_length) from trace WHERE geohash='wtw3sm' AND timestep >= 39654.4 AND timestep < 39800.4 GROUP BY geohash"
    execute_direct_query(vizcube, direct_sql)
    execute_backward_query(vizcube, backward_sql,[Condition('link_id', ['152C909GV90152CL09GVD00', '152D309GVT0152CJ09GVM00'], Type.categorical)])
    
myshop_temporal.csv args:
    --input-dir data/myshop.csv --name myshop_temporal --dimensions category itemname gender nationality date --types categorical categorical categorical categorical temporal --delimiter \t
    sql = "SELECT AVG(quantity) from myshop WHERE gender = '女' AND date BETWEEN '2019' and '2020' GROUP BY category"
    
bike.csv args:
    --input-dir data/Divvy_Trips.csv --name bike --dimensions "FROM_LONGITUDE,FROM_LATITUDE" USER_TYPE START_TIME -types spatial categorical temporal
'''
