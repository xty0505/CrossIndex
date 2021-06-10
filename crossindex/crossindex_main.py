import argparse
import time
import json
import random
from collections import defaultdict

from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np

from calc_type import *
from dispatcher import Dispatcher
from omnisci_conf.omnisci_manager import OmnisciManager
from query import Query,Condition


class CrossIndex(object):
    def __init__(self, name, dimensions=[], types=[], use_omnisci=False):
        self.R = pd.DataFrame()
        self.name = name
        self.dimensions = dimensions
        self.types = [Type.getType(t) for t in types]
        self.bin_width = []
        self.bin_count = []
        self.offset = []
        self.dimensionSetLayers = []
        self.index = pd.DataFrame() # csv form
        self.ready = False
        self.pbar = None
        self.use_omnisci = use_omnisci
        if self.use_omnisci:
            self.omnisci = OmnisciManager()

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

    def save_csv(self, args):
        self.index.to_csv(os.path.join(args['cube_dir'], self.name + '_csv.csv'), index=False, encoding='utf-8')
        self.R.to_csv(os.path.join(args['cube_dir'], self.name + '.csv'), index=False, encoding='utf-8')
        with open(os.path.join(args['cube_dir'], self.name + '.json'), 'w') as f:
            json.dump({"dimensions":self.dimensions,"types":self.types,"bin_count":self.bin_count, "bin_width":self.bin_width, "offset":self.offset}, f, indent=4)        

    def load_csv(self, args):
        with open(args['cube_dir']+args['name']+'.json', 'r') as f:
            meta_info = json.load(f)
            self.dimensions = meta_info['dimensions']
            self.types = meta_info['types']
            self.bin_count = meta_info['bin_count']
            self.bin_width = meta_info['bin_width']
            self.offset = meta_info['offset']
            f.close()
        self.R = pd.read_csv(os.path.join(args['cube_dir'], args['name'] + '.csv'), encoding='utf-8')
        self.index = pd.read_csv(os.path.join(args['cube_dir'], args['name'] + '_csv.csv'), encoding='utf-8')

    def build(self, path, delimiter, options):
        start = time.time()
        # self.R = pd.read_csv(path, encoding='utf-8', delimiter=delimiter)
        # print('pd.read_csv finished.')

        # sorting
        print('sorting...')
        self.R = self.R.sort_values(self.dimensions)
        self.R.reset_index(drop=True, inplace=True)
        print('sorting done.')
        print('sorting time: '+str(time.time()-start))

        # bin numerical
        for i in range(len(self.dimensions)):
            if self.types[i] == Type.numerical:
                bin_label = self.dimensions[i] + '_bin'
                bin_width = 20
                self.R[bin_label] = pd.cut(self.R[self.dimensions[i]], bin_width).tolist()

        # process bar
        self.pbar = tqdm(desc='CrossIndex Build Parallel', total=len(self.R) * len(self.dimensions))
        for i in range(len(self.dimensions)):
            dimension = self.dimensions[i]
            dimension_type = self.types[i]
            dispatcher = Dispatcher(R=self.R, options=options)

            if i == 0:
                root = DimensionSet('root', -1, 'all', Interval(0, len(self.R) - 1))
                layer = dispatcher.dispatch(0, len(self.R) - 1, root, dimension, dimension_type, self.pbar)
                self.dimensionSetLayers.append(layer)
                del layer
            else:
                layer = []
                for ds in self.dimensionSetLayers[i - 1]:
                    l = dispatcher.dispatch(ds.interval.begin, ds.interval.end, ds, dimension, dimension_type, self.pbar)
                    layer.extend(l)
                self.dimensionSetLayers.append(layer)
                del layer
            if dimension_type == Type.numerical:
                self.R.drop(columns=[dimension + '_bin'], inplace=True)
            del dispatcher
        self.pbar.close()
        self.ready = True
        end = time.time()
        print('build time:' + str(end - start))

    def build_parallel(self, path, delimiter, options):
        start = time.time()
        # self.R = pd.read_csv(path, encoding='utf-8', delimiter=delimiter)
        # print('pd.read_csv finished.')

        # sorting
        print('sorting...')
        self.R = self.R.sort_values(self.dimensions)
        self.R.reset_index(drop=True, inplace=True)
        print('sorting done.')
        print('sorting time: '+str(time.time()-start))

        # bin numerical
        for i in range(len(self.dimensions)):
            if self.types[i] == Type.numerical:
                bin_label = self.dimensions[i] + '_bin'
                bin_width = 20
                self.R[bin_label] = pd.cut(self.R[self.dimensions[i]], bin_width).tolist()

        # process bar
        self.pbar = tqdm(desc='CrossIndex Build Parallel', total=len(self.R) * len(self.dimensions))
        for i in range(len(self.dimensions)):
            dimension = self.dimensions[i]
            dimension_type = self.types[i]
            dispatcher = Dispatcher(R=self.R, options=options)

            if i == 0:
                root = DimensionSet('root', -1, 'all', Interval(0, len(self.R) - 1))
                layer = dispatcher.dispatch(0, len(self.R) - 1, root, dimension, dimension_type, self.pbar)
                self.dimensionSetLayers.append(layer)
                del layer
            else:
                results = Parallel(n_jobs=8, backend='threading')(
                   delayed(dispatcher.dispatch)(ds.interval.begin, ds.interval.end, ds, dimension, dimension_type, self.pbar) for
                   ds in self.dimensionSetLayers[i - 1])
                # concat multi thread results
                concat_start = time.time()
                layer = [item for sublist in results for item in sublist]
                print('concat time: '+str(time.time()-concat_start))
                self.dimensionSetLayers.append(layer)
                del results
            if dimension_type == Type.numerical:
                self.R.drop(columns=[dimension + '_bin'], inplace=True)
            del dispatcher
        self.pbar.close()
        self.ready = True
        end = time.time()
        print('build time:' + str(end - start))

    def build_csv(self):
        # sorting
        start = time.time()
        print('sorting...')
        sortby = []
        to_drop = []
        for i in range(len(self.dimensions)):
            sortby.append(self.dimensions[i])
            if self.types[i] == Type.numerical:
                self.R[self.dimensions[i]+'_bin'] = (self.R[self.dimensions[i]]-self.offset[i])//self.bin_width[i]
                to_drop.append(self.dimensions[i]+'_bin')
        self.R = self.R.sort_values(to_drop+sortby) # 先分组有序，然后组内有序
        self.R.reset_index(drop=True, inplace=True)
        print('sorting done.')
        print('sorting time: '+str(time.time()-start))

        def calInterval(row):
            return str(row.min())+','+str(row.max())

        # groupby
        self.pbar = tqdm(desc='CrossIndex Build', total=len(self.R) * len(self.dimensions))
        crossindex = self.R[self.dimensions]
        groupby = list(self.dimensions)
        for i in range(len(self.dimensions)):
            if self.types[i] == Type.numerical:
                crossindex[self.dimensions[i]] = self.R[self.dimensions[i]+'_bin'].astype(int)
        crossindex['idx'] = crossindex[self.dimensions[0]].index.astype(int)
        for i in range(len(self.dimensions)-1, -1, -1):
            # if self.types[i] == Type.numerical:
                # crossindex[self.dimensions[i]] = self.R[self.dimensions[i]+'_bin'].astype(int)
            # groupby.append(self.dimensions[i])
            if i == len(self.dimensions)-1:
                tmp = crossindex.groupby(groupby).agg(
                    interval = ('idx', calInterval),
                ).reset_index()
                crossindex = crossindex.merge(tmp, how='right', left_on=groupby, right_on=groupby)
                groupby.remove(self.dimensions[i])
            else:
                tmp = crossindex.groupby(groupby).agg(
                    interval = ('interval'+str(len(self.dimensions)-1), lambda x: x.iloc[0].split(',')[0]+','+x.iloc[-1].split(',')[1])
                ).reset_index()
                crossindex = crossindex.merge(tmp, how='left', left_on=groupby, right_on=groupby)
                groupby.remove(self.dimensions[i])
            crossindex.rename(columns={'interval':'interval'+str(i)}, inplace=True)
            self.pbar.update(len(crossindex))
        
        crossindex.drop(columns=['idx'], inplace=True)
        crossindex.drop_duplicates(inplace=True)
        self.R.drop(columns=to_drop, inplace=True)

        self.pbar.close()
        self.ready = True
        self.index = crossindex
        end = time.time()
        print('build time:' + str(end - start))
        return end-start

    def query_csv(self, query, search_space=None, index=0):
        res = search_space
        if res is None:
            res = self.index
        idx = index
        start = time.time()
        if self.use_omnisci:
            for i in range(idx, len(query.wheres)):
                filtered_sql, flag = query.get_query_index_sql(i, self.name)
                if flag:
                    print("query index sql: " + filtered_sql)
                    try:
                        res = self.omnisci.get_df(filtered_sql)
                    except:
                        res = pd.DateFrame(index=self.index.index)
                    query.cache[i] = res
                    idx = i
        else:
            for i in range(idx, len(query.wheres)):
                predicte = query.wheres[i]
                if predicte.value is None:
                    continue
                if self.types[i] == Type.numerical:
                    res = predicte.match_csv(res, self.dimensions[i], self.offset[i], self.bin_width[i])
                else:
                    res = predicte.match_csv(res, self.dimensions[i])
                # print(len(res))
                query.cache[i] = res
                idx = i
        print('search time: '+str(time.time()-start))

        if len(res)>1000000: # drop query when overwhelming
            return "dropped"
        start = time.time()
        idx = len(query.wheres)-1 if idx>=len(query.wheres) else idx
        idx = self.dimensions.index(query.groupby) if self.dimensions.index(query.groupby)>idx else idx
        interested = res[[query.groupby, 'interval'+str(idx)]].drop_duplicates(['interval'+str(idx)])
        if query.agg == 'COUNT':
            interested['interval'+str(idx)] = interested['interval'+str(idx)].apply(lambda x:1+int(x.split(',')[1])-int(x.split(',')[0]))
            xy_df = interested.groupby(query.groupby)['interval'+str(idx)].agg('sum')
            print('count computation time:' +str(time.time()-start))
            
            start = time.time()
            for i,value in xy_df.items():
                query.result.x_data.append(str(i))
                query.result.y_data.append(value)
            print('count collection time:' +str(time.time()-start))
        else:
            # interested = interested.groupby(query.groupby)['interval'+str(idx)].apply(list)
            # print('groupby time:'+str(time.time()-start))
            # print(len(interested))
            # for i,value in interested.items(): # bottleneck
            #     xyMap[str(i)] = [Interval(v.split(',')[0], v.split(',')[1]) for v in value] 
            valid_id = []
            for v in list(interested['interval'+str(idx)]):
                valid_id.extend(list(range(int(v.split(',')[0]), int(v.split(',')[1])+1)))
            print('intervals collection time:' + str(time.time() - start))

            start = time.time()
            # for key in sorted(xyMap.keys()):
            #     query.result.x_data.append(key)
            #     query.result.y_intervals.append(xyMap[key])
            query.compute(valid_id)
            print('compute time: '+str(time.time()-start))
        return query.result.output_xy()

    def query(self, query):
        # 记录当前循环中符合之前 where 条件即有效的 DimensionSet
        validDSs = []
        xyMap = defaultdict(lambda: [])
        n = query.where_n
        start_idx = query.get_start_condition()
        if start_idx == 0:
            start = DimensionSet('root', -1, 'all', None)
            start.subSet = self.dimensionSetLayers[0]
        else:
            start = self.dimensionSetLayers[start_idx-1]
        validDSs.extend(start)

        # no wheres
        if n == 0:
            for i in range(len(self.dimensions)):
                if self.dimensions[i] == query.groupby:
                    for ds in self.dimensionSetLayers[i]:
                        xyMap[ds.value].append(ds.interval)
        else:
            for i in range(start_idx ,len(self.dimensions)):
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
                        # tmpDS.extend(where.binary_match(ds.subSet))
                validDSs = tmpDS
                query.cache[i] = tmpDS

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

    def backward_query_csv(self, query, other):
        conditions = other.wheres
        idx, flag = query.get_deepest_overlapped_idx(conditions)
        if idx not in query.cache.keys():
            res = self.query_csv(other)
            return False, res
        if flag:
            for i in range(idx):
                if i in query.cache.keys():
                    other.cache[i] = query.cache[i]
            res = self.query_csv(other, query.cache[idx], idx)
        else:
            for i in range(idx+1):
                if i in query.cache.keys():
                    other.cache[i] = query.cache[i]
            res = self.query_csv(other, other.cache[idx], idx+1)
        return True, res

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

    def backward_query2(self, query, other):
        conditions = other.wheres
        idx, flag = query.get_deepest_overlapped_idx(conditions)
        if idx not in query.cache.keys():
            return self.query(other)

        n = other.where_n
        xyMap = defaultdict(lambda: [])
        validDSs = query.cache[idx]
        # 该层的实际 validDSs 需要再用新谓词筛选一次
        if flag:
            tmp = []
            for ds in validDSs:
                if conditions[idx].match(ds):
                    tmp.append(ds)
            validDSs = tmp
        # 之前重叠的查询空间记录下来
        for i in range(idx+1):
            if i in query.cache.keys():
                other.cache[i] = query.cache[i]
                n -= 1
        for i in range(idx+1, len(conditions)):
            tmpDS = []
            where = conditions[i]
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
            other.cache[i] = validDSs

            # 已经空集则返回
            if len(validDSs) == 0:
                return other.result.output_xy()
            n -= 1
            if n == 0:
                break

        # group by
        valid_dimension_i = self.dimensions.index(validDSs[0].dimension)
        groupby_i = self.dimensions.index(other.groupby)
        if valid_dimension_i == groupby_i:
            for ds in validDSs:
                xyMap[ds.value].append(ds.interval)
        elif valid_dimension_i > groupby_i:
            for ds in validDSs:
                p = ds.find_parent(other.groupby)
                if p is not None:
                    xyMap[p.value].append(ds.interval)
        else:
            while validDSs[0].dimension != other.groupby:
                tmpDS = []
                for ds in validDSs:
                    tmpDS.extend(ds.subSet)
                validDSs = tmpDS
            for ds in validDSs:
                xyMap[ds.value].append(ds.interval)

        other.validDSs = validDSs
        for key in sorted(xyMap.keys()):
            other.result.x_data.append(key)
            other.result.y_intervals.append(xyMap[key])
        other.compute()
        return other.result.output_xy()

    def output(self):
        for ds in self.dimensionSetLayers[0]:
            ds.output()

    def execute_query(self, sql):
        q = Query(cube=crossindex)
        q.parse(sql)

        start = time.time()
        crossindex.query(q)
        end = time.time()

        q.result.pretty_output()
        print('direct query time:' + str(end - start))

    def calculate_cardinality(self, path, bw, os, delimiter):
        self.R = pd.read_csv(path, encoding='utf-8', delimiter=delimiter)
        print('csv read fininshed.')
        cardinalities = {}
        bin_width = {}
        offset = {}
        for i in range(len(self.dimensions)):
            d = self.dimensions[i]
            if self.types[i] == Type.spatial:
                geohash = self.R.apply(lambda x: geohash2.encode(x[d[1]], x[d[0]], 8), axis=1)
                key = ",".join(d)
                if bw is not None and bw[i]!= -1:
                    bin_width[key] = bw[i]
                else:
                    bin_width[key] = 1
                cardinality = len(geohash.unique())
                cardinalities[key] = cardinality/bin_width[key]
            elif self.types[i] == Type.categorical:
                if bw is not None and bw[i]!= -1:
                    bin_width[d] = bw[i]
                else:
                    bin_width[d] = 1
                cardinality = len(geohash.unique())
                cardinalities[d] = cardinality/bin_width[d]
            elif self.types[i] == Type.numerical:
                if os is not None and os[i]!=-1:
                    offset[d] = os[i]
                else:
                    offset[d] = self.R[d].min()
                total_width = self.R[d].max()-offset[d]
                print(d + ' total width:', total_width)
                if bw is not None and bw[i]!= -1:
                    bin_width[d] = bw[i]
                else:
                    if total_width <= 100:
                        bin_width[d] = 2
                    elif total_width <=1000:
                        bin_width[d] = 20
                    elif total_width <= 10000:
                        bin_width[d] = 200
                    else:
                        bin_width[d] = 1000
                
                cardinalities[d] = total_width//bin_width[d]+1
                
        return {'cardinalities':cardinalities, 'bin_width':bin_width, 'offset':offset}

    def adjust_by_cardinality(self, path, bw, offset, delimiter, reverse=False, rd=False):
        meta_info = self.calculate_cardinality(path, bw, offset, delimiter)
        cardinalities, bin_width, offset = meta_info['cardinalities'], meta_info['bin_width'], meta_info['offset']
        if rd:
            cardinalities = list(cardinalities.items())
            random.shuffle(cardinalities)
        else:
            cardinalities = sorted(cardinalities.items(), key=lambda kv: (kv[1], kv[0]), reverse=reverse)
        print(cardinalities)

        tmp_d = []
        tmp_t = []
        for c in cardinalities:
            key = c[0]
            if key.find(',') != -1:
                key = key.split(',')
            tmp_d.append(key)
            tmp_t.append(self.types[self.dimensions.index(key)])
            self.bin_count.append(c[1])
            self.bin_width.append(bin_width[key])
            if key in offset.keys():
                self.offset.append(offset[key])
            else:
                self.offset.append(-1)
        self.dimensions = tmp_d
        self.types = tmp_t


def execute_direct_query(crossindex, sql):
    q = Query(cube=crossindex)
    q.parse(sql)

    start = time.time()
    crossindex.query_csv(q)
    end = time.time()

    q.result.pretty_output()
    print('direct query time:' + str(end - start))

    return q

def execute_backward_query(crossindex, sql, new_sql):
    cached_q = Query(cube=crossindex)
    cached_q.parse(sql)
    crossindex.query_csv(cached_q)

    q = Query(cube=crossindex)
    q.parse(new_sql)
    start = time.time()
    flag,_ = crossindex.backward_query_csv(cached_q, q)
    end = time.time()
    q.result.pretty_output()
    print(flag)
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
    argparser.add_argument('--bin-width', dest='bin_width', nargs='+', type=float, help='bin width of dimensions', default=None)
    argparser.add_argument('--offset', dest='offset', nargs='+', type=int, help='offset of numerical dimensions', default=None)

    argparser.add_argument('--delimiter', dest='delimiter', help='delimiter of csv file', default=',')
    argparser.add_argument('--single', dest='single', action='store_true', help='whether to build in single thread', default=False)
    argparser.add_argument('--csv', dest='csv', action='store_true', help="whether to use crossindex in csv form", default=True)
    argparser.add_argument('--omnisci', dest='omnisci', action='store_true', help="whether to use omnisci_conf to speed up", default=False)

    # some options for different dimension type
    # temporal
    argparser.add_argument('--by', dest='by', help='group by year,month,day etc. for temporal', default='DAY')
    argparser.add_argument('--granularity', dest='granularity', type=int, help='granularity for temporal', default=1)
    argparser.add_argument('--date-format', dest='date_format', help='date format for temporal', default='%Y-%m-%d %H:%M:%S')
    # spatial
    argparser.add_argument('--hash-length', dest='hash_length', type=int, help='geohash code length for spatial', default=8)

    args = vars(argparser.parse_args())

    # combine dimension args if its tpye is spatial
    for i in range(len(args['types'])):
        if args['types'][i] == 'spatial':
            args['dimensions'][i] = args['dimensions'][i].split(',')

    # initialization
    crossindex =CrossIndex(args['name'], args['dimensions'], args['types'], args['omnisci'])
    if args['csv']:
        if os.path.exists(args['cube_dir'] + args['name'] + '_csv.csv'):
            crossindex.load_csv(args)
        else:
            crossindex.adjust_by_cardinality(args['input_dir'], args['bin_width'], args['offset'],  args['delimiter'], reverse=False)
            crossindex.build_csv()
            crossindex.save_csv(args)
    else:
        if os.path.exists(args['cube_dir'] + args['name'] + '.csv'):
            crossindex.load(args['cube_dir'], args['name'])
        else:
            if args['single'] == True:
                crossindex.adjust_by_cardinality(args['input_dir'], args['delimiter'], args['bin_width'], args['offset'], reverse=False)
                crossindex.build(args['input_dir'], args['delimiter'], args)
            else:
                crossindex.adjust_by_cardinality(args['input_dir'], args['delimiter'], args['bin_width'], args['offset'], reverse=False)
                crossindex.build_parallel(args['input_dir'], args['delimiter'], args)
            crossindex.save(args['cube_dir'])

    sql = "SELECT FLOOR(Release_Date/30541006451.612904) AS bin_Release_Date,  COUNT(*) as count FROM movies WHERE (Running_Time_min >= 78.74285714285715 AND Running_Time_min < 93.60000000000001  AND Production_Budget >= 3.428571428571434 AND Production_Budget < 41.14285714285714) GROUP BY bin_Release_Date"
    backward_sql = "SELECT FLOOR(Release_Date/30541006451.612904) AS bin_Release_Date,  COUNT(*) as count FROM movies WHERE (Running_Time_min >= 78.74285714285715 AND Running_Time_min < 93.60000000000001  AND Production_Budget >= 3.428571428571434 AND Production_Budget < 41.14285714285714) GROUP BY bin_Release_Date"
    execute_direct_query(crossindex, backward_sql)
    execute_backward_query(crossindex, sql, backward_sql)


''' 
====EXPERIMENT ARGS====
weather args:
    --input-dir data/Weather/dataset_weather_1M_fixed.csv --cube-dir cube/Weather/ --name weather_1M 
    --dimensions RECORD_DATE ELEVATION LONGITUDE TEMP_MIN TEMP_MAX SNOW PRECIPITATION WIND 
    --types numerical numerical numerical numerical numerical numerical numerical numerical 
    --bin-width 163725000 200 20 5 5 50 0.5 0.5 --offset 1325307600000 -200 -160 -10 -10 0 0 0 --csv
    sql = "SELECT FLOOR(ELEVATION/200) AS bin_ELEVATION,  COUNT(*) as count FROM weather WHERE (TEMP_MIN >= 24.457142857142863 AND TEMP_MIN < 29.6) GROUP BY bin_ELEVATION"
    backward_sql = "SELECT FLOOR(ELEVATION/200) AS bin_ELEVATION,  COUNT(*) as count FROM weather WHERE (TEMP_MIN >= 24.457142857142863 AND TEMP_MIN < 29.6) GROUP BY bin_ELEVATION"

movies args:
    --input-dir data/Movies/dataset_movies_1M_fixed.csv --cube-dir cube/Movies/ --name movies_1M 
    --dimensions Release_Date IMDB_Rating Rotten_Tomatoes_Rating Production_Budget Running_Time_min US_DVD_Sales US_Gross Worldwide_Gross 
    --types numerical numerical numerical numerical numerical numerical numerical numerical 
    --bin-width 30541006451.612904 0.5 5 2 20 2 5 5 --offset 315550800000 0 0 0 0 0 0 0 --csv
    sql = "SELECT FLOOR(US_Gross/200) AS bin_US_Gross,  COUNT(*) as count FROM movies WHERE (US_Gross >= 53.14285714285714 AND US_Gross < 72) GROUP BY bin_US_Gross"
    backward_sql = "SELECT FLOOR(US_Gross/200) AS bin_US_Gross,  COUNT(*) as count FROM movies WHERE (US_Gross >= 53.14285714285714 AND US_Gross < 72) GROUP BY bin_US_Gross"

flights_covid_10M.csv args:
    --input-dir data/Flights_covid/flights_covid.csv --cube-dir cube/Flights_covid/ --name flight_covid_10M 
    --dimensions callsign icao24 registration typecode origin destination day --types categorical categorical categorical categorical categorical categorical temporal
    sql = "SELECT day, COUNT(origin) FROM flights_covid WHERE day BETWEEN '2019-05-05' and '2019-06-05' AND origin = 'KMSP' GROUP BY day"

bike_10M.csv args:
    --input-dir data/Bikes/Divvy_Trips.csv --name bike --dimensions geohash USER_TYPE START_TIME -types spatial categorical temporal
    sql = "SELECT USER_TYPE AS bin_USER_TYPE,  COUNT(*) as count FROM tbl_bike GROUP BY bin_USER_TYPE"
    
flights_numerical_1M.csv args:
    --input-dir data/Flights/dataset_flights_1M.csv --cube-dir cube/Flights/ --name flights_numerical_1M --dimensions DISTANCE AIR_TIME ARR_TIME DEP_TIME ARR_DELAY DEP_DELAY --types numerical numerical numerical numerical numerical numerical 
    sql = "SELECT FLOOR(ARR_TIME/1) AS bin_ARR_TIME,  COUNT(*) as count FROM flights WHERE (AIR_TIME >= 150 AND AIR_TIME < 500 AND DISTANCE >= 0 AND DISTANCE < 1000) GROUP BY bin_ARR_TIME"
    execute_direct_query(crossindex, sql)

flights_1M_categorical.csv args:
    --input-dir data/Flights/dataset_flights_1M.csv --cube-dir cube/Flights/ --name flights_categorical_1M --dimensions AIR_TIME ARR_DELAY ARR_TIME DEP_DELAY DEP_TIME DISTANCE --types categorical categorical categorical categorical categorical categorical
    sql = "SELECT FLOOR(DEP_TIME/1) AS bin_DEP_TIME,  COUNT(*) as count FROM flights WHERE (DISTANCE >= 985.7142857142858 AND DISTANCE < 1200 AND AIR_TIME >= 122.85714285714286 AND AIR_TIME < 500) GROUP BY bin_DEP_TIME"
    backward_sql = "SELECT FLOOR(DEP_TIME/1) AS bin_DEP_TIME,  COUNT(*) as count FROM flights WHERE (DISTANCE >= 985.7142857142858 AND DISTANCE < 1200) GROUP BY bin_DEP_TIME"
    execute_direct_query(crossindex, sql)
    execute_backward_query(crossindex, backward_sql, [Condition('AIR_TIME', [122.85714285714286, 500], Type.categorical)])

traffic_categorical.csv args:
    --input-dir data/traffic.csv --name traffic_categorical --dimensions link_id vehicle_num velocity_ave time --types categorical categorical categorical categorical --delimiter \t
    sql = "SELECT COUNT(vehicle_num) from traffic WHERE vehicle_num >= 1 AND vehicle_num < 2 AND velocity_ave >= 4 AND velocity_ave < 8 GROUP BY time"
    backward_sql = "SELECT COUNT(vehicle_num) from traffic WHERE vehicle_num >= 1 AND vehicle_num < 2 AND velocity_ave >= 4 AND velocity_ave < 5 AND time >= 640 AND time < 674 GROUP BY time"
    execute_direct_query(crossindex, backward_sql)
    execute_backward_query(crossindex, sql, backward_sql)
    
traffic_numerical.csv args:
    --input-dir data/traffic.csv --name traffic_numerical --dimensions link_id vehicle_num velocity_ave time --types categorical numerical numerical categorical --delimiter \t
    sql = "SELECT COUNT(vehicle_num) from traffic WHERE velocity_ave >= 4 AND velocity_ave < 8 GROUP BY time"

traffic.csv args:
    --name traffic --dimensions link_id time --types categorical categorical --delimiter \t
    sql = "SELECT COUNT(vehicle_num) from traffic WHERE time = '651'  GROUP BY link_id"

trace.csv args:
    --input-dir data/trace.csv --name trace --dimensions "lng,lat" link_id vehicle_id timestep --types spatial categorical categorical categorical
    direct_sql = "SELECT COUNT(vehicle_length) from trace WHERE geohash='wtw3sm'  AND link_id IN ['152C909GV90152CL09GVD00', '152D309GVT0152CJ09GVM00'] AND timestep >= 39654.4 AND timestep < 39800.4 GROUP BY geohash"
    backward_sql = "SELECT COUNT(vehicle_length) from trace WHERE geohash='wtw3sm' AND timestep >= 39654.4 AND timestep < 39800.4 GROUP BY geohash"
    execute_direct_query(crossindex, direct_sql)
    execute_backward_query(crossindex, backward_sql,[Condition('link_id', ['152C909GV90152CL09GVD00', '152D309GVT0152CJ09GVM00'], Type.categorical)])
    
myshop_temporal.csv args:
    --input-dir data/myshop.csv --name myshop_temporal --dimensions category itemname gender nationality date --types categorical categorical categorical categorical temporal
    sql = "SELECT COUNT(quantity) from myshop WHERE gender = '女' AND date BETWEEN '2019-05-05' and '2020-05-05' GROUP BY category"
'''
