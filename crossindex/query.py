import geohash2
import pandas as pd

from dimension import Interval
from dimension import DimensionSet
from resultset import ResultSet
from calc_type import Type


# where 条件
class Condition(object):
    def __init__(self, dimension, value=None, type=None):
        self.dimension = dimension
        self.value = value
        self.type = type

    def binary_match(self, subset):
        res = []
        condition = self.value if self.value.__class__ == list else [self.value]

        def binary_search1(l, r, subset, target):
            while l <= r:
                mid = int((l+r)/2)
                if subset[mid].value == target:
                    return mid
                elif subset[mid].value > target:
                    r = mid-1
                else:
                    l = mid+1
            return l
        
        def binary_search2(l, r, subset, target):
            while l <= r:
                mid = int((l+r)/2)
                if float(subset[mid].value) == target:
                    return mid
                elif float(subset[mid].value) > target:
                    r = mid-1
                else:
                    l = mid+1
            return l

        if self.type == Type.categorical:
            if type(condition[0]) is float:
                idx = binary_search2(0, len(subset)-1, subset, condition[0])
                for i in range(idx, len(subset)):
                    if condition[0] <= float(subset[i].value) < condition[1]:
                        res.append(subset[i])
                    else:
                        break # float(subset[i].value) >= condition[1]
            else:
                idx = 0
                for target in sorted(condition):
                    idx = binary_search1(idx, len(subset)-1, subset, target)
                    if idx < len(subset) and subset[idx].value == target:
                        res.append(subset[idx])
        elif self.type == Type.temporal:
            idx = binary_search1(0, len(subset)-1, subset, condition[0])
            for i in range(idx, len(subset)):
                if condition[0] <= subset[i].value <= condition[1]:
                    res.append(subset[i])
                else:
                    break
        elif self.type == Type.spatial:
            idx = binary_search1(0, len(subset)-1, subset, condition[0])
            for i in range(idx, len(subset)):
                if subset[i].value.find(condition[0])>=0:
                    res.append(subset[i])
        elif self.type == Type.numerical:
            # todo
            print('numerical')
        return res

    def match_csv(self, df, dimension):
        condition = self.value if self.value.__class__ == list else [self.value]
        if self.type == Type.categorical:
            if type(condition[0]) is float:
                return df[(df[dimension]>=condition[0])&(df[dimension]<condition[1])]
            return df[df[dimension].isin(condition)]
        elif self.type == Type.temporal:
            return df[(df[dimension]>=condition[0])&(df[dimension]<=condition[1])]
        elif self.type == Type.numerical:
            return df[(df[dimension]>=condition[0])&(df[dimension]<condition[1])]

    def match(self, ds, R=None):
        value = ds.value
        condition = self.value if self.value.__class__ == list else [self.value]

        if self.type == Type.categorical:
            if type(condition[0]) is float:
                return condition[0] <= float(value) < condition[1]
            return value in condition
        elif self.type == Type.temporal:
            return condition[0] <= value <= condition[1]
        elif self.type == Type.spatial:
            if len(value) > len(condition[0]):
                return value.find(condition[0]) != -1
            elif len(value) == len((condition[0])):
                return value == condition[0]
        elif self.type == Type.numerical:
            # bin_boundary: (x, y]
            # condition: [x, y)
            bin_boundary = [float(x.strip()) for x in value[1:-1].split(',')]
            if bin_boundary[0] >= condition[1] or bin_boundary[1] < condition[0]:
                return False, None
            if bin_boundary[0] >= condition[0] and bin_boundary[1] < condition[1]:
                return True, None
            if bin_boundary[0] < condition[0] <= bin_boundary[1] < condition[1]:
                r = R.iloc[ds.interval.begin:ds.interval.end + 1, :]
                r = r[r[ds.dimension] >= condition[0]]
                if len(r) == 0:
                    return False, None
                idx = r.index.tolist()
                new_ds = DimensionSet(ds.dimension, Type.numerical, '(' + str(condition[0]) + ',' + value.split(',')[1],
                                      Interval(idx[0], idx[-1]), ds.parent)
                new_ds.subSet = ds.subSet
                self.adjust_subset(new_ds)
                return True, new_ds
            if condition[0] <= bin_boundary[0] < condition[1] <= bin_boundary[1]:
                r = R.iloc[ds.interval.begin:ds.interval.end + 1, :]
                r = r[r[ds.dimension] < condition[1]]
                if len(r) == 0:
                    return False, None
                idx = r.index.tolist()
                new_ds = DimensionSet(ds.dimension, Type.numerical, value.split(',')[0] + ',' + str(condition[1]) + ')',
                                      Interval(idx[0], idx[-1]), ds.parent)
                new_ds.subSet = ds.subSet
                self.adjust_subset(new_ds)
                return True, new_ds
            if bin_boundary[0] < condition[0] <= condition[1] <= bin_boundary[1]:
                r = R.iloc[ds.interval.begin:ds.interval.end + 1, :]
                r = r[(condition[0] <= r[ds.dimension]) & (r[ds.dimension] < condition[1])]
                if len(r) == 0:
                    return False, None
                idx = r.index.tolist()
                new_ds = DimensionSet(ds.dimension, Type.numerical, '[' + str(condition[0]) + ',' + str(condition[1]) + ')',
                                      Interval(idx[0], idx[-1]), ds.parent)
                new_ds.subSet = ds.subSet
                self.adjust_subset(new_ds)
                return True, new_ds

    def adjust_subset(self, ds):
        new_subSet = []
        if ds.subSet[0].interval.begin >= ds.interval.begin and ds.subSet[-1].interval.end <= ds.interval.end:
            return
        for sub in ds.subSet:
            # 原ds下的sub不再属于new_ds
            if sub.interval.end < ds.interval.begin:
                continue
            if sub.interval.begin > ds.interval.end:
                break

            # 原ds下的部分sub属于new_ds, 调整相应sub的interval, 并递归调整sub的subSet
            if sub.interval.begin < ds.interval.begin <= sub.interval.end <= ds.interval.end:
                new_sub = DimensionSet(sub.dimension, sub.type, sub.value, Interval(ds.interval.begin, sub.interval.end), ds)
                new_sub.subSet = sub.subSet
                self.adjust_subset(new_sub)
            elif ds.interval.begin <= sub.interval.begin <= ds.interval.end < sub.interval.end:
                new_sub = DimensionSet(sub.dimension, sub.type, sub.value, Interval(sub.interval.begin, ds.interval.end), ds)
                new_sub.subSet = sub.subSet
                self.adjust_subset(new_sub)

            # sub 正好属于new_ds, 无需调整
            else:
                new_sub = sub

            new_subSet.append(new_sub)
        ds.subSet = new_subSet


aggregation = {'CNT': 'COUNT',
               'AVG': 'AVG',
               'SUM': 'SUM'}


class Query(object):
    def __init__(self, measure='', agg='', groupby='', cube=None):
        self.measure = measure
        self.agg = agg
        self.groupby = groupby
        self.result = ResultSet(groupby, agg + '(' + measure + ')')
        self.cube = cube
        self.wheres = []
        self.where_n = 0
        for d in self.cube.dimensions:
            c = Condition(dimension=d)
            self.wheres.append(c)
        self.validDSs = []
        self.cache = {}

    def set_cube(self, cube):
        self.cube = cube
        self.wheres = []
        self.where_n = 0
        for d in self.cube.dimensions:
            c = Condition(dimension=d)
            self.wheres.append(c)

    def parse(self, sql):
        # projection
        projection = sql[sql.find("SELECT") + 6: sql.rfind("FROM")].strip()
        for column in projection.split(','):
            column = column.strip()
            if column.startswith("FLOOR"):
                continue
            # parse measure and agg
            b = column.find('(') + 1
            e = column.find(')')
            if sql.find(aggregation.get('CNT')) != -1:
                self.agg = aggregation.get('CNT')
            elif sql.find(aggregation.get('AVG')) != -1:
                self.agg = aggregation.get('AVG')
            elif sql.find(aggregation.get('SUM')) != -1:
                self.agg = aggregation.get('SUM')
            self.measure = column[b:e]

        # parse where conditions
        conditions = self.parse_conditions(sql)
        for c in conditions:
            self.wheres[self.cube.dimensions.index(c.dimension)] = c
        self.where_n = len(conditions)

        # parse group by
        if sql.find('GROUP') != -1:
            groupby = sql[sql.find('GROUP BY') + 9:].strip()
            if groupby.startswith('bin'):
                groupby = groupby[4:]
            self.groupby = groupby
        self.result = ResultSet(self.groupby, self.agg + '(' + self.measure + ')')

    def parse_conditions(self, sql):
        # parse where conditions
        conditions = []
        if sql.find('WHERE') != -1:
            wheres = sql[sql.find('WHERE') + 6:sql.find('GROUP')].split('AND')
            for where in wheres:
                # >= and <
                if where.find('>=') != -1:
                    dimension = where.split('>=')[0].strip().strip('(')
                    value = where.split('>=')[1].strip().strip(')')
                    d_type = self.cube.types[self.cube.dimensions.index(dimension)]
                    condition = Condition(dimension, [float(value)], d_type)
                elif where.find('<') != -1:
                    dimension = where.split('<')[0].strip().strip('(')
                    value = where.split('<')[1].strip().strip(')')
                    for w in conditions:
                        if w.dimension == dimension:
                            w.value.append(float(value))
                            break
                # =
                elif where.find('=') != -1:
                    dimension = where.split('=')[0].strip()
                    value = where.split('=')[1].replace('\'', '').strip()
                    t = Type.categorical
                    # spatial
                    if dimension == 'geohash':
                        t = Type.spatial
                    condition = Condition(dimension, value, t)
                # in
                elif where.find('IN') != -1:
                    dimension = where.split('IN')[0].strip().strip('(')
                    value = where.split('IN')[1].replace('\'', '').strip().strip(')')[1:-1].replace(' ', '').split(',')
                    condition = Condition(dimension, value, Type.categorical)
                # between
                elif where.find('BETWEEN') != -1:
                    dimension = where.split('BETWEEN')[0].strip().strip('(')
                    value = [s.strip() for s in where.split('BETWEEN')[1].replace('\'', '').split('and')]
                    condition = Condition(dimension, value, Type.temporal)

                conditions.append(condition)
        return list(set(conditions))

    def compute(self):
        if self.agg == aggregation.get('CNT'):
            for i in range(len(self.result.x_data)):
                count = 0
                intervals = self.result.y_intervals[i]
                for interval in intervals:
                    count = count + interval.count
                self.result.y_data.append(count)
        elif self.agg == aggregation.get('AVG'):
            for i in range(len(self.result.x_data)):
                sum = 0
                count = 0
                intervals = self.result.y_intervals[i]
                for interval in intervals:
                    sum = sum + self.cube.R.iloc[interval.begin:interval.end + 1][self.measure].sum()
                    count = count + interval.count
                average = sum / count
                self.result.y_data.append(average)
        elif self.agg == aggregation.get('SUM'):
            for i in range(len(self.result.x_data)):
                sum = 0
                intervals = self.result.y_intervals[i]
                for interval in intervals:
                    sum = sum + self.cube.R.iloc[interval.begin:interval.end + 1][self.measure].sum()
                self.result.y_data.append(sum)

    def get_geo_result(self, limit):
        max = 0
        data = []
        if limit is None:
            for i in range(len(self.result.x_data)):
                intervals = self.result.y_intervals[i]
                point = {'lat': self.cube.R.iloc[intervals[0].begin]['lat'],
                         'lng': self.cube.R.iloc[intervals[0].begin]['lng'],
                         'count': int(self.result.y_data[i])}
                if self.result.y_data[i] > max:
                    max = int(self.result.y_data[i])
                data.append(point)
        else:
            # tmp_x = self.result.x_data
            # tmp_y = self.result.y_intervals
            for i in range(len(self.result.x_data)):
                intervals = self.result.y_intervals[i]
                valid_count = 0
                valid_row = pd.DataFrame()
                # tmp_intervals = []
                for interval in intervals:
                    tmp = self.cube.R.iloc[interval.begin:interval.end + 1]
                    valid_row = tmp[((limit['bottom'] <= tmp['lat']) & (tmp['lat'] <= limit['top'])) &
                                    ((limit['left'] <= tmp['lng']) & (tmp['lng'] <= limit['right']))]
                    valid_count = valid_count + valid_row[self.measure].count()
                    # tmp_intervals.append(valid_row)
                if valid_count == 0:
                    # tmp_x.remove(self.result.x_data[i])
                    # tmp_y.remove(intervals)
                    continue
                # tmp_y[i] = tmp_intervals
                lat, lng = valid_row.iloc[0]['lat'], valid_row.iloc[0]['lng']
                point = {'lat': lat,
                         'lng': lng,
                         'count': int(valid_count)}
                if valid_count > max:
                    max = int(valid_count)
                data.append(point)
            # self.result.x_data = tmp_x
            # self.result.y_intervals = tmp_y
            # self.compute()
        return data, max

    def add_condition(self, condition):
        index = self.cube.dimensions.index(condition.dimension)
        if self.wheres[index].value is None:
            self.where_n = self.where_n + 1
        self.wheres[index] = condition

    def get_start_condition(self):
        if self.where_n == 0:
            return -1
        for i,w in enumerate(self.wheres):
            if w.value is not None:
                return i

    def clear_conditions(self):
        self.wheres = []
        for d in self.cube.dimensions:
            c = Condition(dimension=d)
            self.wheres.append(c)
        self.where_n = 0

    # 返回最后一个重叠 dimension 的下标
    # 第二个返回参数代表最后一个 dimension 的新谓词是否在原谓词上进行缩小
    def get_deepest_overlapped_idx(self, new_conditions):
        for i in range(len(new_conditions)):
            old_condition = self.wheres[i]
            new_condition = new_conditions[i]
            if old_condition.value is None and new_condition.value is None:
                continue
            elif old_condition.value is None or new_condition.value is None:
                return i-1, False
            elif new_condition.value != old_condition.value:
                flag = False
                if old_condition.type == Type.categorical:
                    if type(old_condition.value[0]) is float and \
                            old_condition.value[0] <= new_condition.value[0] \
                            and new_condition.value[1] <= old_condition.value[1]:
                        flag = True
                    elif set(new_condition.value) <= set(old_condition.value):
                        flag = True
                elif old_condition.type == Type.temporal:
                    if old_condition.value[0] <= new_condition.value[0] and new_condition.value[1] <= old_condition.value[1]:
                        flag = True
                elif old_condition.type == Type.spatial:
                    if old_condition.value.find(new_condition.value) > 0:
                        flag = True
                elif old_condition.type == Type.numerical:
                    if old_condition.value[0] <= new_condition.value[0] and new_condition.value[1] <= old_condition.value[1]:
                        flag = True

                if flag:
                    return i, True
                return i-1, False
        return len(new_conditions)-1, False

    def is_backward_query(self, sql):
        if self.where_n == 0:
            return False
        conditions = self.parse_conditions(sql)
        for c in conditions:
            old_condition = self.wheres[self.cube.dimensions.index(c.dimension)]
            if old_condition.value is None:
                continue

            # compare whether can backward query
            if old_condition.type == Type.categorical:
                if type(old_condition.value[0]) is float:
                    if c.value[0] < old_condition.value[0] or c.value[1] > old_condition.value[1]:
                        return False
                else:
                    for v in c.value:
                        if not v in old_condition.value:
                            return False
            elif old_condition.type == Type.temporal:
                if c.value[0] < old_condition.value[0] or c.value[1] > old_condition.value[1]:
                    return False
            elif old_condition.type == Type.spatial:
                if len(c.value) < len(old_condition.value):
                    return False
                elif len(c.value) > len(old_condition.value):
                    if old_condition.value.find(c.value) == -1:
                        return False
            elif old_condition.type == Type.numerical:
                if c.value[0] < old_condition.value[0] or c.value[1] > old_condition.value[1]:
                    return False
        return True

    def clear(self):
        self.result = ResultSet(self.groupby, self.agg + '(' + self.measure + ')')

if __name__ == '__main__':
    q = Query()
    q.parse("SELECT USER_TYPE AS bin_USER_TYPE,  AVG(FROM_LONGITUDE) as average_FROM_LONGITUDE FROM tbl_bike GROUP BY bin_USER_TYPE")