import geohash2
import pandas as pd

from resultset import ResultSet
from type import Type


# where 条件
class Condition(object):
    def __init__(self, dimension, value=None, type=None):
        self.dimension = dimension
        self.value = value
        self.type = type

    def match(self, value):
        condition = self.value if self.value.__class__ == list else [self.value]
        if self.type == Type.categorical:
            return value in condition
        elif self.type == Type.temporal:
            return condition[0] <= value <= condition[1]
        elif self.type == Type.spatial:
            if len(value) > len(condition[0]):
                return value.find(condition[0]) != -1
            elif len(value) == len((condition[0])):
                return value == condition[0]


aggregation = {'CNT': 'COUNT',
               'AVG': 'AVG',
               'SUM': 'SUM'}


class Query(object):
    def __init__(self, measure, agg, groupby, cube):
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

    def set_cube(self, cube):
        self.cube = cube
        self.wheres = []
        self.where_n = 0
        for d in self.cube.dimensions:
            c = Condition(dimension=d)
            self.wheres.append(c)

    def parse(self, sql):
        # parse measure and agg
        b = sql.find('(') + 1
        e = sql.find(')')
        if sql.find(aggregation.get('CNT')) != -1:
            self.agg = aggregation.get('CNT')
        elif sql.find(aggregation.get('AVG')) != -1:
            self.agg = aggregation.get('AVG')
        elif sql.find(aggregation.get('SUM')) != -1:
            self.agg = aggregation.get('SUM')
        self.measure = sql[b:e]

        # parse where conditions
        if sql.find('WHERE') != -1:
            wheres = sql[sql.find('WHERE') + 6:sql.find('GROUP')].split('AND')
            for where in wheres:
                # =
                if where.find('=') != -1:
                    dimension = where.split('=')[0].strip()
                    value = where.split('=')[1].replace('\'', '').strip()
                    t = Type.categorical
                    # spatial
                    if dimension == 'geohash':
                        t = Type.spatial
                    condition = Condition(dimension, value, t)
                # in
                elif where.find('IN') != -1:
                    dimension = where.split('IN')[0].strip()
                    value = where.split('IN')[1].replace('\'', '').strip()[1:-1].replace(' ', '').split(',')
                    condition = Condition(dimension, value, Type.categorical)
                # between
                elif where.find('BETWEEN') != -1:
                    dimension = where.split('BETWEEN')[0].strip()
                    value = [s.strip() for s in where.split('BETWEEN')[1].replace('\'', '').split('and')]
                    condition = Condition(dimension, value, Type.temporal)
                self.wheres[self.cube.dimensions.index(condition.dimension)] = condition
                self.where_n = self.where_n + 1

        # parse group by
        if sql.find('GROUP') != -1:
            self.groupby = sql[sql.find('GROUP BY') + 9:].strip()
        self.result = ResultSet(self.groupby, self.agg + '(' + self.measure + ')')

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

    def clear_conditions(self):
        self.wheres = []
        for d in self.cube.dimensions:
            c = Condition(dimension=d)
            self.wheres.append(c)
        self.where_n = 0

    def clear(self):
        self.result = ResultSet(self.groupby, self.agg + '(' + self.measure + ')')

