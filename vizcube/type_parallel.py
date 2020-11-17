import datetime
import os

import geohash2
import pandas as pd
from joblib import Parallel, delayed

from dimension import DimensionSet, Interval


class Type(object):
    spatial = 0
    categorical = 1
    temporal = 2
    numerical = 3

    @staticmethod
    def getType(s):
        if s == 'spatial':
            return Type.spatial
        elif s == 'categorical':
            return Type.categorical
        elif s == 'temporal':
            return Type.temporal
        elif s == 'numerical':
            return Type.numerical


class TemporalDimension(object):
    def __init__(self, R, by, granularity, format):
        self.R = R
        self.by = by
        self.granularity = granularity
        self.format = format

    def bin_by_granularity(self, temporal_data):
        date = datetime.datetime.strptime(temporal_data, self.format)
        if self.by == 'YEAR':
            return date.year
        elif self.by == 'MONTH':
            return datetime.date(date.year, date.month, 1)
        elif self.by == 'DAY':
            return datetime.date(date.year, date.month, date.day)

    def bin(self, dimension, ds, begin, end, pbar):
        layer = []
        r = self.R.iloc[begin:end + 1]
        r = r.sort_values(by=dimension)
        r['temporalBin'] = r[dimension].apply(self.bin_by_granularity)
        r.set_index(pd.Index(range(begin, end + 1)), inplace=True)
        for index, value in r['temporalBin'].value_counts().sort_index().items():
            sub = DimensionSet(dimension, index, Interval(begin, begin + value - 1), ds)
            begin = begin + value
            ds.subSet.append(sub)
            layer.append(sub)
            pbar.update(sub.interval.count)
        r.drop(columns=['temporalBin'], inplace=True)
        self.R.iloc[ds.interval.begin:ds.interval.end + 1, :] = r[:]
        return layer


class CategoricalDimension(object):
    def __init__(self, R):
        self.R = R

    def bin(self, dimension, ds, begin, end, pbar):
        layer = []
        r = self.R.iloc[begin:end + 1]
        r = r.sort_values(by=dimension)
        r.set_index(pd.Index(range(begin, end + 1)), inplace=True)
        for index, value in r[dimension].value_counts().sort_index().items():
            sub = DimensionSet(dimension, index, Interval(begin, begin + value - 1), ds)
            begin = begin + value
            ds.subSet.append(sub)
            layer.append(sub)
            pbar.update(sub.interval.count)
        self.R.iloc[ds.interval.begin:ds.interval.end + 1, :] = r[:]
        return layer


class SpatialDimension(object):
    def __init__(self, R, hashLength):
        self.R = R
        self.length = hashLength

    def bin(self, lnglat, ds, begin, end, pbar):
        layer = []
        r = self.R.iloc[begin:end + 1]
        r['geohash'] = r.apply(lambda x: geohash2.encode(x[lnglat[1]], x[lnglat[0]], self.length), axis=1)
        r = r.sort_values(by='geohash')
        r.set_index(pd.Index(range(begin, end + 1)), inplace=True)
        for index, value in r['geohash'].value_counts().sort_index().items():
            sub = DimensionSet('geohash', index, Interval(begin, begin + value - 1), ds)
            begin = begin + value
            ds.subSet.append(sub)
            layer.append(sub)
            pbar.update(sub.interval.count)
        r.drop(columns=['geohash'], inplace=True)
        self.R.iloc[ds.interval.begin:ds.interval.end + 1, :] = r[:]
        return layer

class NumericalDimension(object):
    def __init__(self, R, bin_width):
        self.R = R
        self.bin_width = bin_width

    def bin(self, dimension, ds, begin, end, pbar):
        layer = []
        bin_label = dimension + '_bin'
        # 在R上进行分箱, 保证每个bin宽度相同
        self.R[bin_label] = pd.cut(self.R[dimension], self.bin_width).tolist()
        r = self.R.iloc[begin: end + 1]
        r = r.sort_values(by=dimension)
        r.set_index(pd.Index(range(begin, end + 1)), inplace=True)
        for index, value in r[bin_label].value_counts().sort_index().items():
            sub = DimensionSet(dimension, index, Interval(begin, begin + value - 1), ds)
            begin = begin + value
            ds.subSet.append(sub)
            layer.append(sub)
            pbar.update(sub.interval.count)
        r.drop(columns=[bin_label], inplace=True)
        self.R.iloc[ds.interval.begin:ds.interval.end + 1, :] = r[:]
        return layer