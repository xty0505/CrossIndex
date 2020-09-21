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

    @staticmethod
    def getType(s):
        if s == 'spatial':
            return Type.spatial
        elif s == 'categorical':
            return Type.categorical
        elif s == 'temporal':
            return Type.temporal


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

    def bin_parallel(self, dimension, ds, begin, end, pbar):
        r = self.R.iloc[begin: end + 1]
        items = self.R[dimension].value_counts().sort_index().items()
        layer = Parallel(n_jobs=8)(delayed(build_parallel)(item, dimension, ds) for item in items)
        pbar.update(len(r))
        return layer

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


def build_parallel(item_tuple, dimension, parent):
    index = item_tuple[0]
    value = item_tuple[1]
    sub = DimensionSet(dimension, index, Interval(value, value + 1), parent)
    if parent is not None:
        parent.subSet.append(sub)
    return sub
