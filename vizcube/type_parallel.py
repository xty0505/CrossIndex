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
    def __init__(self, R, ds, by, granularity, format):
        self.R = R
        self.by = by
        self.granularity = granularity
        self.format = format

        dimension_to_sort = ['temporalBin']
        p = ds
        while p is not None:
            if p.type == Type.numerical:
                dimension_to_sort.append(p.dimension)
            p = p.parent
        dimension_to_sort.reverse()
        self.dimension_to_sort = dimension_to_sort

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
        r['temporalBin'] = r[dimension].apply(self.bin_by_granularity)
        r = r.sort_values(by=self.dimension_to_sort)
        r.set_index(pd.Index(range(begin, end + 1)), inplace=True)

        last_bin = list(r['temporalBin'])[0]
        value = 0
        for bin_name in r['temporalBin']:
            if last_bin != bin_name:
                sub = DimensionSet(dimension, Type.temporal, str(last_bin), Interval(begin, begin + value - 1), ds)
                begin = begin + value
                ds.subSet.append(sub)
                layer.append(sub)
                pbar.update(value)
                # reset
                value = 0
                last_bin = bin_name
            value += 1
        sub = DimensionSet(dimension, Type.temporal, str(last_bin), Interval(begin, begin + value - 1), ds)
        ds.subSet.append(sub)
        layer.append(sub)
        pbar.update(value)

        r.drop(columns=['temporalBin'], inplace=True)
        # self.R.iloc[ds.interval.begin:ds.interval.end + 1, :] = r[:]
        return {'layer': layer, 'r': r}


class CategoricalDimension(object):
    def __init__(self, R, dimension, ds):
        self.R = R

        dimension_to_sort = [dimension]
        p = ds
        while p is not None:
            if p.type == Type.numerical:
                dimension_to_sort.append(p.dimension)
            p = p.parent
        dimension_to_sort.reverse()
        self.dimension_to_sort = dimension_to_sort

    def bin(self, dimension, ds, begin, end, pbar):
        layer = []
        r = self.R.iloc[begin:end + 1]
        r = r.sort_values(by=self.dimension_to_sort)
        r.set_index(pd.Index(range(begin, end + 1)), inplace=True)

        last_bin = list(r[dimension])[0]
        value = 0
        for bin_name in r[dimension]:
            if last_bin != bin_name:
                sub = DimensionSet(dimension, Type.categorical, str(last_bin), Interval(begin, begin + value - 1), ds)
                begin = begin + value
                ds.subSet.append(sub)
                layer.append(sub)
                pbar.update(value)
                # reset
                value = 0
                last_bin = bin_name
            value += 1
        sub = DimensionSet(dimension, Type.categorical, str(last_bin), Interval(begin, begin + value - 1), ds)
        ds.subSet.append(sub)
        layer.append(sub)
        pbar.update(value)

        # self.R.iloc[ds.interval.begin:ds.interval.end + 1, :] = r[:]
        return {'layer': layer, 'r': r}


class SpatialDimension(object):
    def __init__(self, R, ds, hashLength):
        self.R = R
        self.length = hashLength

        dimension_to_sort = ['geohash']
        p = ds
        while p is not None:
            if p.type == Type.numerical:
                dimension_to_sort.append(p.dimension)
            p = p.parent
        dimension_to_sort.reverse()
        self.dimension_to_sort = dimension_to_sort

    def bin(self, lnglat, ds, begin, end, pbar):
        layer = []
        r = self.R.iloc[begin:end + 1]
        r['geohash'] = r.apply(lambda x: geohash2.encode(x[lnglat[1]], x[lnglat[0]], self.length), axis=1)
        r = r.sort_values(by=self.dimension_to_sort)
        r.set_index(pd.Index(range(begin, end + 1)), inplace=True)

        last_bin = list(r['geohash'])[0]
        value = 0
        for bin_name in r['geohash']:
            if last_bin != bin_name:
                sub = DimensionSet('geohash', Type.spatial, str(last_bin), Interval(begin, begin + value - 1), ds)
                begin = begin + value
                ds.subSet.append(sub)
                layer.append(sub)
                pbar.update(value)
                # reset
                value = 0
                last_bin = bin_name
            value += 1
        sub = DimensionSet('geohash', Type.spatial, str(last_bin), Interval(begin, begin + value - 1), ds)
        ds.subSet.append(sub)
        layer.append(sub)
        pbar.update(value)

        r.drop(columns=['geohash'], inplace=True)
        # self.R.iloc[ds.interval.begin:ds.interval.end + 1, :] = r[:]
        return {'layer': layer, 'r': r}


class NumericalDimension(object):
    def __init__(self, R, dimension, ds, bin_width):
        self.R = R
        self.bin_width = bin_width

        dimension_to_sort = [dimension]
        p = ds
        while p is not None:
            if p.type == Type.numerical:
                dimension_to_sort.append(p.dimension)
            p = p.parent
        dimension_to_sort.reverse()
        self.dimension_to_sort = dimension_to_sort

    def bin(self, dimension, ds, begin, end, pbar):
        layer = []
        bin_label = dimension+"_bin"

        r = self.R.iloc[begin: end + 1]
        r = r.sort_values(by=self.dimension_to_sort)
        r.set_index(pd.Index(range(begin, end + 1)), inplace=True)

        last_bin = list(r[bin_label])[0]
        value = 0
        for bin_name in r[bin_label]:
            if last_bin.left != bin_name.left:
                dimension_value = '(' + str(last_bin.left) + ', ' + str(last_bin.right) + ']'
                sub = DimensionSet(dimension, Type.numerical, dimension_value, Interval(begin, begin + value - 1), ds)
                begin = begin + value
                ds.subSet.append(sub)
                layer.append(sub)
                pbar.update(value)
                # reset
                value = 0
                last_bin = bin_name
            value += 1
        sub = DimensionSet(dimension, Type.numerical, str(last_bin), Interval(begin, begin + value - 1), ds)
        ds.subSet.append(sub)
        layer.append(sub)
        pbar.update(value)

        # self.R.iloc[ds.interval.begin:ds.interval.end + 1, :] = r[:]
        return {'layer': layer, 'r': r}
