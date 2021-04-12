import datetime
import os

import geohash2
import pandas as pd

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

    def bin_by_granularity(self, temporal_data):
        # remove time zone (2020-02-01 00:00:00+00:00)
        if '+' in temporal_data:
            temporal_data = temporal_data[:-6]
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
        cur = begin
        for key, value in r[dimension].apply(self.bin_by_granularity).value_counts().items():
            sub = DimensionSet(dimension, Type.categorical, key, Interval(cur, cur+value-1), ds)
            cur += value
            ds.subSet.append(sub)
            layer.append(sub)
            pbar.update(value)
        
        return layer


class CategoricalDimension(object):
    def __init__(self, R, dimension, ds):
        self.R = R

    def bin(self, dimension, ds, begin, end, pbar):
        layer = []
        r = self.R.iloc[begin:end + 1]
        cur = begin
        for key, value in r[dimension].value_counts().items():
            sub = DimensionSet(dimension, Type.categorical, key, Interval(cur, cur+value-1), ds)
            cur += value
            ds.subSet.append(sub)
            layer.append(sub)
            pbar.update(value)
        
        return layer


class SpatialDimension(object):
    def __init__(self, R, ds, hashLength):
        self.R = R
        self.length = hashLength

    def bin(self, lnglat, ds, begin, end, pbar):
        layer = []
        r = self.R.iloc[begin:end + 1]
        r['geohash'] = r.apply(lambda x: geohash2.encode(x[lnglat[1]], x[lnglat[0]], self.length), axis=1)
        cur = begin
        for key,value in r['geohash'].value_counts().items():
            sub = DimensionSet('geohash', Type.spatial, key, Interval(cur, cur+value-1), ds)
            cur += value
            ds.subSet.append(sub)
            layer.append(sub)
            pbar.update(value)

        r.drop(columns=['geohash'], inplace=True)
        return layer


class NumericalDimension(object):
    def __init__(self, R, dimension, ds, bin_width):
        self.R = R
        self.bin_width = bin_width

    def bin(self, dimension, ds, begin, end, pbar):
        layer = []
        bin_label = dimension+"_bin"
        r = self.R.iloc[begin: end + 1]

        cur = begin
        for key,value in r[bin_label].value_counts().items():
            sub = DimensionSet(dimension, Type.numerical, key, Interval(cur, cur+value-1), ds)
            cur += value
            ds.subSet.append(sub)
            layer.append(sub)
            pbar.update(value)

        return layer
