import datetime

import geohash2
import pandas as pd

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

    def bin(self, dimension, ds, begin, end, pbar):
        layer = []
        if begin == 0 and end == -1:
            self.R.sort_values(by=dimension, inplace=True)
            self.R['temporalBin'] = self.R[dimension]
            if self.by == 'YEAR':
                currentYear = 0
                for index, date in self.R[dimension].items():
                    year = datetime.datetime.strptime(date, self.format).year
                    if year > currentYear + self.granularity:
                        currentYear = year
                    self.R['temporalBin'][index] = currentYear
            elif self.by == 'MONTH':
                currentMonth = 0
                for index, date in self.R[dimension].items():
                    month = datetime.datetime.strptime(date, self.format).month
                    if month > currentMonth + self.granularity:
                        currentMonth = month
                    self.R['temporalBin'][index] = datetime.date(date.year, month, 1)
            elif self.by == 'DAY':
                currentDay = 0
                for index, date in self.R[dimension].items():
                    day = datetime.datetime.strptime(date, self.format).day
                    if day > currentDay + self.granularity:
                        currentDay = day
                    self.R['temporalBin'][index] = datetime.date(date.year, date.month, day)
            self.R.reset_index(drop=True, inplace=True)
            for index, value in self.R['temporalBin'].value_counts().sort_index().items():
                firstLayerDs = DimensionSet(dimension, index, Interval(begin, begin + value - 1))
                begin = begin + value
                layer.append(firstLayerDs)
                pbar.update(firstLayerDs.interval.count)
            self.R.drop(columns=['temporalBin'], inplace=True)
        else:
            partialRelation = self.R.iloc[begin:end + 1]
            partialRelation = partialRelation.sort_values(by=dimension)
            partialRelation['temporalBin'] = partialRelation[dimension]
            if self.by == 'YEAR':
                currentYear = 0
                for index, date in partialRelation[dimension].items():
                    year = datetime.datetime.strptime(date, self.format).year
                    if year > currentYear + self.granularity:
                        currentYear = year
                    partialRelation['temporalBin'][index] = currentYear
            elif self.by == 'MONTH':
                currentMonth = 0
                for index, date in partialRelation[dimension].items():
                    month = datetime.datetime.strptime(date, self.format).month
                    if month > currentMonth + self.granularity:
                        currentMonth = month
                    partialRelation['temporalBin'][index] = datetime.date(date.year, month, 1)
            elif self.by == 'DAY':
                currentDay = 0
                for index, date in partialRelation[dimension].items():
                    day = datetime.datetime.strptime(date, self.format).day
                    if day > currentDay + self.granularity:
                        currentDay = day
                    partialRelation['temporalBin'][index] = datetime.date(date.year, date.month, day)
            partialRelation.set_index(pd.Index(range(begin, end + 1)), inplace=True)
            for index, value in partialRelation['temporalBin'].value_counts().sort_index().items():
                sub = DimensionSet(dimension, index, Interval(begin, begin + value - 1), ds)
                begin = begin + value
                ds.subSet.append(sub)
                layer.append(sub)
                pbar.update(sub.interval.count)
            partialRelation.drop(columns=['temporalBin'], inplace=True)
            self.R.iloc[ds.interval.begin:ds.interval.end + 1, :] = partialRelation[:]
        return layer


class CategoricalDimension(object):
    def __init__(self, R):
        self.R = R

    def bin(self, dimension, ds, begin, end, pbar):
        layer = []
        if begin == 0 and end == -1:
            self.R.sort_values(by=dimension, inplace=True)
            self.R.reset_index(drop=True, inplace=True)
            for index, value in self.R[dimension].value_counts().sort_index().items():
                firstLayerDs = DimensionSet(dimension, index, Interval(begin, begin + value - 1))
                begin = begin + value
                layer.append(firstLayerDs)
                pbar.update(firstLayerDs.interval.count)
        else:
            partialRelation = self.R.iloc[begin:end + 1]
            partialRelation = partialRelation.sort_values(by=dimension)
            partialRelation.set_index(pd.Index(range(begin, end + 1)), inplace=True)
            for index, value in partialRelation[dimension].value_counts().sort_index().items():
                sub = DimensionSet(dimension, index, Interval(begin, begin + value - 1), ds)
                begin = begin + value
                ds.subSet.append(sub)
                layer.append(sub)
                pbar.update(sub.interval.count)
            self.R.iloc[ds.interval.begin:ds.interval.end + 1, :] = partialRelation[:]
        return layer


class SpatialDimension(object):
    def __init__(self, R, hashLength):
        self.R = R
        self.length = hashLength

    def bin(self, lnglat, ds, begin, end, pbar):
        layer = []
        if begin == 0 and end == -1:
            self.R['geohash'] = ''
            self.R['geohash'] = self.R.apply(lambda x: geohash2.encode(x[lnglat[1]], x[lnglat[0]], self.length), axis=1)

            self.R.sort_values(by='geohash', inplace=True)
            self.R.reset_index(drop=True, inplace=True)
            for index, value in self.R['geohash'].value_counts().sort_index().items():
                firstLayerDs = DimensionSet('geohash', index, Interval(begin, begin + value - 1))
                begin = begin + value
                layer.append(firstLayerDs)
                pbar.update(firstLayerDs.interval.count)
            self.R.drop(columns=['geohash'], inplace=True)
        else:
            partialRelation = self.R.iloc[begin:end + 1]
            partialRelation['geohash'] = ''
            partialRelation['geohash'] = partialRelation['geohash'].apply(
                lambda x: geohash2.encode(x[lnglat[1]], x[lnglat[0]], self.length), axis=1)

            partialRelation = partialRelation.sort_values(by='geohash')
            partialRelation.set_index(pd.Index(range(begin, end + 1)), inplace=True)
            for index, value in partialRelation['geohash'].value_counts().sort_index().items():
                sub = DimensionSet('geohash', index, Interval(begin, begin + value - 1), ds)
                begin = begin + value
                ds.subSet.append(sub)
                layer.append(sub)
                pbar.update(sub.interval.count)
            partialRelation.drop(columns=['geohash'], inplace=True)
            self.R.iloc[ds.interval.begin:ds.interval.end + 1, :] = partialRelation[:]
        return layer
