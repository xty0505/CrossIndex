from type_parallel import *


class Sort(object):
    def __init__(self, R, begin, end):
        self.R = R
        self.begin = begin
        self.end = end
        self.ds = None

    def setBeginEnd(self, begin, end):
        self.begin = begin
        self.end = end

    def setDs(self, ds):
        self.ds = ds

    def sort(self, begin, end, ds, dimension, dimensionType, pbar):
        self.setBeginEnd(begin, end)
        self.setDs(ds)
        if dimensionType == Type.categorical:
            d = CategoricalDimension(self.R)
            return d.bin(dimension, self.ds, self.begin, self.end, pbar)
        elif dimensionType == Type.temporal:
            by = 'YEAR'
            granularity = 1
            format = '%m/%d/%Y %H:%M:%S %p'
            d = TemporalDimension(self.R, by, granularity, format)
            return d.bin(dimension, self.ds, self.begin, self.end, pbar)
        elif dimensionType == Type.spatial:
            hashLength = 8
            d = SpatialDimension(self.R, hashLength)
            return d.bin(dimension, self.ds, self.begin, self.end, pbar)
