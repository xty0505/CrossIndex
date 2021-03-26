from type_parallel import *


class Sort(object):
    def __init__(self, R, begin=0, end=0):
        self.R = R
        self.begin = begin
        self.end = end
        self.ds = None

    def setBeginEnd(self, begin, end):
        self.begin = begin
        self.end = end

    def setDs(self, ds):
        self.ds = ds

    def sort(self, begin, end, ds, dimension, dimension_type, pbar=None):
        self.setBeginEnd(begin, end)
        self.setDs(ds)
        if dimension_type == Type.categorical:
            d = CategoricalDimension(self.R, dimension, ds)
            return d.bin(dimension, self.ds, self.begin, self.end, pbar)
        elif dimension_type == Type.temporal:
            by = 'DAY'
            granularity = 1
            format = '%Y/%m/%d %H:%M'
            d = TemporalDimension(self.R, ds=ds, by=by, granularity=granularity, format=format)
            return d.bin(dimension, self.ds, self.begin, self.end, pbar)
        elif dimension_type == Type.spatial:
            hashLength = 8
            d = SpatialDimension(self.R, ds=ds, hashLength=hashLength)
            return d.bin(dimension, self.ds, self.begin, self.end, pbar)
        elif dimension_type == Type.numerical:
            bin_width = 10
            d = NumericalDimension(self.R, dimension, ds, bin_width)
            return d.bin(dimension, ds, begin, end, pbar)
