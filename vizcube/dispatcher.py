from calc_type import *


class Dispatcher(object):
    def __init__(self, R, options, begin=0, end=0):
        self.R = R
        self.begin = begin
        self.end = end
        self.ds = None
        self.options = options

    def setBeginEnd(self, begin, end):
        self.begin = begin
        self.end = end

    def setDs(self, ds):
        self.ds = ds

    def dispatch(self, begin, end, ds, dimension, dimension_type, pbar=None):
        self.setBeginEnd(begin, end)
        self.setDs(ds)
        if dimension_type == Type.categorical:
            d = CategoricalDimension(self.R, dimension, ds)
            return d.bin(dimension, self.ds, self.begin, self.end, pbar)
        elif dimension_type == Type.temporal:
            by = self.options['by']
            granularity = self.options['granularity']
            format = self.options['date_format']
            d = TemporalDimension(self.R, ds=ds, by=by, granularity=granularity, format=format)
            return d.bin(dimension, self.ds, self.begin, self.end, pbar)
        elif dimension_type == Type.spatial:
            hash_length = self.options['hash_length']
            d = SpatialDimension(self.R, ds=ds, hashLength=hash_length)
            return d.bin(dimension, self.ds, self.begin, self.end, pbar)
        elif dimension_type == Type.numerical:
            bin_width = self.options['bin_width']
            d = NumericalDimension(self.R, dimension, ds, bin_width)
            return d.bin(dimension, ds, begin, end, pbar)
