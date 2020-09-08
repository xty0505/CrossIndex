class Interval(object):
    def __init__(self, begin, end):
        self.begin = begin
        self.end = end
        self.count = end - begin + 1

    def compare(self, other):
        if self.begin == other.begin and self.end == other.end:
            return 0
        # interval2 包含 interval1
        if self.begin >= other.begin and self.end <= other.end:
            return 2
        # interval1 包含 interval2
        if self.begin <= other.begin and self.end >= other.end:
            return 1
        # interval1 < interval2
        if self.end < other.begin:
            return -1
        # interval > interval2
        if self.begin > other.end:
            return -2

    def output(self):
        return 'interval: [' + str(self.begin) + ',' + str(self.end) + ']'

    def save(self):
        return '[' + str(self.begin) + ',' + str(self.end) + ']'


class DimensionSet(object):
    def __init__(self, dimension_name, dimension_value, interval, parent=None):
        self.dimension = dimension_name
        self.value = dimension_value
        self.interval = interval
        self.parent = parent
        self.subSet = []

    def output(self):
        print('dimension: ' + self.dimension + ', value: ' + str(self.value) + ', ' + self.interval.output())
        for d in self.subSet:
            print('\t=>', end='')
            d.output()

    def save(self):
        return self.dimension + ';' + str(self.value) + ';' + self.interval.save()

    def load(self, line):
        args = line.split(';')
        self.dimension = args[0]
        self.value = args[1]
        begin = int(args[2][1:-1].split(',')[0])
        end = int(args[2][1:-1].split(',')[1])
        self.interval = Interval(begin, end)

    def find_parent(self, dimension):
        p = self.parent
        while p.dimension != dimension:
            p = p.parent
        return p
