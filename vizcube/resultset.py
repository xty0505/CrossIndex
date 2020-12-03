from dimension import DimensionSet, Interval


class ResultSet(object):
    def __init__(self, x_name, y_name):
        self.x_name = x_name
        self.y_name = y_name
        self.x_data = []
        self.y_intervals = []
        self.y_data = []

    def convert_to_filters(self):
        res = []
        for intervals in self.y_intervals:
            for interval in intervals:
                filt = "%s <= id <= %s" % (str(interval.begin), str(interval.end))
                res.append(filt)
        return " AND ".join(res)

    def convert_to_filters_IN(self):
        res = []
        for intervals in self.y_intervals:
            for interval in intervals:
                for i in range(interval.begin, interval.end
                                + 1):
                    res.append(i)
        return res

    def to_dict(self):
        result = {}
        for i in range(len(self.x_data)):
            result[self.x_data[i]] = [self.y_data[i]]
        return result

    def output(self):
        result = '"x_name":"' + self.x_name + '","y_name":"' + self.y_name + '","x_data":"' + str(
            self.x_data) + '","y_data":"' + str(self.y_data)
        print(result)

    def pretty_output(self):
        width = 10
        print('x_name:' + self.x_name + ',y_name:' + self.y_name)
        print(self.x_name.ljust(width, ' ') + '\t' + self.y_name)
        for i in range(len(self.x_data)):
            print(self.x_data[i].ljust(width, ' ') + '\t' + str(self.y_data[i]))

    def output_xy(self):
        return {'x_data': self.x_data,
                'y_data': self.y_data}
