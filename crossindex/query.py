import geohash2
import pandas as pd

from dimension import Interval
from dimension import DimensionSet
from resultset import ResultSet
from calc_type import Type


# where 条件
class Condition(object):
    def __init__(self, dimension, value=None, type=None):
        self.dimension = dimension
        self.value = value
        self.type = type

    def binary_match(self, subset):
        res = []
        condition = self.value if self.value.__class__ == list else [self.value]

        def binary_search1(l, r, subset, target):
            while l <= r:
                mid = int((l+r)/2)
                if subset[mid].value == target:
                    return mid
                elif subset[mid].value > target:
                    r = mid-1
                else:
                    l = mid+1
            return l
        
        def binary_search2(l, r, subset, target):
            while l <= r:
                mid = int((l+r)/2)
                if float(subset[mid].value) == target:
                    return mid
                elif float(subset[mid].value) > target:
                    r = mid-1
                else:
                    l = mid+1
            return l

        if self.type == Type.categorical:
            if type(condition[0]) is float:
                idx = binary_search2(0, len(subset)-1, subset, condition[0])
                for i in range(idx, len(subset)):
                    if condition[0] <= float(subset[i].value) < condition[1]:
                        res.append(subset[i])
                    else:
                        break # float(subset[i].value) >= condition[1]
            else:
                idx = 0
                for target in sorted(condition):
                    idx = binary_search1(idx, len(subset)-1, subset, target)
                    if idx < len(subset) and subset[idx].value == target:
                        res.append(subset[idx])
        elif self.type == Type.temporal:
            idx = binary_search1(0, len(subset)-1, subset, condition[0])
            for i in range(idx, len(subset)):
                if condition[0] <= subset[i].value <= condition[1]:
                    res.append(subset[i])
                else:
                    break
        elif self.type == Type.spatial:
            idx = binary_search1(0, len(subset)-1, subset, condition[0])
            for i in range(idx, len(subset)):
                if subset[i].value.find(condition[0])>=0:
                    res.append(subset[i])
        elif self.type == Type.numerical:
            # todo
            print('numerical')
        return res

    def match_csv(self, df, dimension, offset=0, bin_width=1):
        condition = self.value if self.value.__class__ == list else [self.value]
        if self.type == Type.categorical:
            if type(condition[0]) is float:
                return df[(df[dimension]>=condition[0])&(df[dimension]<condition[1])]
            return df[df[dimension].isin(condition)]
        elif self.type == Type.temporal:
            return df[(df[dimension]>=condition[0])&(df[dimension]<=condition[1])]
        elif self.type == Type.numerical:
            condition_bin = [int((c-offset)//bin_width) for c in condition]
            return df[(df[dimension]>=condition_bin[0])&(df[dimension]<condition_bin[1])]

    def match(self, ds, R=None):
        value = ds.value
        condition = self.value if self.value.__class__ == list else [self.value]

        if self.type == Type.categorical:
            if type(condition[0]) is float:
                return condition[0] <= float(value) < condition[1]
            return value in condition
        elif self.type == Type.temporal:
            return condition[0] <= value <= condition[1]
        elif self.type == Type.spatial:
            if len(value) > len(condition[0]):
                return value.find(condition[0]) != -1
            elif len(value) == len((condition[0])):
                return value == condition[0]
        elif self.type == Type.numerical:
            # bin_boundary: (x, y]
            # condition: [x, y)
            bin_boundary = [float(x.strip()) for x in value[1:-1].split(',')]
            if bin_boundary[0] >= condition[1] or bin_boundary[1] < condition[0]:
                return False, None
            if bin_boundary[0] >= condition[0] and bin_boundary[1] < condition[1]:
                return True, None
            if bin_boundary[0] < condition[0] <= bin_boundary[1] < condition[1]:
                r = R.iloc[ds.interval.begin:ds.interval.end + 1, :]
                r = r[r[ds.dimension] >= condition[0]]
                if len(r) == 0:
                    return False, None
                idx = r.index.tolist()
                new_ds = DimensionSet(ds.dimension, Type.numerical, '(' + str(condition[0]) + ',' + value.split(',')[1],
                                      Interval(idx[0], idx[-1]), ds.parent)
                new_ds.subSet = ds.subSet
                self.adjust_subset(new_ds)
                return True, new_ds
            if condition[0] <= bin_boundary[0] < condition[1] <= bin_boundary[1]:
                r = R.iloc[ds.interval.begin:ds.interval.end + 1, :]
                r = r[r[ds.dimension] < condition[1]]
                if len(r) == 0:
                    return False, None
                idx = r.index.tolist()
                new_ds = DimensionSet(ds.dimension, Type.numerical, value.split(',')[0] + ',' + str(condition[1]) + ')',
                                      Interval(idx[0], idx[-1]), ds.parent)
                new_ds.subSet = ds.subSet
                self.adjust_subset(new_ds)
                return True, new_ds
            if bin_boundary[0] < condition[0] <= condition[1] <= bin_boundary[1]:
                r = R.iloc[ds.interval.begin:ds.interval.end + 1, :]
                r = r[(condition[0] <= r[ds.dimension]) & (r[ds.dimension] < condition[1])]
                if len(r) == 0:
                    return False, None
                idx = r.index.tolist()
                new_ds = DimensionSet(ds.dimension, Type.numerical, '[' + str(condition[0]) + ',' + str(condition[1]) + ')',
                                      Interval(idx[0], idx[-1]), ds.parent)
                new_ds.subSet = ds.subSet
                self.adjust_subset(new_ds)
                return True, new_ds

    def adjust_subset(self, ds):
        new_subSet = []
        if ds.subSet[0].interval.begin >= ds.interval.begin and ds.subSet[-1].interval.end <= ds.interval.end:
            return
        for sub in ds.subSet:
            # 原ds下的sub不再属于new_ds
            if sub.interval.end < ds.interval.begin:
                continue
            if sub.interval.begin > ds.interval.end:
                break

            # 原ds下的部分sub属于new_ds, 调整相应sub的interval, 并递归调整sub的subSet
            if sub.interval.begin < ds.interval.begin <= sub.interval.end <= ds.interval.end:
                new_sub = DimensionSet(sub.dimension, sub.type, sub.value, Interval(ds.interval.begin, sub.interval.end), ds)
                new_sub.subSet = sub.subSet
                self.adjust_subset(new_sub)
            elif ds.interval.begin <= sub.interval.begin <= ds.interval.end < sub.interval.end:
                new_sub = DimensionSet(sub.dimension, sub.type, sub.value, Interval(sub.interval.begin, ds.interval.end), ds)
                new_sub.subSet = sub.subSet
                self.adjust_subset(new_sub)

            # sub 正好属于new_ds, 无需调整
            else:
                new_sub = sub

            new_subSet.append(new_sub)
        ds.subSet = new_subSet


aggregation = {'CNT': 'COUNT',
               'AVG': 'AVG',
               'SUM': 'SUM'}


class Query(object):
    def __init__(self, measure='', agg='', groupby='', cube=None):
        self.measure = measure
        self.agg = agg
        self.groupby = groupby
        self.result = ResultSet(groupby, agg + '(' + measure + ')')
        self.cube = cube
        self.wheres = []
        self.where_n = 0
        for d in self.cube.dimensions:
            c = Condition(dimension=d)
            self.wheres.append(c)
        self.validDSs = []
        self.cache = {}

    def set_cube(self, cube):
        self.cube = cube
        self.wheres = []
        self.where_n = 0
        for d in self.cube.dimensions:
            c = Condition(dimension=d)
            self.wheres.append(c)

    def parse(self, sql):
        # projection
        projection = sql[sql.find("SELECT") + 6: sql.rfind("FROM")].strip()
        for column in projection.split(','):
            column = column.strip()
            if column.startswith("FLOOR"):
                continue
            # parse measure and agg
            b = column.find('(') + 1
            e = column.find(')')
            if sql.find(aggregation.get('CNT')) != -1:
                self.agg = aggregation.get('CNT')
            elif sql.find(aggregation.get('AVG')) != -1:
                self.agg = aggregation.get('AVG')
            elif sql.find(aggregation.get('SUM')) != -1:
                self.agg = aggregation.get('SUM')
            self.measure = column[b:e]

        # parse where conditions
        conditions = self.parse_conditions(sql)
        for c in conditions:
            self.wheres[self.cube.dimensions.index(c.dimension)] = c
        self.where_n = len(conditions)

        # parse group by
        if sql.find('GROUP') != -1:
            groupby = sql[sql.find('GROUP BY') + 9:].strip()
            if groupby.startswith('bin'):
                groupby = groupby[4:]
            self.groupby = groupby
        self.measure = self.groupby if self.measure=='*' else self.measure
        self.result = ResultSet(self.groupby, self.agg + '(' + self.measure + ')')

    def parse_conditions(self, sql):
        # parse where conditions
        conditions = []
        if sql.find('WHERE') != -1:
            wheres = sql[sql.find('WHERE') + 6:sql.find('GROUP')].split(' AND ')
            for where in wheres:
                # >= and <
                if where.find('>=') != -1:
                    dimension = where.split('>=')[0].strip().strip('(')
                    value = where.split('>=')[1].strip().strip(')')
                    d_type = self.cube.types[self.cube.dimensions.index(dimension)]
                    condition = Condition(dimension, [float(value)], d_type)
                elif where.find('<') != -1:
                    dimension = where.split('<')[0].strip().strip('(')
                    value = where.split('<')[1].strip().strip(')')
                    for w in conditions:
                        if w.dimension == dimension:
                            w.value.append(float(value))
                            break
                # =
                elif where.find('=') != -1:
                    dimension = where.split('=')[0].strip()
                    value = where.split('=')[1].replace('\'', '').strip()
                    t = Type.categorical
                    # spatial
                    if dimension == 'geohash':
                        t = Type.spatial
                    condition = Condition(dimension, value, t)
                # in
                elif where.find('IN') != -1:
                    dimension = where.split('IN')[0].strip().strip('(')
                    value = where.split('IN')[1].replace('\'', '').strip().strip(')')[1:].replace(' ', '').split(',')
                    condition = Condition(dimension, value, Type.categorical)
                # between
                elif where.find('BETWEEN') != -1:
                    dimension = where.split('BETWEEN')[0].strip().strip('(')
                    value = [s.strip() for s in where.split('BETWEEN')[1].replace('\'', '').split('and')]
                    condition = Condition(dimension, value, Type.temporal)

                conditions.append(condition)
        return list(set(conditions))

    def compute(self, valid_id):
        res = self.cube.R.loc[valid_id]
        idx = self.cube.dimensions.index(self.groupby)
        res[self.groupby] = (res[self.groupby]-self.cube.offset[idx])//self.cube.bin_width[idx]
        if self.agg == aggregation.get('CNT'):
            xy_series = res[self.groupby].value_counts().sort_index()
            self.result.x_data = list(xy_series.index.astype(str))
            self.result.y_data = list(xy_series.values)
        elif self.agg == aggregation.get('AVG'):
            xy_series = res.groupby(self.groupby)[self.measure].sum()
            self.result.x_data = list(xy_series.index.astype(str))
            self.result.y_data = list(xy_series.values)
        elif self.agg == aggregation.get('SUM'):
            xy_series = res.groupby(self.groupby)[self.measure].mean()
            self.result.x_data = list(xy_series.index.astype(str))
            self.result.y_data = list(xy_series.values)

    def get_query_index_sql(self, idx, tbname):
        where_clause = ""
        if self.wheres[idx].value is None:
            return "", False
        for i in range(idx+1):
            where = self.wheres[i]
            if where.value is not None:
                value = where.value if where.value.__class__ == list else [where.value]
                if where.type == Type.categorical:
                    if type(value[0]) is float:
                        where_clause += "{0} >= {1} and {0} < {2} AND ".format(where.dimension, str(value[0]), str(value[1]))
                    else:
                        where_clause += "{0} IN ({1}) AND ".format(where.dimension, ','.join(["'%s'" % x for x in value]))
                elif where.type == Type.numerical:
                    where_clause += "{0} >= {1} and {0} < {2} AND ".format(where.dimension, str(value[0]), str(value[1]))
                elif where.type == Type.temporal:
                    where_clause += "{0} >= {1} and {0} < {2} AND ".format(where.dimension, str(value[0]), str(value[1]))
        where_clause = where_clause[:where_clause.rfind(" AND ")]
        sql = "SELECT {0} FROM {1} WHERE {2}".format(self.groupby, tbname, where_clause)
        return sql, True


    def get_geo_result(self, limit):
        max = 0
        data = []
        if limit is None:
            for i in range(len(self.result.x_data)):
                intervals = self.result.y_intervals[i]
                point = {'lat': self.cube.R.iloc[intervals[0].begin]['lat'],
                         'lng': self.cube.R.iloc[intervals[0].begin]['lng'],
                         'count': int(self.result.y_data[i])}
                if self.result.y_data[i] > max:
                    max = int(self.result.y_data[i])
                data.append(point)
        else:
            # tmp_x = self.result.x_data
            # tmp_y = self.result.y_intervals
            for i in range(len(self.result.x_data)):
                intervals = self.result.y_intervals[i]
                valid_count = 0
                valid_row = pd.DataFrame()
                # tmp_intervals = []
                for interval in intervals:
                    tmp = self.cube.R.iloc[interval.begin:interval.end + 1]
                    valid_row = tmp[((limit['bottom'] <= tmp['lat']) & (tmp['lat'] <= limit['top'])) &
                                    ((limit['left'] <= tmp['lng']) & (tmp['lng'] <= limit['right']))]
                    valid_count = valid_count + valid_row[self.measure].count()
                    # tmp_intervals.append(valid_row)
                if valid_count == 0:
                    # tmp_x.remove(self.result.x_data[i])
                    # tmp_y.remove(intervals)
                    continue
                # tmp_y[i] = tmp_intervals
                lat, lng = valid_row.iloc[0]['lat'], valid_row.iloc[0]['lng']
                point = {'lat': lat,
                         'lng': lng,
                         'count': int(valid_count)}
                if valid_count > max:
                    max = int(valid_count)
                data.append(point)
            # self.result.x_data = tmp_x
            # self.result.y_intervals = tmp_y
            # self.compute()
        return data, max

    def add_condition(self, condition):
        index = self.cube.dimensions.index(condition.dimension)
        if self.wheres[index].value is None:
            self.where_n = self.where_n + 1
        self.wheres[index] = condition

    def get_start_condition(self):
        if self.where_n == 0:
            return -1
        for i,w in enumerate(self.wheres):
            if w.value is not None:
                return i

    def clear_conditions(self):
        self.wheres = []
        for d in self.cube.dimensions:
            c = Condition(dimension=d)
            self.wheres.append(c)
        self.where_n = 0

    # 返回最后一个重叠 dimension 的下标
    # 第二个返回参数代表最后一个 dimension 的新谓词是否在原谓词上进行缩小
    def get_deepest_overlapped_idx(self, new_conditions):
        idx = -1
        flag = False
        for i in range(len(new_conditions)):
            old_condition = self.wheres[i]
            new_condition = new_conditions[i]
            if old_condition.value is None and new_condition.value is None:
                continue
            elif old_condition.value is None or new_condition.value is None:
                break
            elif new_condition.value != old_condition.value:
                if old_condition.type == Type.categorical:
                    if type(old_condition.value[0]) is float and \
                            old_condition.value[0] <= new_condition.value[0] \
                            and new_condition.value[1] <= old_condition.value[1]:
                        flag = True
                    elif set(new_condition.value) <= set(old_condition.value):
                        flag = True
                elif old_condition.type == Type.temporal:
                    if old_condition.value[0] <= new_condition.value[0] and new_condition.value[1] <= old_condition.value[1]:
                        flag = True
                elif old_condition.type == Type.spatial:
                    if old_condition.value.find(new_condition.value) > 0:
                        flag = True
                elif old_condition.type == Type.numerical:
                    # print(old_condition.value, new_condition.value)
                    if old_condition.value[0] <= new_condition.value[0] and new_condition.value[1] <= old_condition.value[1]:
                        flag = True

                if flag:
                    idx = i
                break
            else:
                idx = i
        return idx, flag

    def is_backward_query(self, sql):
        if self.where_n == 0:
            return False
        conditions = self.parse_conditions(sql)
        for c in conditions:
            old_condition = self.wheres[self.cube.dimensions.index(c.dimension)]
            if old_condition.value is None:
                continue

            # compare whether can backward query
            if old_condition.type == Type.categorical:
                if type(old_condition.value[0]) is float:
                    if c.value[0] < old_condition.value[0] or c.value[1] > old_condition.value[1]:
                        return False
                else:
                    for v in c.value:
                        if not v in old_condition.value:
                            return False
            elif old_condition.type == Type.temporal:
                if c.value[0] < old_condition.value[0] or c.value[1] > old_condition.value[1]:
                    return False
            elif old_condition.type == Type.spatial:
                if len(c.value) < len(old_condition.value):
                    return False
                elif len(c.value) > len(old_condition.value):
                    if old_condition.value.find(c.value) == -1:
                        return False
            elif old_condition.type == Type.numerical:
                if c.value[0] < old_condition.value[0] or c.value[1] > old_condition.value[1]:
                    return False
        return True

    def clear(self):
        self.result = ResultSet(self.groupby, self.agg + '(' + self.measure + ')')

if __name__ == '__main__':
    print('a')
    # crossindex = crossindex.crossindex_main.CrossIndex('flights_covid_10M', ["day","typecode","origin","destination","registration","icao24","callsign"],
    #                         ["temporal", "categrocial", "categorical", "categorical", "categorical", "categorical", "categorical"])
    # q = Query(cube=crossindex)
    # q.parse("SELECT icao24 AS bin_icao24,  COUNT(*) as count FROM flights_covid WHERE (typecode IN ('CNJ','CRJ-1000','CRJ1','CRJ2','CRJ7','CRJ9','CRJX','CRUZ','CT4','CVLT','D11','D228','D328','DA40','DA42','DC10','DC3','DH2T','DH3T','DH82','DH8A','DH8C','DH8D','DHC2','DHC6','DIMO','DR40','DV20','E120','E135','E140','E145','E170','E190','E195','E35L','E45X','E50P','E545','E550','E55P','E737','E75L','E75S','EA50','EA55','EC20','EC25','EC30','EC35','EC45','ECHO','EPIC','EUPA','EV97','F100','F2EX','F2LX','F2TH','F2TS','F5EX','F70','F900','F9DX','F9EX','F9LX','FA10','FA20','FA50','FA7X','FDCT','FK14','FK9','FURY','G150','G200','G280','G2CA','G300','G350','G450','G550','G650','GA8','GALX','GL5T','GL6T','GLEX','GLF3','GLF4','GLF5','GX','H25B','H25C','H269','H500','H60','H900','HA4T','HDJT','J328','K35R','KFIR','KODI','L39','L8','LJ24','LJ31','LJ35','LJ40','LJ45') AND registration IN ('N353AK','N353FS','N353JB','N353NB','N353TS','N354JB','N354K','N354NW','N354P','N354PT','N355CA','N355NB','N355PU','N355VJ','N356NW','N356P','N357FC','N357PV','N357UP','N358HA','N358JB','N358LL','N358NB','N358QS','N358TV','N359FB','N359NB','N359ST','N3604T','N360NB','N360QS','N360UP','N3616','N361FX','N361UP','N361VA','N36247','N36272','N362NB','N362QS','N363NB','N363VA','N36444','N36469','N364CM','N364NB','N364VA','N365NB','N365NW','N365QS','N365VA','N366NB','N366NW','N367FX','N368CA','N368CS','N368FX','N368JB','N368NW','N369FX','N36BL','N36VU','N37018','N370HA','N37172','N371DA','N371FL','N371QS','N37252','N37273','N37277','N37293','N37298','N372CM','N372DA','N372NW','N372QS','N3731T','N3733Z','N3737C','N3738B','N3739P','N373DA','N373FX','N373JB','N373KM','N373NW','N373QS','N37408','N37413','N37419','N37422','N3743H','N37456','N3745B','N37468','N3746H','N37470','N37471','N37474','N3747D','N3748Y','N3749D','N374AC','N374CA','N374JB','N374NW','N3751B','N3756','N3757D','N375ET','N375JB','N3760C','N3761R','N3765','N3766','N3767','N3769L','N376DA','N3772H','N3773D','N377DA','N377NW','N378AX','N378BD','N378NW','N3796B','N379CA','N379DA','N37FD','N37XX','N381DN','N38257','N38268','N382DA','N38443','N38454','N38458','N38459','N384CF','N385PB','N386AZ','N386DA','N386ME','N387CP','N387LS','N388DA','N388MD','N388MM','N389DA','N389HC','N389P','N38BZ','N38RX','N390CA','N390QS','N391PH','N39297','N392DA','N393DA','N39415','N39450','N39461','N39463','N39475','N394DA','N395DC','N395DN','N395MB','N396AF','N396AN','N396DA','N396DN','N396LG','N39728','N397NG','N398DA','N3990A','N399DA','N399DH','N399KK','N399LF','N399LG','N3AS','N3CP','N3NJ','N3PC','N3VJ','N4005X','N400AY','N400FX','N400RV','N400WN','N401FW','N401KZ','N401LV','N401PG','N401PH','N401QX','N401UA','N401VA','N401VR','N401WN','N402CT','N402UA','N402UP','N402WN','N402YX','N4032T','N403AS','N403SH','N403TD','N403WN','N404BC','N404DP','N404UA','N404WN','N405AX','N405QS','N405UP','N405WN','N406FX','N406UA','N406UP','N406WN','N406YX','N407BB','N407CC','N407CF','N407PG','N407QS','N407SS','N407TK','N407WA','N407WN','N407YX','N408AM','N408AS','N408DF','N408GG','N408LM','N408PC','N408RK','N408UA','N408UP','N408WN','N408YX','N409MC','N409QS','N409WN','N409YX','N40PD','N40VC','N410AW','N410UA','N410UP','N410WN','N41140','N4118K','N411QS','N411TV','N411UA','N411UP','N411WN','N411ZW','N412DJ','N412MC','N412MW','N412QX','N412TL','N412UP','N412WN','N412YX','N413AW','N413UA','N413UH','N413WN','N413YX','N414UA','N414UP','N414WN','N415RB','N415WN','N415YX','N416AW','N416CF','N416WN','N417UA','N417UP','N417WN','N418GJ','N418SW','N418UA','N418WN','N418YX','N419AS','N419CF','N419MC','N419P','N419UP','N41PM','N4204','N420DT','N420DZ','N420EM','N420PC','N420SK','N420WN','N420YX','N421LV','N421MA','N421QX','N421YX','N422MA','N422QX','N422UA','N422UP','N422WN','N422YX','N423AS','N423EM','N423JG','N423SW','N423YX','N424DF','N424YX','N425UA','N426CB','N426DF','N426QS','N426QX','N426WN','N427SW','N427UA','N427UP','N427V','N427WN','N428UP','N428WN','N429AW','N429CF','N429SW','N429UA','N429WM','N429WN','N42J','N42PC','N430AW','N430WN','N430YX','N431P','N431SW','N432AW','N432CV','N432UP','N432YX','N433AK','N433AM','N433HC','N433LV','N433P','N433QX','N433SW','N433YX','N434MK','N434WN','N434YX','N435AS','N435QX','N435SW','N435WN','N436AW','N436QX','N436WN','N436YX','N43722','N437AW','N437JD','N437QX','N437UA','N437WN','N437YX','N438AW','N438SW','N438WN','N438WR','N438YX','N439QX','N439UA','N439WN','N439YX','N43GJ','N43PC','N440HA','N440LN','N440LV','N440QX','N440SF','N440SW','N440UP','N441DT','N441MD','N441QX','N441WF','N442AS','N442JP','N442QX','N442SW','N442UA','N442WN','N443AW','N443EA','N443QX','N443UA','N443UP','N443WN','N444DF','N444NW','N444QX','N444UA','N444ZW','N445QX','N445SW','N445UP','N445WN','N445YX','N446QX','N446SW','N446WN','N446YX','N447AW','N447QX','N447RC','N447SF','N447YX','N448QX','N448WN','N449SW','N449WN','N44SW','N450AW','N450RG','N450UP','N451QX','N452WN','N453AW','N453FX','N453MH','N453PA','N453QX','N453UA','N453UP','N453WN','N454AW','N454SW','N454UA','N454WN','N455DK','N455SW','N455UP','N455WN','N456SB','N456TK','N456UA','N456UP','N456ZW','N457AS','N457UA','N458AW','N458SF','N458UA','N458WN','N45956','N459SB','N459UP','N459WN','N45RF','N45YF','N460AW','N460TM','N460UP','N460WN','N461AW','N461SW','N461UA','N461WN','N462UP','N463UP','N463WN','N464UA','N464WN','N465PC','N465UA','N465WN','N4662B','N466WN','N467AW','N467CS','N467P','N467UA','N467WN','N468GH','N469AF','N469UA','N46FE','N46HL','N46VE','N470FX','N470WN','N470ZW','N471AS','N471FX','N471UP','N471ZW','N472AS','N472CA','N472UA','N473UA','N474AS','N474UA','N475UA','N475UP','N476BJ','N476UA','N476WN','N477JB','N477MC','N477UA','N477WN','N477XP','N478AS','N478DR','N478WN','N479AS','N479CA','N479PR','N479TA','N479UA','N479WN','N47CD','N47EG','N47NS','N480M','N480WN','N481AS','N481WN','N483UA','N483WN','N484BW','N484UA','N485AF','N485WN','N486AS','N486LF','N486MT','N486WN','N487CA','N487WN','N488DF','N488WN','N48901','N490WN','N491AM','N491AS','N491N','N491WN','N492AS','N492BB','N492SC','N492SW','N492UA','N493WN','N4940E','N4945F','N495AS','N495PL','N495RS','N495WN','N496CA','N496TA','N496WN','N497CA','N497P','N497SP','N497WN','N498UA','N498WN','N499CZ','N499WN','N49AD','N49VA','N5000B','N500AD','N500AE','N500CE','N500DR','N500EH','N500N','N500PC','N500RR','N500UP','N500WR','N500ZE','N501KQ','N501LS','N502AE','N502MJ','N503JB','N503MJ','N503NK','N503VJ','N504KQ','N504MJ','N504QS','N505AE','N505SP','N505U','N506AE','N506DN','N506MJ','N507AM','N507GF','N507JT','N507SF','N507UP','N508AY','N508MJ','N509AE','N509AY','N509MJ','N509UP','N50AE','N510MJ','N510NK','N510TL','N511AE','N511D','N511SC','N511SL','N511TP','N511UW','N511VL','N512AR','N512AS','N512NK','N513QS','N515AE','N515AN','N515JT','N516AE','N516AS','N516JB','N516NK','N517AE','N517FX','N517MT','N517PD','N5185G','N518AE','N518MT','N518WC','N519AS','N519JB','N519LR','N519TV','N520AP','N520DC','N521LR','N521NK','N521UW','N521VA','N522AE','N522FE','N522KL','N522ME','N522NK','N522QS','N522VA','N523AE','N524AE','N524AS','N524FX','N524JB','N524LR','N524NK','N524UW','N524VA','N524XA','N525CH','N525CR','N525J','N525LD','N525NE','N525NK','N525SD','N525TK','N526DV','N526EA','N526FG','N526LF','N526TW','N526VA','N527AS','N527EA','N527JL','N527VK','N528FE','N528FX','N528NK','N529AS','N529EA','N529FE','N529NK','N529VA','N52FW','N530JL','N530SC','N531AS','N532AS','N532NK','N533QS','N534AE','N534AS','N534CC','N534NK','N534UW','N5355S','N535AS','N535EA','N535FX','N535JB','N535LN','N536UW','N537US','N537UW','N538CA','N538EG','N538KL','N538QS','N538UW','N539EA','N539UW','N539XJ','N540EG','N540MA','N540UW','N54178','N541FX','N54241','N542KD','N543AE','N543CM','N543PB','N543QS','N543TX','N543US','N544EA','N544QS','N544UW','N545K','N545PB','N545QS','N545SA','N545US','N545UW','N546FF','N546MA','N546UA','N546US','N546UW','N547JB','N547NN','N547QS','N549NN','N549US','N549WB','N54FL','N54HG','N54TJ','N550DR','N550DX','N550NN','N550NT','N550NW','N550TH','N550WN','N551CP','N551DN','N551GJ','N551NN','N551UW','N551WN','N5520N','N552JB','N552NN','N552NW','N552UW','N552XJ','N553FX','N553NN','N553UW','N553WN','N554CA','N554NW','N554WN','N555LG','N555LV','N555NN','N556AL','N556CP','N556NN','N556NW','N556UW','N557AS','N557QS','N557WN','N558JB','N558QS','N558UW','N558WN','N559JB','N559NN','N559WN','N55FP','N560HN','N560JW','N560LS','N560PA','N560RK','N560RP','N560SJ','N560UW','N561JB','N561SR','N562FE','N562JB','N562NA','N562P','N562UW','N563AS','N564AS','N564WN','N565JB','N565NN','N565SK','N565WN','N566JB','N566KB','N566LF','N566TX','N566WN','N567AV','N567CA','N567FH','N567NN','N567UW','N567WN','N56859','N568JB','N568UW','N56CS','N56LW','N570FX','N570JB','N570UP','N570WN','N571FX','N571NN','N571RM','N572NN','N572UW','N573NN','N573UP','N573UW','N57439','N574FE','N574NN','N575FX','N575MW','N575RE','N576FE','N576SA','N577AS','N577NN','N577QS','N57855','N57857','N57863','N579AS','N579BJ','N579FX','N579NC','N579UW','N57PA','N580JB','N580NN','N580UW','N581AS','N581HC','N581JN','N582NN','N582UW','N583JB','N583JS','N583NN','N583NW','N583UP','N584A','N584AS','N584UW','N585UW','N586AS','N586JB','N586UW','N587JB','N587NN','N587UW','N588AS','N588JB','N588NG','N588NN','N588NW','N588QS','N588UA','N589AS','N589NW','N58AJ','N590AE','N590JB','N590NN','N590UA','N591NN','N592AS','N592FE','N5938Y','N593AS','N594JB','N594NN','N595JB','N595NN','N595NW','N595UA','N596NW','N597JB','N598JB','N598NN','N599JB','N599SD','N5PX','N5QV','N600BP','N600CK','N600LR','N600QS','N600VM','N601AW','N601DW','N601LR','N601NK','N602AE','N602CZ','N603AT','N603BA','N603KC','N603L','N603NK','N603SK','N604AW','N604CF','N604CR','N604CZ','N604DH','N604GA','N604MD','N604SD','N604WH','N605BS','N605CZ','N605L','N605LR','N605NK','N606AE','N606CZ','N606NK','N606UP','N607AT','N607CZ','N607NK','N607UP','N6081E','N608AT','N608CZ','N608LM','N608NK','N608SM','N608UP','N609NK','N609QS','N60SB','N60TX','N610CZ','N610UP','N611BF','N611MR','N611QS','N612CZ','N612JB','N612NK','N612SH','N613AE','N613CZ','N613JB','N613QS','N614AE','N614CZ','N614QS','N614SK','N615FE','N615JB','N615NK','N615PG','N616AE','N616DC','N616NK','N616PR','N616UP','N617AC','N617AE','N617SP','N61882','N61886','N61887','N61898','N619CZ','N619NK','N61RK','N61UP','N620NK','N620QX','N620WA','N621AE','N621JB','N621NK','N621VA','N621VS','N622CZ','N622QX','N623AE','N623CZ','N623NK','N623QX','N624AL','N624FE','N624MY','N624NK','N624QX','N625CA','N625NK','N625VA','N626AE','N626NK','N627AE','N627CZ','N627NK','N627QX','N627VA','N62883','N62884','N62889','N62892','N62894','N62895','N62896','N628AE','N628NK','N628QX','N629AE','N629VA','N630NK','N630QX','N630TK','N630VA','N631AE','N631CZ','N631QX','N631SK','N631VA','N632NK','N632SK','N632VA','N632XL','N633AE','N633HC','N633SK','N633VA','N634CZ','N634JB','N635AE','N635CZ','N635NK','N635RW','N636AE','N636CZ','N636JB','N636LF','N636RW','N637CZ','N637NG','N637NK','N637RW','N63820','N63890','N638AE','N638JB','N638NK','N639CS','N639CZ','N639NK','N639RW','N63L','N63TK','N640AE','N640NK','N640RW','N640VA','N641CA','N641NK','N641VA','N642AE','N642AN','N642FE','N642GT','N642QS','N642UA','N642VA','N6437D','N643NK','N643RW','N644AE','N644AS','N644GT','N644MW','N644NK','N644RW','N644SD','N645AE','N645GT','N645NK','N646AE','N646RW','N646UA','N647AE','N647GT','N647QS','N647RW','N64809','N64844','N648NK','N648RW','N649AW','N649DL','N649DX','N649GT','N649NK','N649PP','N649RW','N649UA','N64GT','N64PJ','N650AE','N650AW','N650FE','N650HP','N650NK','N650RL','N650RW','N651AE','N651PB','N652BR','N652FE','N652RW','N652UA','N653AE','N653AW','N653CA','N653JB','N653UA','N654AW','N654FE','N654RW','N655AE','N655AW','N655UA','N656AE','N656AW','N656CA','N657AE','N657AW','N657KT','N657QS','N657T','N65832','N658AE','N658DL','N658GT','N658JB','N658NK','N658QS','N659AE','N659BX','N659CA','N659DL','N65PJ','N66','N660NK','N660UA','N661AW','N661FE','N661JA','N661NK','N661UA','N662AW','N662EH','N662NK','N662UA','N663DN','N663NK','N663UA','N664AW','N664DN','N664MS','N664NK','N664QS','N665AW','N665BC','N665NK','N665PD','N667AW','N667GB','N66803','N66814','N66831','N66893','N668AF','N668AW','N668CA','N668FE','N668HH','N669CA','N669MB','N669NK','N66W','N66ZC','N67058','N6706Q','N6709','N670AT','N670NK','N6710E','N6713Y','N67171','N671AE','N671NK','N672CG','N672NK','N673AW','N673UA','N67413','N674NK','N674UA','N675FE','N676MA','N676NK','N676UA','N677NK','N67815','N67845','N678AE','N679DA','N67CF','N680AW','N680CA','N680DA','N681AE','N681DA','N681GH','N682AC','N682AE','N682DA','N683AE','N683DA','N683UF','N68452','N68453','N684DA','N684JW','N684PS','N684TA','N685DA','N685MF','N685NK','N686AE','N686BR','N686RC','N686T','N687DL','N68805','N68811','N68821','N68822','N68834','N68842','N68880','N688CB','N688DL','N688TA','N689DL','N689EC','N69063','N690AE','N690ES','N690XL','N691AA','N691AE','N691CA','N692AE','N692CA','N692DL','N693AE','N693BR','N693DL','N695AE','N696DL','N696QS','N697DL','N69804','N69810','N69816','N69819','N69824','N69826','N69830','N69839','N69888','N698CB','N698DL','N699XP','N6RA','N6TY','N700BW','N700CK','N700FX','N700GS','N700JE','N700NA','N700UW','N701CK','N701FR','N701FX','N701GS','N701MA','N701UW','N7027U','N702DN','N702DR','N702FR','N702GT','N702PC','N702PS','N702SK','N702SS','N702TW','N702UW','N7039N','N703DN','N703PS','N703VZ','N70464','N70465','N704FR','N704SW','N704US','N705FX','N705JB','N705KC','N705MP','N705PS','N705SW','N705TW','N706CK','N706FR','N706FX','N706PS','N706SW','N707FX','N707SJ','N708FR','N708JB','N708SH','N709JB','N709MT','N709SK','N709SW','N709UW','N70AE','N70VM','N70X','N710BG','N710SW','N710UW','N711AW','N711FX','N711HK','N712PS','N712SK','N712SW','N713FR','N713SW','N713TW','N714AX','N714TS','N714US','N715AF','N715FR','N716AV','N716FR','N716PS','N716UW','N717AN','N717JL','N717SA','N718HT','N718PS','N718SK','N718SW','N718WA','N7195P','N719PC','N719SW','N71EX','N71SC','N71SY','N71WP','N72','N7200Z','N720CH','N720FR','N720HW','N721AN','N721BB','N721FD','N721TW','N721YX','N722YX','N723GH','N723UW','N723YX','N724HF','N724SK','N724SW','N724UW','N725SW','N725UW','N726AN','N726QS','N726SK','N726YX','N727AN','N727DD','N727KB','N727SK','N727SW','N727YX','N728LM','N728SK','N728SW','N729AN','N729FX','N729QS','N729SW','N729YX','N72EH','N72KA','N730AN','N730EV','N730QS','N730SK','N730US','N731AN','N731FX','N731SA','N731YX','N73256','N73259','N73270','N73275','N732AN','N732JR','N732MD','N732SK','N732US','N732YX','N733FD','N733SA','N733UW','N73445','N734CB','N734SA','N734TE','N736AT','N736SA','N736YX','N737AT','N737JW','N738MA','N738SK','N739GB','N739YX','N73KT','N740SW','N740UW','N741AV','N742E','N742FD','N742QS','N742SK','N742SW','N742YX','N743AE','N743CK','N743DB','N743FD','N743SW','N743YX','N744SK','N744SW','N745CK','N745SK','N745SW','N745YX','N746FX','N746SK','N746SW','N746UW','N746YX','N747FE','N747SA','N747YX','N748EV','N748SW','N748YX','N749AX','N749SW','N74FS','N750AX','N750EV','N750GJ','N750GX','N750SD','N750SK','N750WR','N750Z','N7510E','N7511W','N751AN','N751UW','N752DS','N752SW','N7534U','N753AN','N753JL','N75425','N75426','N75429','N75432','N75435','N75436','N755SA','N755SK','N756SK','N756US','N757UW','N757XJ','N75858','N75861','N758A','N758EV','N758SK','N758US','N75925','N759AN','N759EV','N759GB','N759P','N75G','N75LY','N75UM','N75VC','N76062','N76065','N760SW','N760XJ','N761CK','N761CX','N761FE','N761LE','N76288','N762CX','N762JP','N762MT','N762QS','N762SK','N762SW','N762T','N763SK','N76503','N76504','N76508','N76515','N76532','N76533','N765AN','N765CK','N765KA','N765SK','N765US','N766AN','N766SK','N767AX','N767SK','N767SW','N768FE','N768KD','N768SK','N768US','N769SW','N769US','N769XJ','N76PM','N7702A','N7704B','N77066','N7706A','N7707C','N7709A','N770AN','N770FE','N770QS','N770SK','N7710A','N7711N','N7712G','N7713A','N7715E','N7716A','N7717D','N7718B','N771FE','N771MG','N771SA','N7721E','N7722B','N7723E','N7725A','N7726A','N7727A','N7728D','N7729A','N7730A','N7732A','N7733B','N7734H','N7736A','N7737E','N7738A','N773AN','N7740A','N7741C','N7742B','N7743B','N7745A','N7746C','N7747C','N7748A','N7749B','N774DE','N774SK','N774SW','N774UA','N774WF','N7750A','N77510','N77518','N77530','N77535','N77536','N775AN','N775DE','N775FE','N775SW','N776DE','N776QS','N776WN','N777AN','N777KK','N777QC','N777QS','N777SA','N77865','N778FE','N778SW','N779PA','N779SK','N779SW','N77NG','N77SF','N78001','N78003','N78009','N78017','N780AN','N780DW','N780P','N780QS','N780SK','N780SW','N7814B','N7815L','N7816B','N7818L','N7819A','N781H','N781UA','N781WN','N7821L','N7822A','N7823A','N7825A','N7826B','N7828A','N7829B','N782UA','N7830A','N7832A','N7833A','N7834A','N7835A','N7836A','N7838A','N7839A','N783AV','N783AX','N783SW','N783XJ','N7841A','N7842A','N7843A','N78448','N7844A','N7845A','N7846A','N7847A','N7848A','N784SW','N78501','N78509','N7850B','N7851A','N7852A','N7853B','N78540','N7854B','N7855A','N7856A','N7857B','N7858A','N785AM','N785AV','N785DW','N785MM','N785SW','N785WW','N786AN','N786SK','N786SW','N786WM','N7873A','N7874B','N7875A','N7877H','N7879A','N787SA','N7880D','N7884G','N78866','N788AN','N788UA','N789FE','N789SW','N790SW','N791SK','N791SW','N792AV','N792CP','N792SK','N792SW','N793SA','N793SK','N793WF','N794SW','N794XJ','N79541','N795SK','N795SW','N796JB','N796JS','N796SW','N797MX','N797QS','N798P','N798QS','N798SW','N799CZ','N799TS','N7NJ','N7QY','N7UF','N8009T','N800AN','N800CR','N800GF','N800GN','N800KV','N800NN','N800SK','N801AW','N801AY','N801PB','N801PN','N801SB','N801SY','N802AA','N802RR','N802SK','N802UA','N8030F','N8031M','N803AE','N803CP','N803DN','N803F','N803HC','N803NN','N803UA','N804AN','N804CE','N804DN','N804JB','N8052A','N805AE','N805AN','N805JB','N805MP','N805SK','N805TH','N806AA','N806AW','N806ER','N806JB','N806SK','N806UA','N8073W','N807D','N807DD','N807DN','N807JB','N807UA','N808AE','N808AN','N808MT','N808NN','N809AA','N809JA','N809JB','N809NN','N809NW','N80WW','N8100E','N810DN','N810TD','N810UA','N8116N','N811AB','N811DZ','N811FD','N812AW','N812DN','N812NN','N812UA','N8132R','N813AW','N813NN','N813TA','N814BB','N814LV','N814SS','N814SY','N814UA','N815AE','N815AW','N815UA','N816AW','N816CE','N816LF','N816NN','N816NW','N816SY','N817AE','N817AN','N817DN','N817NN','N817NW','N817SY','N817UA','N818HD','N818MD','N818NN','N819AY','N819DN','N819DX','N819KR','N819UA','N81CR','N81FJ','N81NM','N81P','N81SF','N8200E','N820AB','N820AT','N820AW','N820DN','N820SK','N820SY','N821AV','N821AW','N821DN','N821DS','N821DX','N821MH','N821NN','N821PP','N821SK','N821SY','N821UP','N822AW','N822DN','N822DX','N822NW','N82338','N823AW','N823DF','N823DN','N823DX','N823SK','N824AW','N824DN','N824NN','N824SK','N824UA','N825NN','N825UA','N826AN','N826TG','N826UA','N827AN','N827AW','N827JB','N827JS','N827MH','N827NN','N828AW','N828JB','N828VV','N829NN','N829NW','N829UA','N82WP','N8301J','N8305E','N8306H','N830AN','N830AW','N830DN','N830GC','N830MH','N830NW','N830UA','N8310C','N8312C','N8313F','N8314L','N8315C','N8319F','N831AA','N831AW','N831FL','N831ME','N8322X','N8323C','N8326F','N8327A','N8329B','N832AW','N832HK','N832JS','N832UP','N833AY','N833DN','N833NN','N833PA','N833UA','N834AY','N834DN','N834UA','N835HK','N835UA','N835VA','N836AY','N836GC','N836UA','N836VA','N837AW','N837VA','N838AE','N838AF','N838BP','N838CC','N838CS','N838NN','N838SB','N838UA','N839AW','N839CS','N839DN','N839HK','N839NN','N83ML','N83MP','N83TF','N8407','N840AW','N840AY','N840DN','N840VA','N841DN','N841FE','N842NN','N842UP','N842WF','N84307','N843DN','N843UA','N843VA','N844FE','N844VA','N845DN','N845UA','N845VA','N846DN','N847UA','N848DN','N848MH','N848NN','N849DN','N849MH','N849NN','N849UA','N84CM','N84JL','N84UP','N8502Z','N8503A','N8504G','N8507C','N8508W','N850FE','N850GS','N850KP','N850MA','N850MB','N850MH','N850NN','N850UA','N8510E','N8511K','N8512U','N8513F','N8514F','N8515X','N8517F','N851AC','N851FD','N851NW','N851VA','N8520Q','N8522P','N8524Z','N8526W','N8527Q','N8528Q','N8529Z','N852NW','N852UA','N852VA','N85323','N8532S','N8533S','N85340','N8534Z','N85351','N85352','N85354','N8535S','N8537Z','N8539V','N853CC','N853GT','N853NN','N853VA','N8540V','N8542Z','N8543Z','N8544Z','N8545V','N8546V','N8547V','N8549Z','N854AS','N854UA','N854VA','N8552Z','N8556Z','N8557Q','N855NN','N855RW','N855UA','N8565Z','N856GT','N856NN','N856RW','N8572X','N8576Z','N8579Z','N857DZ','N857FD','N857GT','N857NN','N857NW','N857RW','N858FE','N858GT','N858MY','N858NN','N858NW','N858Q','N859NN','N85VM','N8600F','N8602F','N8605E','N8606C','N8607M','N8608N','N8609A','N860DN','N860NN','N8610A','N8611F','N8613K','N8614M','N8615E','N8616C','N8618N','N8619F','N861DA','N861NN','N8622A','N8623F','N8624J','N8625A','N8627B','N8628A','N8629A','N862AS','N862DN','N862FD','N862GS','N862LG','N862MH','N862RW','N86309','N8630B','N86316','N8631A','N8632A','N8633A','N86344','N8634A','N86350','N8635F','N8637A','N8638A','N8639B','N863AS','N863DN','N863FD','N863FE','N863NC','N863NN','N863RW','N863UP','N8640D','N8641B','N8642E','N8643A','N8644C','N8646B','N8648A','N8649A','N864AS','N864MB','N864MH','N864NN','N8651A','N8655D','N8657B','N8659D','N865DN','N865NN','N865RW','N8660A','N8661A','N8664J','N8665D','N8667D','N8669B','N866DN','N8671D','N8672F','N8675A','N8676A','N8677A','N8679A','N867DN','N867NN','N8680C','N8683D','N8685B','N8686A','N8687A','N8688J','N8689C','N868DN','N868MH','N868NN','N8693A','N8694E','N8695D','N8696E','N8697C','N8698B','N8699A','N869NN','N86CW','N870DN','N870NN','N871DN','N871RW','N871ST','N872DN','N872MH','N87303','N87319','N87337','N87345','N874NN','N874RW','N874WD','N87527','N875DM','N875DN','N875NN','N876DN','N876LF','N876MH','N876UC','N87745','N877DN','N877H','N877NN','N8783E','N878DN','N878MC','N878NN','N878RW','N879AS','N879DN','N879NN','N879RW','N87BC','N87CE','N87ME','N880DN','N880MH','N880NN','N881DN','N881NN','N881UP','N8821C','N882FE','N882LT','N882RW','N882UP','N88327','N88330','N88331','N88335','N88341','N8836A','N8839E','N883UP','N884EA','N884NN','N885A','N885EA','N885JF','N885MH','N885NN','N88692','N8869B','N886AW','N886DC','N886DS','N886NN','N886TX','N886UP','N8877A','N887DN','N887NN','N8884E','N8886A','N888AA','N888DU','N888FD','N888HA','N888TV','N888WG','N8896A','N889DN','N889EA','N88LV','N88VN','N890UA','N8918B','N891AT','N891CS','N891DN','N891FE','N891PA','N891UA','N892UA','N89313','N89315','N89317','N89321','N8936A','N893NN','N893P','N893PA','N893UA','N894AT','N894PA','N895NN','N895UA','N8968E','N8969A','N896JH','N896NN','N896SK','N8976E','N897NN','N897SK','N8986B','N898CD','N898NN','N898PA','N899AT','N899FE','N89WC','N8KD','N8YJ','N900AE','N900DE','N900EB','N900EV','N900JS','N900KA','N900MT','N900PB','N900PC','N900TV','N900WN','N900ZP','N9012','N9013A','N901AN','N901DE','N901FD','N9022G','N9025B','N902BC','N902DE','N902EV','N902FE','N902FJ','N902MZ','N902NK','N902ST','N902XJ','N903AA','N903AN','N903DE','N903DP','N903FJ','N903NK','N903SW','N904AA','N904AN','N904DE','N904XJ','N905DE','N905DL','N905J','N905JB','N905JH','N905NN','N905SW','N906AE','N906AN','N906AS','N906WN','N907AE','N907AN','N907DE','N907LW','N907NN','N907SW','N907WN','N908AN','N908DA','N908EV','N908FJ','N908NN','N908SW','N908XJ','N909DE','N909EV','N909FJ','N909SW','N909XJ','N90EW','N90TH','N910AC','N910AN','N910AT','N910AU','N910FJ','N910FR','N910NN','N910TB','N910TR','N910WN','N911DC','N911DE','N911ET','N911FJ','N911LS','N911MU','N911RP','N911VU','N911WY','N912DE','N912DL','N912NN','N912SW','N912WN','N913AN','N913DL','N913DN','N913FD','N913FX','N913JB','N913RX','N913SW','N913US','N913WN','N914AN','N914TH','N914WN','N915AT','N915DN','N915FX','N915HG','N915NA','N915NN','N915SW','N915WN','N915XJ','N916DN','N916EV','N916FJ','N916SW','N916US','N916WN','N917AN','N917DN','N917FD','N917FJ','N917UY','N917WW','N917XJ','N9187','N918DE','N918DH','N918FE','N919AN','N919AT','N919CM','N919DN','N919PE','N919PK','N919XJ','N91AC','N920AN','N920DE','N920DN','N920FJ','N920NL','N920US','N920WN','N920XJ','N921AT','N921DN','N921FJ','N921US','N921WN','N922AE','N922AN','N922AT','N922DX','N922FJ','N922NN','N922QS','N922US','N922WN','N923AE','N923AT','N923FD','N923FJ','N923JB','N923NA','N923NN','N923SW','N923US','N923XJ','N924AT','N924DN','N924MA','N924SW','N924US','N924XJ','N925DN','N925FE','N925NN','N925SW','N925TV','N925WN','N925XJ','N926AT','N926HL','N926LR','N926NN','N926WN','N927AT','N927FD','N927NN','N927UW','N928AE','N928AN','N928DN','N928EV','N928JK','N928NN','N928QS','N929AN','N929AT','N929DN','N929EV','N929GW','N929JB','N929LR','N929MM','N929NN','N929TG','N929WN','N92FX','N92KF','N92MK','N93003','N930AT','N930AU','N930SW','N930TA','N930VM','N930WN','N930YY','N931DN','N931EV','N931LR','N931WN','N932AE','N932AT','N932EV','N932NN','N932SP','N932SW','N932WN','N9331B','N933AM','N933AT','N933EV','N933ML','N933WN','N933XJ','N934AN','N934AT','N934FJ','N934NN','N934XJ','N935AN','N935AT','N935DN','N935JB','N935WN','N936AN','N936AT','N936NN','N936SW','N936WN','N937AN','N937AT','N937JB','N937SW','N937XJ','N938AN','N938FR','N939AE','N939AT','N939TT','N939TW','N93MP','N940AN','N940AT','N941FD','N941FR','N941SW','N942AE','N942AT','N942LL','N942NN','N942WN','N943FE','N943JT','N943SW','N944AN','N944JT','N944LR','N944NN','N944WN','N9454B','N945AN','N945AT','N945LR','N945WN','N946AN','N946AT','N9479B','N947AN','N947SW','N947UW','N947WN','N9481F','N9481T','N948AT','N948LR','N948NN','N948UW','N949AN','N949AT','N949JT','N949WN','N950AN','N950AT','N950JB','N950JT','N950LA','N950LR','N950NN','N950RL','N950WN','N951LR','N951WN','N952AL','N952AT','N952CA','N952JB','N952NN','N952SW','N953AN','N953AT','N953LA','N953WN','N954AL','N954AT','N954LR','N954NN','N954SW','N954WN','N955AF','N955AN','N955H','N955NN','N955SW','N955WN','N956NN','N956SW','N956WN','N957AM','N957AT','N957JB','N957LR','N957NN','N957UW','N958AC','N958AN','N958DL','N958FD','N958SP','N958SW','N958UW','N958WN','N959AN','N959AT','N959CR','N959NN','N959UW','N95GJ','N95TV','N95VM','N960DL','N960DN','N960DT','N960FE','N960NN','N960SF','N960SW','N960WN','N961AN','N961AT','N961DL','N961FD','N961LA','N961WN','N962AN','N962JT','N962NN','N962WN','N963AN','N963AT','N963SW','N963UW','N963WN','N9642F','N9655B','N965AT','N965DL','N965JT','N965UW','N965WN','N966AT','N966JT','N967AT','N967TG','N968AN','N968AT','N968AV','N968DL','N969DL','N969JT','N969NN','N969RE','N969SW','N969TC','N969WN','N96AG','N96FT','N970SW','N971AT','N971BW','N971DL','N971MT','N971SW','N971TB','N972AE','N972AN','N972AT','N972DL','N972NN','N972PC','N973AA','N973AN','N973AV','N973JT','N973UY','N974AT','N974FD','N974TA','N975UY','N9761B','N9762B','N9766B','N976JT','N976NN','N977HG','N977UY','N978AT','N978DL','N978JB','N978NN','N978UY','N979AT','N979DL','N979HP','N979JT','N979NN','N979SW','N979TB','N979UY','N97DZ','N980NN','N981AT','N981HP','N9820F','N982AN','N982AT','N982JB','N982NN','N982QS','N983AN','N983FE','N983NN','N983SW','N984DV','N984ME','N984NN','N985CE','N985FE','N985JT','N985NN','N986JB','N986NN','N9877R','N987AM','N987FX','N987JT','N987NN','N988AL','N988AT','N988DN','N989AT','N989CJ','N989DL','N989HK','N989NN','N989PS','N98LA','N98ZA','N990AT','N990DL','N990MM','N990NN','N991AT','N991AU','N991MK','N991NN','N992AN','N992AU','N992DC','N992FD','N993AM','N993DL','N993SA','N9945Q','N994AN','N994DL','N994FE','N995AT','N995DL','N995JG','N995X','N9963H','N996AT','N996DL','N996GA','N996NN','N99700','N997AT','N997NN','N997SD','N998AN','N998FD','N998G','N999LR','N99AT','N99EF','OD-MEB','OD-MRT','OE-AAE','OE-CLA','OE-FDI','OE-FDN','OE-FHA','OE-FHK','OE-FNP','OE-FTP','OE-FUX','OE-FZB','OE-GDP','OE-GVX','OE-IAC','OE-IAG','OE-IAP','OE-IAR','OE-IAT','OE-IBI','OE-IBO','OE-ICU','OE-IHH','OE-IJD','OE-IJI','OE-IJW','OE-IJZ','OE-INH','OE-INP','OE-IQC','OE-IQD','OE-IVA','OE-IVX','OE-IZE','OE-IZJ','OE-IZL','OE-IZS','OE-IZW','OE-LAW','OE-LBD','OE-LBF','OE-LBI','OE-LBJ','OE-LBL','OE-LBN','OE-LBO','OE-LBQ','OE-LBS','OE-LBU','OE-LBY','OE-LDC','OE-LDD','OE-LDE','OE-LDF','OE-LDG','OE-LFB','OE-LKB','OE-LKD','OE-LKJ','OE-LKL','OE-LKO','OE-LMK','OE-LOA','OE-LOJ','OE-LOR','OE-LOY','OE-LPD','OE-LQG','OE-LQM','OE-LQX','OE-LWA','OE-LWB','OE-LWD','OE-LWI','OE-LWJ','OE-LWK','OE-LWL','OE-LWN','OE-LWO','OE-LWP','OE-LWQ','OE-LXC','OE-LYZ','OH-ATI','OH-ATK','OH-ATM','OH-ATN','OH-ATP','OH-DAM','OH-IOD','OH-LKE','OH-LKF','OH-LKG','OH-LKL','OH-LKM','OH-LKO','OH-LKP','OH-LKR','OH-LTO','OH-LTU','OH-LVB','OH-LVC','OH-LVD','OH-LVI','OH-LWC','OH-LXA','OH-LXC','OH-LXI','OH-LXL','OH-LZD','OH-LZE','OH-LZO','OH-SWJ','OK-BRO','OK-CTP','OK-FTR','OK-NEM','OK-NEN','OK-NEO','OK-OKP','OK-SWW','OK-TVJ','OK-TVS','OK-TVT','OM-ATU','OM-HEX','OM-OIG','OM-S039','OO-CEJ','OO-HEY','OO-JAL','OO-JAR','OO-JDL','OO-KOR','OO-PAR','OO-PCI','OO-PCJ','OO-PCK','OO-SFG','OO-SNB','OO-SNH','OO-SNI','OO-SSB','OO-SSD','OO-SSE','OO-SSH','OO-SSJ','OO-SSL','OO-SSM','OO-SSN','OO-SSO','OO-SSQ','OO-SSS','OO-SSU','OO-SSW','OO-SSX','OO-TCH','OO-TCQ','OY-EVO','OY-JPT','OY-JTR','OY-KAL','OY-KAN','OY-KAO','OY-KAP','OY-KAY','OY-KBB','OY-KBC','OY-KBF','OY-KBK','OY-KBO','OY-PDO','OY-PTL','OY-SRF','OY-SRG','OY-SRH','OY-SRJ','OY-SRK','OY-SRL','OY-SRP','OY-SRV','P4-AND','P4-GMS','P4-KBF','P4-KEA','P4-KEB','P4-LIG','P4-NAS','PH-1133','PH-ALB','PH-BCA','PH-BCB','PH-BCD','PH-BCE','PH-BCG','PH-BCL','PH-BFW','PH-BGA','PH-BGF','PH-BGG','PH-BGK','PH-BGL','PH-BGN','PH-BGP','PH-BGQ','PH-BGR','PH-BGT','PH-BHD','PH-BHL','PH-BKF','PH-BQB','PH-BQC','PH-BQN','PH-BVK','PH-BVS','PH-BVU','PH-BXA','PH-BXB','PH-BXE','PH-BXF','PH-BXG','PH-BXH','PH-BXI','PH-BXM','PH-BXN','PH-BXO','PH-BXR','PH-BXS','PH-BXT','PH-BXU','PH-CDF','PH-CGC','PH-CKB','PH-DIX','PH-DKF','PH-DOC','PH-DWA','PH-EXE','PH-EXG','PH-EXH','PH-EXI','PH-EXK','PH-EXL','PH-EXM','PH-EXR','PH-EXS','PH-EXV','PH-EXX','PH-EXY','PH-EXZ','PH-FJK','PH-HSC','PH-HSD','PH-HSE','PH-HSK','PH-HSM','PH-HXA','PH-HXB','PH-HXC','PH-HXF','PH-HXI','PH-HXJ','PH-HZE','PH-HZG','PH-HZW','PH-KFB','PH-LAB','PH-MAA','PH-SRP','PH-TFA','PH-XRB','PH-ZTI','PI-06','PK-AXV','PK-AZE','PK-AZG','PK-GMN','PP-ADZ','PP-BBI','PP-JBM','PP-KSL','PP-LID','PP-VDR','PR-ACO','PR-ALU','PR-AUH','PR-AUK','PR-AUO','PR-AXO','PR-AXR','PR-AXT','PR-BCC','PR-CAN','PR-CBA','PR-GEC','PR-GED','PR-GGD','PR-GGG','PR-GGL','PR-GGM','PR-GGP','PR-GGU','PR-GGV','PR-GGW','PR-GTA','PR-GTF','PR-GTM','PR-GTN','PR-GTP','PR-GTV','PR-GUB','PR-GUE','PR-GUF','PR-GUG','PR-GUM','PR-GUN','PR-GUY','PR-GXA','PR-GXE','PR-GXF','PR-GXH','PR-GXI','PR-GXJ','PR-GXR','PR-GXT','PR-GXU','PR-GXV','PR-LDG','PR-MBA','PR-MBF','PR-MBV','PR-MHG','PR-MHM','PR-MHR','PR-MHW','PR-MYH','PR-MYI','PR-MYK','PR-MYL','PR-MYM','PR-MYQ','PR-MYX','PR-OBK','PR-OCO','PR-TKM','PR-VBL','PR-WUC','PR-XBD','PR-XTA','PR-YRH','PT-MSV','PT-MXE','PT-MXN','PT-TMA','PT-TMB','PT-TMO','PT-XPE','RA-09010','RA-09602','RA-61723','RA-61726','RA-82046','RA-82081','RA-85686','RA-89042','RA-89051','RA-89056','RA-89067','RA-89116','RP-C3262','RP-C3264','RP-C3271','RP-C3275','RP-C3342','RP-C3343','RP-C3345','RP-C3347','RP-C3348','RP-C4104','RP-C4106','RP-C7774','RP-C7776','RP-C8619','RP-C8620','RP-C8762','RP-C8764','RP-C8765','RP-C8783','RP-C8784','RP-C8789','RP-C8971','RP-C8972','RP-C8974','RP-C9905','RP-C9916','RP-C9917','RP-C9925','RP-C9937','S5-BDM','S5-DSM','S5-JVA','SE-DOX','SE-DOY','SE-DSV','SE-KOL','SE-LJS','SE-LVU','SE-MEP','SE-MKA','SE-MKB','SE-MKC','SE-MKD','SE-MKE','SE-MKF','SE-MKG','SE-MKH','SE-RER','SE-RET','SE-RJR','SE-RJX','SE-RLT','SE-RMC','SE-RMR','SE-ROB','SE-ROC','SE-ROD','SE-ROE','SE-ROF','SE-ROJ','SE-ROS','SE-RPD','SE-RRC','SE-RRG','SE-RRH','SE-RRT','SE-RRY','SP-ENN','SP-ENR','SP-ENW','SP-KPH','SP-LDF','SP-LIB','SP-LIC','SP-LII','SP-LIN','SP-LNA','SP-LNC','SP-LNE','SP-LNL','SP-LWB','SP-LWD','SP-RKF','SP-RSQ','SP-RSS','SP-RSV','SP-RSZ','SP-TVZ','SP-ZSZ','SU-GEA','SU-GEK','SX-DGC','SX-DGJ','SX-DGL','SX-DGP','SX-DGQ','SX-DGT','SX-DGZ','SX-DNA','SX-DNB','SX-DND','SX-DNF','SX-DVG','SX-DVM','SX-DVO','SX-DVP','SX-DVQ','SX-DVT','SX-DVU','SX-DVV','SX-DVX','SX-DVZ','SX-MAI','SX-OAX','SX-OBD','SX-OBE','SX-SOF','TC-AAU','TC-ACG','TC-ADP','TC-ANP','TC-AZP','TC-CPA','TC-CPC','TC-CPD','TC-CPI','TC-CPP','TC-CPS','TC-DCA','TC-DCF','TC-DCH','TC-DCI','TC-FBV','TC-IST','TC-IZI','TC-JCI','TC-JDR','TC-JDS','TC-JFM','TC-JFU','TC-JGD','TC-JGV','TC-JHA','TC-JHB','TC-JHC','TC-JHE','TC-JHK','TC-JHP','TC-JIS','TC-JJH','TC-JJT','TC-JLZ','TC-JMH','TC-JNM','TC-JOA','TC-JOK','TC-JOZ','TC-JPK','TC-JPO','TC-JPP','TC-JPR','TC-JPT','TC-JRH','TC-JRR','TC-JSA','TC-JSN','TC-JSR','TC-JSZ','TC-JTK','TC-JTR','TC-JVB','TC-JVE','TC-JVL','TC-JVT','TC-JZE','TC-LJK','TC-LKC','TC-LNB','TC-LNC','TC-LOC','TC-LOG','TC-MCC','TC-MCG','TC-MNV','TC-NBD','TC-NBF','TC-NBG','TC-ONS','TC-SBS','TC-SCG','TC-SNZ','TC-VEL','TF-AMN','TF-BBJ','TF-FIC','TF-FIG','TF-FIH','TF-FIO','TF-FIR','TF-FIU','TF-FMS','TF-ISF','TF-ISJ','TF-ISK','TF-ISO','TF-ISS','TF-ISV','TF-LLX','TI-BJC','TS-IMU','TS-INA','UK67005','UP-C8505','UR-PSI','UR-PSO','UR-PSS','UR-PSU','UR-PSV','UR-PSW','UR-PSX','UR-PSY','V2-LIB','V2-LID','V8-DLC','V8-RBE','VH-AMQ','VH-AWE','VH-BDG','VH-BTR','VH-BZG','VH-CXJ','VH-CZI','VH-DVS','VH-DWH','VH-DYD','VH-DZQ','VH-EBA','VH-EBB','VH-EBF','VH-EBJ','VH-EBK','VH-EBL','VH-EBS','VH-EFR','VH-EGK','VH-EMS','VH-EQH','VH-EQJ','VH-EQS','VH-ESZ','VH-FDC','VH-FOX','VH-FVQ','VH-FVR','VH-FVZ','VH-HAM','VH-IHQ','VH-ING','VH-IOV','VH-JQX','VH-KIY','VH-KRX','VH-LCF','VH-LGO','VH-LJQ','VH-LQB','VH-LQD','VH-LQK','VH-LQQ','VH-LVN','VH-LWX','VH-MVW','VH-MWH','VH-MWV','VH-NEQ','VH-NJF','VH-NJI','VH-NJZ','VH-NKQ','VH-NPC','VH-NSI','VH-NXE','VH-NXI','VH-NXJ','VH-NXQ','VH-NXV','VH-OQB','VH-OQC','VH-OQH','VH-OQK','VH-OWA','VH-OWV','VH-OXK','VH-PQS','VH-PSK','VH-PVM','VH-QOB','VH-QOD','VH-QON','VH-QOV','VH-QPA','VH-QPD','VH-RTS','VH-RXE','VH-RXQ','VH-SBT','VH-SFW','VH-SGE','VH-SIF','VH-SKR','VH-TJF','VH-TJG','VH-TJI','VH-TOQ','VH-TQE','VH-TQH','VH-TQZ','VH-TSV','VH-UAH','VH-UJP','VH-UMV','VH-UUN','VH-UZI','VH-VAE','VH-VAH','VH-VAI','VH-VBY','VH-VBZ','VH-VFF','VH-VFH','VH-VFJ','VH-VFK','VH-VFL','VH-VFN','VH-VFP','VH-VFU','VH-VFV','VH-VFY','VH-VGF','VH-VGH','VH-VGI','VH-VGN','VH-VGO','VH-VGP','VH-VGQ','VH-VGY','VH-VLX','VH-VNB','VH-VNC','VH-VNK','VH-VNO','VH-VNR','VH-VOK','VH-VOL','VH-VOM','VH-VON','VH-VOR','VH-VOS','VH-VOT','VH-VOY','VH-VPE','VH-VPJ','VH-VQA','VH-VQC','VH-VQE','VH-VQF','VH-VQK','VH-VQL','VH-VSE','VH-VUA','VH-VUC','VH-VUD','VH-VUE','VH-VUF','VH-VUH','VH-VUI','VH-VUJ','VH-VUK','VH-VUO','VH-VUQ','VH-VUS','VH-VUW','VH-VUY','VH-VUZ','VH-VWN','VH-VWQ','VH-VWT','VH-VWU','VH-VWX','VH-VWY','VH-VWZ','VH-VXB','VH-VXC','VH-VXJ','VH-VXK','VH-VXN','VH-VXO','VH-VXP','VH-VXT','VH-VXU','VH-VYA','VH-VYB','VH-VYC','VH-VYD','VH-VYF','VH-VYG','VH-VYH','VH-VYJ','VH-VYK','VH-VZA','VH-VZB','VH-VZC','VH-VZD','VH-VZE','VH-VZG','VH-VZL','VH-VZM','VH-VZO','VH-VZP','VH-VZT','VH-VZU','VH-VZX','VH-WMW','VH-XFC','VH-XFJ','VH-XIH','VH-XJK','VH-XLF','VH-XLV','VH-XMO','VH-XMR','VH-XSJ','VH-XUG','VH-XUH','VH-XUS','VH-XXR','VH-XZB','VH-XZC','VH-XZD','VH-XZE','VH-XZF','VH-XZG','VH-XZI','VH-XZJ','VH-XZK','VH-XZL','VH-XZM','VH-XZN','VH-XZO','VH-XZP','VH-YCM','VH-YCN','VH-YFC','VH-YFE','VH-YFF','VH-YFH','VH-YFI','VH-YFJ','VH-YFK','VH-YFN','VH-YFP','VH-YFQ','VH-YFT','VH-YFU','VH-YFV','VH-YFX','VH-YFY','VH-YFZ','VH-YGQ','VH-YHJ','VH-YIA','VH-YIB','VH-YIF','VH-YIH','VH-YIM','VH-YIO','VH-YIQ','VH-YIT','VH-YMV','VH-YNP','VH-YQS','VH-YQT','VH-YQU','VH-YQW','VH-YQX','VH-YQY','VH-YTB','VH-YTK','VH-YUD','VH-YVA','VH-YVC','VH-YWA','VH-YXG','VH-YXH','VH-YXJ','VH-ZCX','VH-ZFD','VH-ZFE','VH-ZFO','VH-ZJP','VH-ZLS','VH-ZLV','VH-ZOU','VH-ZPA','VH-ZSM','VH-ZXA','VN-A608','VP-BAE','VP-BAV','VP-BBH','VP-BBT','VP-BBU','VP-BBY') AND day BETWEEN '2019-09-12 00:00:00' and '2020-06-22 00:00:00' AND destination IN ('EDDP','EDDR','EDDS','EDDT','EDDV','EDDW','EDEL','EDEW','EDFC','EDFE','EDFH','EDFM','EDFP','EDFV','EDFX','EDFZ','EDGE','EDGJ','EDGS','EDHS','EDJA','EDKB','EDKF','EDKM','EDLA','EDLM','EDLN','EDLR','EDLS','EDLT','EDMA','EDMK','EDMO','EDMW','EDNY','EDNZ','EDPR','EDQL','EDRA','EDRN','EDRZ','EDSB','EDST','EDTB','EDTD','EDTG','EDTM','EDTQ','EDTS','EDTZ','EDVE','EDVK','EDWC','EDWN','EDWQ','EDXH','EDXJ','EDXR','EDXW','EETN','EFHK','EFOU','EFPI','EFRO','EFTP','EGAA','EGAC','EGBB','EGBE','EGBF','EGBG','EGBJ','EGBK','EGBM','EGBP','EGCB','EGCC','EGCN','EGCV','EGDD','EGFF','EGGD','EGGP','EGGW','EGHF','EGHG','EGHH','EGHI','EGHJ','EGHQ','EGHS','EGJA','EGJB','EGJJ','EGKB','EGKK','EGKR','EGLC','EGLD','EGLF','EGLL','EGLM','EGLS','EGMA','EGMC','EGMD','EGNH','EGNJ','EGNM','EGNP','EGNR','EGNS','EGNT','EGNV','EGNX','EGOS','EGPD','EGPF','EGPG','EGPH','EGPJ','EGSC','EGSG','EGSO','EGSS','EGSX','EGTB','EGTF','EGTK','EGTR','EGUO','EGUY','EGVN','EGWC','EGXC','EHAM','EHBD','EHBK','EHEH','EHGG','EHHV','EHRD','EHSE','EHVK','EICK','EIDW','EIKN','EINN','EKAH','EKBI','EKCH','EKGR','EKOD','EKRK','EKYT','ELLX','ENBO','ENBR','ENCN','ENGM','ENHD','ENKJ','ENUL','ENVA','ENZV','EPBY','EPGD','EPKK','EPKT','EPLL','EPMO','EPPO','EPPT','EPRJ','EPSC','EPWA','EPWR','ESGG','ESKN','ESMS','ESMT','ESNZ','ESOE','ESOW','ESSA','ESSB','ESTA','ETAR','ETNT','ETSI','ETSL','EVRA','EYRD','EYRU','EYVI','EYVP','FA09','FA37','FA40','FA54','FA80','FABA','FACT','FALA','FAOR','FARA','FASK','FATA','FAYP','FD02','FD09','FD20','FD26','FD35','FD81','FD82','FD83','FD90','FD92','FL35','FL57','FL59','FL62','FL88','FL97','GA35','GA53','GA92','GA95','GABS','GBYD','GCRR','GCXO','GMME','GMMN','GMTT','GOBD','GOGG','IA30','IA56','ID00','ID19','ID26','IG07','II03','II13','II45','II71','II85','II95','IL39','IL63','IN13','IN60','IS40','K06C','K07R','K0L7','K12J','K12N','K1A9','K1C2','K1G3','K1G4','K1G5','K1H0','K1H2','K1H3','K1O3','K20A','K29M','K2G9','K2H2','K2O6','K2R2','K2W6','K36U','K39N','K3C8','K3J7','K3L2','K47N','K4A0','K4B8','K5W4','K61B','K6I6','K70J','K74S','K7L8','K7S9','K8A6','K9F9','K9X1','KABE','KABQ','KACK','KACY','KADS','KAFF','KAFN','KAFW','KAGC','KAJO','KAKH','KALB','KALN','KANE','KAPA','KAPC','KARB','KASH','KATL','KATW','KAUN','KAUS','KAVQ','KAVX','KAWO','KAXH','KBAK','KBDN','KBDR','KBED','KBFI','KBJC','KBKL','KBLI','KBLM','KBLV','KBMG','KBMI','KBNA','KBNG','KBNW','KBOI','KBOS','KBTF','KBTP','KBUF','KBUR','KBVS','KBVY','KBWI','KC15','KC20','KC29','KC47','KCAE','KCBF','KCCB','KCCR','KCDA','KCDW','KCGF','KCHD','KCHN','KCHO','KCLE','KCLL','KCLT','KCMA','KCMH','KCMI','KCNO','KCOE','KCOS','KCPK','KCPM','KCPS','KCRQ','KCTK','KCVG','KCXY','KD74','KDAB','KDAL','KDAY','KDCA','KDCM','KDEC','KDED','KDEN','KDFW','KDMA','KDMO','KDPA','KDPL','KDTO','KDTW','KDUJ','KDVT','KDWA','KDWH','KE16','KE25','KE60','KE95','KEBS','KECP','KEFD','KEMT','KERI','KESN','KETC','KEUG','KEVY','KEWN','KEWR','KEYE','KEZI','KF69','KF70','KFAT','KFCM','KFDK','KFFO','KFFZ','KFKA','KFLG','KFLL','KFME','KFMY','KFNL','KFRG','KFRH','KFTG','KFTW','KFTY','KFUL','KFWQ','KFXE','KGEG','KGLH','KGNT','KGNV','KGOO','KGOP','KGPM','KGRB','KGRK','KGRR','KGYH','KGYR','KGZH','KH68','KH71','KHEE','KHEF','KHFD','KHFJ','KHFY','KHHR','KHIO','KHLG','KHLM','KHMT','KHND','KHOU','KHPN','KHQU','KHRJ','KHRL','KHRO','KHRT','KHSD','KHSV','KHVN','KHWD','KHWO','KHXD','KHYA','KI68','KI86','KIAD','KIAG','KIAH','KIKG','KIKK','KILG','KIND','KIOW','KIPJ','KISM','KISP','KITH','KIWA','KIZA','KJFK','KJGG','KJOT','KJVL','KJVY','KK02','KL19','KL26','KL65','KL73','KLAF','KLAS','KLAX','KLCK','KLDJ','KLEB','KLGA','KLGB','KLNK','KLOM','KLOU','KLPR','KLSE','KLSV','KLUD','KLUK','KLXV','KLZU','KM33','KMAN','KMCC','KMCF','KMCI','KMCO','KMCW','KMDT','KMDW','KMFR','KMGE','KMGW','KMHR','KMHT','KMHV','KMIA','KMKC','KMKE','KMLE','KMLI','KMMK','KMMU','KMNM','KMOD','KMPO','KMQS','KMQY','KMSN','KMSP','KMSY','KMTN','KMVC','KMVY','KMWC','KMYF','KMYJ','KMYV','KMZJ','KN53','KNBJ','KNDZ','KNEL','KNEN','KNFD','KNFW','KNGS','KNGU','KNHK','KNIP','KNPA','KNRQ','KNSE','KNUC','KNUN','KNUQ','KNY2','KNYL','KNZY','KO22','KO24','KO27','KO37','KO41','KO42','KOAK','KOAR','KOCF','KOJC','KOKB','KOKC','KOLU','KOMA','KONP','KONT','KONZ','KOPF','KOQN','KOQU','KORD','KORF','KORL','KORS','KOSH','KOSU','KOWD','KOXI','KOXR','KP13','KP19','KPAE','KPAO','KPDK','KPDX','KPEA','KPGD','KPHF','KPHL','KPHX','KPIA','KPIE','KPIT','KPMP','KPNE','KPNS','KPOB','KPOU','KPRB','KPRC','KPSM','KPSP','KPTD','KPTK','KPUB','KPUJ','KPVD','KPVU','KPWA','KPWK','KPYM','KRAC','KRAL','KRBD','KRBL','KRDG','KRDU','KRFD','KRHV','KRIR','KRMN','KRND')) GROUP BY bin_icao24")
    # print(q.groupby)