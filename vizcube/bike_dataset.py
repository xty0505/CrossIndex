import time

import pandas as pd

from query import Query
from vizcube import VizCube
import datetime

# dataset = pd.read_csv('../data/Divvy_Trips.csv', encoding='utf-8')
# print('pd.read_csv finished.')

bike_vizcube = VizCube('bike',
                       [['FROM LONGITUDE', 'FROM LATITUDE'], 'USER TYPE', 'START TIME'],
                       ['spatial', 'categorical', 'temporal'])
bike_vizcube.load('../cube', 'bike')
print(bike_vizcube.R.head(10))
sql = "SELECT COUNT(TRIP ID) from bike WHERE geohash = 'dp3tuy8y' GROUP BY geohash"
q = Query(measure='', agg='', groupby='', cube=bike_vizcube)
q.parse(sql)
start1 = time.time()
bike_vizcube.query2(q)
end1 = time.time()

# print(datetime.datetime.strptime('01/01/2017 12:18:00 PM', '%m/%d/%Y %H:%M:%S %p'))