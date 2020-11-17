import time
import os

import pandas as pd
from tqdm import tqdm

from query import Query
from vizcube_parallel import VizCube
from type_parallel import *

if __name__ == '__main__':
    # trace
    # cube = VizCube('trace', [['lng', 'lat'], 'link_id', 'vehicle_id', 'timestep'], ['spatial', 'categorical', 'categorical', 'categorical'])
    # cube.build_parallel('../data/trace.csv', ',')

    # traffic
    # cube = VizCube('traffic', ['link_id', 'time'], ['categorical', 'categorical'])
    # cube.build_parallel('../data/traffic.csv', '\t')

    # bike
    cube = VizCube('bike', [['FROM LONGITUDE', 'FROM LATITUDE'], 'FROM STATION NAME', 'USER TYPE', 'START TIME'], ['spatial', 'categorical', 'categorical', 'categorical', 'temporal'])
    cube.build_parallel('../data/Divvy_Trips.csv', ',')

