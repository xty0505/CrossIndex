import time
import os

import pandas as pd
from tqdm import tqdm

from query import Query
from vizcube_parallel import VizCube
from type_parallel import *

if __name__ == '__main__':
    cube = VizCube('trace', [['lng', 'lat'], 'link_id', 'vehicle_id', 'timestep'], ['spatial', 'categorical', 'categorical', 'categorical'])
    cube.build_parallel('../data/trace.csv', ',')


