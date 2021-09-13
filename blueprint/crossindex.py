import os
import geohash2
from flask import Flask, Blueprint, g, request, flash, redirect, json, render_template

from query import Query, Condition
from resultset import ResultSet
from crossindex_main import CrossIndex, Type

cube = CrossIndex('')
UPLOAD_PATH = './data'
CUBE_PATH = './cube'
ALLOWED_EXTENSIONS = {'csv'}

q1 = Query(measure='', agg='', groupby='', cube=cube)
q2 = Query(measure='', agg='', groupby='', cube=cube)
q3 = Query(measure='', agg='', groupby='', cube=cube)
q4 = Query(measure='', agg='', groupby='', cube=cube)
Q = [q1, q2, q3, q4]

bp = Blueprint('vizcube', __name__, url_prefix='/vizcube')
app = Flask(__name__)
app.config['UPLOAD_PATH'] = UPLOAD_PATH
app.config['CUBE_PATH'] = CUBE_PATH


def get_query():
    if 'query' not in g:
        g.query = Q
    return g.query


def get_cube():
    if 'cube' not in g:
        g.cube = cube
    return g.cube


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@bp.route('/upload', methods=('POST', 'GET'))
def upload():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        if not os.path.exists(os.path.join(app.config['UPLOAD_PATH'], file.filename)):
            file.save(os.path.join(app.config['UPLOAD_PATH'], file.filename))
        return 'Uploaded'


@bp.route('/build', methods=('POST', 'GET'))
def build():
    if request.method == 'POST':
        cube = get_cube()
        if not cube.name == '':
            cube = CrossIndex('')
        data = request.get_json()
        cube.name = data['name']
        cube.dimensions = data['dimensions']
        cube.types = data['types']
        path = os.path.join(app.config['UPLOAD_PATH'], cube.name + '.csv')
        # lng,lat 合并为 [lng, lat]
        if 'spatial' in cube.types:
            lng_i = cube.types.index('spatial')
            lat_i = cube.types.index('spatial', lng_i + 1)
            lnglat = [cube.dimensions[lng_i], cube.dimensions[lat_i]]
            cube.dimensions[lng_i] = lnglat
            cube.dimensions.remove(cube.dimensions[lat_i])
            cube.types.remove(cube.types[lng_i])
            cube.types = [Type.getType(t) for t in cube.types]
        options = {
            "hash_length": 7
        }
        cube.build(path, ',', options)
        cube.save(app.config['CUBE_PATH'])
    return 'CrossIndex Built in ' + app.config['CUBE_PATH']


@bp.route('/build/status', methods=('POST', 'GET'))
def get_build_status():
    cube = get_cube()
    if cube.pbar.disable:
        return json.jsonify({'value': 100})
    else:
        return json.jsonify({'value': int(cube.pbar.last_print_n * 100 / cube.pbar.total)})


@bp.route('/load', methods=('POST', 'GET'))
def load():
    if request.method == 'POST':
        cube = get_cube()
        filename = request.form.get('filename')
        name = filename.split('.')[0]
        cube.load(app.config['CUBE_PATH'], name)
        return 'Cube Loaded'


@bp.route('/query', methods=('POST', 'GET'))
def query():
    cube = get_cube()
    Q = get_query()
    for q in Q:
        q.set_cube(cube)
    if cube.ready:
        categorical = query_with_request(cube, Type.categorical)
        temporal = query_with_request(cube, Type.temporal)
        spatial = query_with_request(cube, Type.spatial)
        result = {'categorical': categorical,
                  'temporal': temporal,
                  'spatial': spatial}
        return result
    else:
        raise Exception('No VizCube!(Build or Load Cube first)')


@bp.route('/backward_query', methods=('POST', 'GET'))
def backward_query():
    cube = get_cube()
    Q = get_query()
    if cube.ready:
        data = request.get_json()
        flag = data['flag']
        condition = [Condition(data['dimension'], data['value'], Type.getType(data['type']))]
        for q in Q:
            q.add_condition(condition[0])
            q.clear()

        if flag == 0:
            link_cnt_rs = cube.query(Q[0])
            vehicle_cnt_rs = cube.query(Q[1])
            timestep_cnt_rs = cube.query(Q[2])
            cube.query(Q[3])
            data, max = Q[3].get_geo_result(None)
        else:
            # backward_query()
            link_cnt_rs = cube.backward_query(Q[0], condition)
            vehicle_cnt_rs = cube.backward_query(Q[1], condition)
            timestep_cnt_rs = cube.backward_query(Q[2], condition)
            cube.backward_query(Q[3], condition)
            data, max = Q[3].get_geo_result(None)

        categorical = {0: link_cnt_rs, 1: vehicle_cnt_rs}
        temporal = timestep_cnt_rs
        spatial = {'data': data, 'max': max}
        return {'categorical': categorical,
                'temporal': temporal,
                'spatial': spatial}
    else:
        return 'No VizCube!(Build or Load Cube first)'


@bp.route('/query/spatial', methods=('POST', 'GET'))
def query_spatial():
    cube = get_cube()
    Q = get_query()
    if cube.ready:
        data = request.get_json()
        limit = data['limit']
        geo_length = data['geohashLength'] + 1
        lnglat = data['lnglat']
        geohash = geohash2.encode(lnglat['lat'], lnglat['lng'], geo_length)
        condition = Condition('geohash', geohash, Type.spatial)
        for q in Q:
            q.clear_conditions()
            q.add_condition(condition)
            q.clear()
        # query()
        link_cnt_rs = cube.query(Q[0])
        vehicle_cnt_rs = cube.query(Q[1])
        timestep_cnt_rs = cube.query(Q[2])
        cube.query(Q[3])
        # todo get_geo_result takes too long time, remove the timestep condition
        data, max = Q[3].get_geo_result(limit)

        categorical = {0: link_cnt_rs, 1: vehicle_cnt_rs}
        temporal = timestep_cnt_rs
        spatial = {'data': data, 'max': max}
        return {'categorical': categorical,
                'temporal': temporal,
                'spatial': spatial}
    else:
        return 'No VizCube!(Build or Load Cube first)'


def query_with_request(cube, type):
    if type == Type.categorical:
        sql = "SELECT COUNT(velocity) FROM trace WHERE geohash = 'wtw3sm' GROUP BY link_id"
        q1.parse(sql)
        link_cnt_result = cube.query(q1)
        sql = "SELECT COUNT(velocity) FROM trace WHERE geohash = 'wtw3sm' GROUP BY vehicle_id"
        q2.parse(sql)
        vehicle_cnt_result = cube.query(q2)
        return {0: link_cnt_result, 1: vehicle_cnt_result}
    elif type == Type.temporal:
        sql = "SELECT COUNT(velocity) FROM trace WHERE geohash = 'wtw3sm' GROUP BY timestep"
        q3.parse(sql)
        timestep_cnt_result = cube.query(q3)
        return timestep_cnt_result
    elif type == Type.spatial:
        sql = "SELECT COUNT(velocity) FROM trace WHERE geohash = 'wtw3sm' GROUP BY geohash"
        q4.parse(sql)
        cube.query(q4)
        data, max = q4.get_geo_result(None)
        return {'data': data, 'max': max}
