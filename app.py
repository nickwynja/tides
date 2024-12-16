from flask import Flask, request, redirect, abort, render_template, make_response, url_for, jsonify, send_from_directory
from urllib.parse import urlparse
import logging
import re
import requests
import requests_cache
import json
from datetime import datetime, timedelta
import pandas as pd
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
from pprint import pprint as print
import math
import xml.etree.ElementTree as ET
import pytz
from turbo_flask import Turbo
import multiprocessing
from skyfield import almanac
from skyfield.api import load, wgs84
import time
import hashlib
import os
from flask_compress import Compress
import numpy as np
# import openmeteo_requests
# from retry_requests import retry

app = Flask(__name__)
turbo = Turbo(app)
Compress(app)

# https://requests-cache.readthedocs.io/en/stable/user_guide/general.html#patching
requests_cache.install_cache()

# cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
# retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
# openmeteo = openmeteo_requests.Client(session = retry_session)

NOAA_TC_API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
CACHE_DIR = "cache"
COMPASS=["N","NNE","NE","ENE","E","ESE", "SE", "SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]


if not app.debug:
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)


DEBUG_CACHE_SOLUNAR = True
## Should always be True if prod (app.debug = False)
CACHE_ENABLED = False if DEBUG_CACHE_SOLUNAR is False and app.debug else True

def get_current_stations_by_group(region_filter):
    current_station_group = requests.get(
            f"https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/current_geo_groups.json",
            )

    current_stations = {
            "0": [],
            "1": [],
            "2": [],
            "3": [],
            "4": [],
            "5": [],
            }

    for st in current_station_group.json()['currPredGeoGroupList']:
        current_stations[st['level']].append(st)

    stations_by_group = {}

    region_id = [x['groupID'] for x in current_stations['2'] if x['stationName'] == region_filter][0]

    for grp in current_stations["3"]:
            if grp['parentGroupID'] == region_id:
                stations_by_group[grp['groupID']] = {
                    "meta": grp,
                    "stations": []
                    }

    for st in current_stations['4']:
        if st['stationID'] is not None:  # look into whether bin 1 is what I want
            try:
                stations_by_group[st['parentGroupID']]['stations'].append(st)
            except KeyError as e:
                pass

    return stations_by_group

def get_states_by_region(region=444):
    states = requests.get(
            f"https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/current_geo_groups.json",
            )

    return [x['stationName'] for x in states.json()['currPredGeoGroupList'] if x['parentGroupID'] == region]

def get_station_by_id(station_id):
    if station_id == "" or station_id is None:
        return None

    try:
        station = requests.get(
                f"https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{station_id}.json",
                )

        station.raise_for_status()
    except requests.exceptions.HTTPError as err:
        return None

    return station.json()['stations'][0]

def get_tide_stations_by_group(region_filter):
    tide_stations_by_group = {}

    tide_station_state = requests.get(
            f"https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/geogroups.json?type=ETIDES&lvl=5",
            )

    st = tide_station_state.json()['geoGroupList']

    region_id = [x['geoGroupId'] for x in st if x['geoGroupName'] == region_filter][0]

    tide_station_children = requests.get(
            f"https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/geogroups/{region_id}/children.json",
            )

    st = tide_station_children.json()['stationList']

    for g in st:
        if g['geoGroupId'] != region_id:
            tide_stations_by_group[g['geoGroupId']] = {
                'meta': g,
                'stations': [],
                }

    for s in st:
        if s['stationId'] is not None:  # look into whether bin 1 is what I want
            try:
                tide_stations_by_group[s['parentGeoGroupId']]['stations'].append(
                        s
                        )
            except KeyError as e:
                pass

    return tide_stations_by_group



def sun_vlines(sun):
    vlines = []
    for s in sun:
        vlines = vlines + [
            dict(
                type='line',
                x0=s['dawn'],
                x1=s['dawn'],
                yref="paper",
                y0=0,
                y1=1,
                line=dict(color="blue", width=2, dash="dash"),
                ),
            dict(
                type='line',
                x0=s['rise'],
                x1=s['rise'],
                yref="paper",
                y0=0,
                y1=1,
                line=dict(color="orange", width=2, dash="dash"),
                ),
            dict(
                type='line',
                x0=s['set'],
                x1=s['set'],
                yref="paper",
                y0=0,
                y1=1,
                line=dict(color="orange", width=2, dash="dash"),
                ),
            dict(
                type='line',
                x0=s['dusk'],
                x1=s['dusk'],
                yref="paper",
                y0=0,
                y1=1,
                line=dict(color="blue", width=2, dash="dash"),
                ),
            ]
    return vlines

def sun_annots(sun):
    annots = []
    for s in sun:
        annots = annots + [
                dict(x=s['dawn'], yref="paper", y=0, text=s['dawn'].strftime("%H:%M"),
                     showarrow=False,
                     xshift=-20),
                dict(x=s['rise'], yref="paper", y=0, text=s['rise'].strftime("%H:%M"),
                     showarrow=False,
                     xshift=20),
                dict(x=s['set'], yref="paper", y=0, text=s['set'].strftime("%H:%M"),
                     showarrow=False,
                     xshift=-20),
                dict(x=s['dusk'], yref="paper", y=0, text=s['dusk'].strftime("%H:%M"),
                     showarrow=False,
                     xshift=20),
                ]
    return annots

def improve_text_position(x):
    """ it is more efficient if the x values are sorted """
    positions = ['bottom center', 'top center']  # you can add more: left center ...
    return [positions[i % len(positions)] for i in range(len(x))]

def seconds_until_hour():
    delta = timedelta(hours=1)
    now = datetime.now()
    next_hour = (now + delta).replace(microsecond=0, second=0, minute=0)
    wait_seconds = (next_hour - now).seconds
    return wait_seconds

def deg_to_compass(num):
    if num is None:
        return ""
    num=int(num)
    val=int((num/22.5)+.5)
    return COMPASS[(val % 16)]

def mps_to_kt(mps):
    knots = float(mps) * 1.94384449
    return round(knots)

def c_to_f(celsius):
    fahrenheit = (float(celsius) * 9/5) + 32
    return round(fahrenheit)

def until_next_event(row, df):
    if row['Knots'] == 0 :
        current = f"Slack at {row['Time']}"
        event = "Building"
    else:
        current = f"{row['Knots']} kt at {row['Time']}"
        event = "Waning"
    try:
        next_event = df.iloc[row.name + 1]['Date']
        time_until = (next_event - pd.to_datetime(row['Date']))
        until_string = f"{time_until.seconds // 3600}h {(time_until.seconds//60)%60}m"
        until = f"{event} for {until_string} until {next_event.strftime('%H:%M')}"
        return f"{current}<br>{until}"
    except IndexError as e:
        return None

def get_stations_from_bbox(lat_coords, lon_coords, station_type=None):
    """ from https://github.com/GClunies/noaa_coops"""
    station_list = []
    if station_type:
        data_url = f"https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json?type={station_type}"
    else:
        data_url = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"
    response = requests.get(data_url)
    json_dict = response.json()

    if len(lat_coords) != 2 or len(lon_coords) != 2:
        raise ValueError("lat_coords and lon_coords must be of length 2.")

    # Ensure lat_coords and lon_coords are in the correct order
    lat_coords = sorted(lat_coords)
    lon_coords = sorted(lon_coords)

    # Find stations in bounding box
    for station_dict in json_dict["stations"]:
        if lon_coords[0] < station_dict["lng"] < lon_coords[1]:
            if lat_coords[0] < station_dict["lat"] < lat_coords[1]:
                station_list.append(station_dict)

    return station_list

def get_stations_details_from_group(ids, station_type=None):
    """ from https://github.com/GClunies/noaa_coops"""
    station_list = []
    if station_type:
        data_url = f"https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json?type={station_type}"
    else:
        data_url = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"
    response = requests.get(data_url)
    json_dict = response.json()

    # Find stations in bounding box
    for station_dict in json_dict["stations"]:
        if station_dict['id'] in ids:
            station_list.append(station_dict)

    return station_list

def calc_tide_offset(row, offset):
    return pd.to_datetime(row['Date']) + timedelta(minutes=offset)

def deg_to_phase_primary(deg):
    if int(deg) in range(0, 90):
        phase = 'New Moon'
    elif int(deg) in range(90, 180):
        phase = 'First Quarter'
    elif int(deg) in range(180, 270):
        phase = 'Full Moon'
    elif int(deg) in range(270, 360):
        phase = 'Last Quarter'
    else:
        pass
    return phase

def deg_to_phase_intermediate(deg):
    if int(deg) in range(45, 90):
        phase = 'Waxing Crescent'
    elif int(deg) in range(135, 180):
        phase = 'Waxing Gibbous'
    elif int(deg) in range((360 - 135), (360-90)):
        phase = 'Waning Gibbous'
    elif int(deg) in range((360 - 45), 360):
        phase = 'Waning Crescent'
    else:
        phase = ""

    return phase


def solar(date, lat, lon):
    key = cache_key(f"solar_{date}{lat}{lon}")
    cache = get_obj_from_cache(key)
    if cache and CACHE_ENABLED:
        return cache


    ts = load.timescale()
    eph = load('de421.bsp')
    sun = eph['Sun']
    topos = wgs84.latlon(lat, lon)
    et = pytz.timezone('US/Eastern')

    day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(hours=24)
    t0 = ts.from_datetime(et.localize(day_start))
    t1 = ts.from_datetime(et.localize(day_end))

    f = almanac.dark_twilight_day(eph, topos)
    times, events = almanac.find_discrete(t0, t1, f)

    d =  {
            'dawn': times[1].astimezone(et),
            'rise': times[3].astimezone(et),
            'set': times[4].astimezone(et),
            'dusk': times[7].astimezone(et),
            }

    store_obj_in_cache(key, d)

    return d


def lunar(start_date, end_date, lat, lon):

    key = cache_key(f"lunar_{start_date}{lat}{lon}")
    cache = get_obj_from_cache(key)
    if cache and CACHE_ENABLED:
        return cache

    ts = load.timescale()
    eph = load('de421.bsp')
    moon = eph['Moon']
    topos = wgs84.latlon(lat, lon)
    observer = eph['Earth'] + topos
    et = pytz.timezone('US/Eastern')

    day_start = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = end_date
    t0 = ts.from_datetime(day_start)
    t1 = ts.from_datetime(day_end)

    events = []
    mr,y = almanac.find_risings(observer, moon, t0, t1)
    mr_alt, mr_az, mr_distance = observer.at(mr).observe(moon).apparent().altaz()
    for t,az in zip(mr, mr_az.degrees):
        events.append({
            'event': 'rise',
            'time': t.astimezone(et),
            'degrees': az,
            'pos': deg_to_compass(az)
            })


    mtr = almanac.find_transits(observer, moon, t0, t1)
    mtr_alt, mtr_az,mtr_distance = observer.at(mtr).observe(moon).apparent().altaz()
    for t,a,az in zip(mtr, mtr_alt.degrees, mtr_az.degrees):
        mp = almanac.moon_phase(eph, t)
        illum = mp.degrees / 180 if mp.degrees / 180 < 1 else (360 / mp.degrees) - 1
        events.append({
            'event': 'transit',
            'time': t.astimezone(et),
            'degrees': mp.degrees,
            'pos': f"{int(a)}&deg; {deg_to_compass(az)}",
            'illumination': illum,
            'phase_primary': deg_to_phase_primary(mp.degrees),
            'phase_intermediate': deg_to_phase_intermediate(mp.degrees),
            'text': f"{int(a)}&deg; {deg_to_compass(a)}",
            })
        events.append({
            'event': 'offset',
            'time': t.astimezone(et) + timedelta(hours=12),
            })

    ms,y = almanac.find_settings(observer, moon, t0, t1)
    ms_alt, ms_az, ms_distance = observer.at(ms).observe(moon).apparent().altaz()
    for t,a in zip(ms, ms_az.degrees):
        events.append({
            'event': 'set',
            'time': t.astimezone(et),
            'pos': deg_to_compass(a),
            })


    store_obj_in_cache(key, events)

    return events


def get_obj_from_cache(key):
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        with open(f"{CACHE_DIR}/{key}.json") as f:
            j = json.loads(f.read(), object_hook=date_hook)
            return j
    except FileNotFoundError as e:
        return False

def store_obj_in_cache(key, obj):
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        with open(f"{CACHE_DIR}/{key}.json", "w") as file:
            json.dump(obj, file, indent=4, default=json_serial)
    except Exception as e:
        # app.logger.error(e)
        return False


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError ("Type %s not serializable" % type(obj))


def date_hook(json_dict):
    for (key, value) in json_dict.items():
        try:
            json_dict[key] = datetime.fromisoformat(value)
        except Exception as e:
            # app.logger.error(e)
            pass
    return json_dict

def cache_key(s):
    # return hashlib.md5(s.encode('utf-8')).hexdigest()
    return s

def get_moon_phases(start, end):
    ts = load.timescale()
    et = pytz.timezone('US/Eastern')
    t0 = ts.from_datetime(start)
    t1 = ts.from_datetime(end)
    eph = load('de421.bsp')
    t, y = almanac.find_discrete(t0, t1, almanac.moon_phases(eph))
    phases = []

    for t,p in zip(t.astimezone(et),[almanac.MOON_PHASES[yi] for yi in y]):
        phases.append({
                "date": t,
                "phase": p,
                })
    return phases

def get_metadata_for_station(station_id):
    station_metadata = requests.get(
            f"https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{station_id}.json",
            )

    station_metadata = station_metadata.json()

    return station_metadata['stations'][0]


def process_marine_forecast(fcm):
    fc = []
    if fcm['location']['county'] != "marine":
        return fc

    for f in fcm['data']['text']:
        fx = {}
        for l in f.split('.'):
            l = l.strip()
            if 'wind' in l.lower():
                if l.split(' ')[0] in COMPASS:
                    wind_dir = l.split(' ')[0]
                    s = l.removeprefix(f"{wind_dir} wind ")
                else:
                    if "Winds could gust" in l:
                        # wind_dir = ""
                        fx['gusts'] = re.findall("\d+", l)[0]
                    if l.split(' ')[0] == "Variable":
                        wind_dir = "-"
                        s = l.removeprefix(f"Variable winds ")
                fx['wind_dir'] = wind_dir
                ss = s.split('kt', 1)
                s = ss[0].strip()
                s = s.replace("Winds could gust as high as ", "")
                s = s.replace("less than ", '<')
                s = s.replace(" to ", '-')
                s = s.replace("around ", '')
                fx['wind_speed'] = s
                if len(ss) > 1:
                    if 'gusts' in ss[1]:
                        fx['gusts'] = re.findall("\d+", ss[1])[0]
                    else:
                        fx['wind_more'] = ss[1].strip()
            if 'seas' in l.lower():
                s = l.removeprefix('Seas ')
                s = s.removeprefix('around')
                s = s.replace(' ft', '')
                s = s.replace(" building to", "' then ")
                s = s.replace(" subsiding to", "' then ")
                s = s.replace(' or less', ">")
                s = s.replace(' to ', '-')
                # fx['seas'] = s.replace(' ', '')
                fx['seas'] = s.strip()

        fc.append(fx)
    return fc


@app.route('/', methods=["GET", "POST"])
def tides():

    timer_start = time.perf_counter()

    station_defaults = {
            'tide': '8510560',
            'current': 'ACT2401',
            'met': "8510560",
            }


    if request.method == "POST":
        current_param = request.form.get('current')
        tide_param = request.form.get('tide')
        met_param = request.form.get('met')
        offset_param = request.form.get('offset')
        fav = request.form.get('fav', None)
        fav_state = request.form.get('fav_state')

        fav_string = f"{tide_param}:{current_param}" if current_param != "None" else tide_param

        if fav and fav != fav_state:
            update_fav = True
            set_fav = True
        elif fav is None and fav_state == "on":
            update_fav = True
            set_fav = False
        else:
            update_fav = None

        resp = make_response(redirect(url_for('tides',
                                              current=current_param if current_param != station_defaults['current'] else None,
                                              tide=tide_param if tide_param != station_defaults['tide'] else None,
                                              met=met_param if met_param != station_defaults['met'] else None,
                                              offset=offset_param if offset_param != "" else None,
                                              )
                                      )
                             )

        if update_fav:
            favs = json.loads(request.cookies.get('favs', '[]'))
            if set_fav and fav_string not in favs:
                favs.append(fav_string)
            if set_fav is False and fav_string in favs:
                favs.remove(fav_string)
            resp.set_cookie('favs',
                            json.dumps(favs,
                                       separators=(',', ':')),
                            max_age=157784760)

        return resp


    DAYS = 7

    tz = 'US/Eastern'
    EASTERN = pytz.timezone(tz)
    local_now = datetime.now(pytz.timezone(tz))
    tz_offset = int(local_now.utcoffset().total_seconds()/60/60)
    start_date_dt = (local_now).replace(microsecond=0, second=0, minute=0, hour=0)
    end_date_dt = (local_now + timedelta(days = DAYS))
    start_date = start_date_dt.strftime('%Y%m%d')
    end_date = end_date_dt.strftime('%Y%m%d')
    today = local_now.strftime('%Y-%m-%d')
    fig = go.Figure()

    bbox = {
            "lat": [
                40.7435,
                41.3655
                ],
            "lon": [
                -71.7354,
                -72.7853,
                ]
            }

    state = "New York"
    states = get_states_by_region('444')

    current_station_group = get_current_stations_by_group(state)
    tide_station_group = get_tide_stations_by_group(state)

    current_cookie = request.cookies.get('current', station_defaults['current'])
    tide_cookie = request.cookies.get('tide', station_defaults['tide'])

    station_offsets = json.loads(request.cookies.get('station_offsets', '{}'))

    current_param = request.args.get('current', current_cookie )
    tide_param = request.args.get('tide', tide_cookie)
    met_param = request.args.get('met', station_defaults['met'])

    offset_param = (request.args.get('offset')
                    if request.args.get('offset') != None else
                    station_offsets.get(f"{tide_param}_{current_param}", 0)
                    )

    tide_offset = int(offset_param) if offset_param is not None else 0

    if tide_param != 0:
        station_offsets[f"{tide_param}_{current_param}"] = tide_offset

    try:
        current_station = get_station_by_id(current_param)
    except IndexError as e:
        current_station = None

    try:
        tide_station = get_station_by_id(tide_param)
    except IndexError as e:
        current_station = None

    favs = json.loads(request.cookies.get('favs', '[]'))
    fav_string = f"{tide_param}:{current_param}" if current_station is not None else tide_param
    is_fav = True if fav_string in favs else False


    station_for_metadata = current_station if current_station is not None else tide_station

    station_metadata = get_metadata_for_station(station_for_metadata['id'])

    lat = station_metadata['lat']
    lon = station_metadata['lng']

    app.logger.debug(f"{(time.perf_counter()-timer_start):.2f}: " +
            'metadata collected'
            )

    forecast_daily = requests.get(
            f"https://marine.weather.gov/MapClick.php?lat={lat}&lon={lon}&FcstType=json",
            expire_after=seconds_until_hour(),
            )

    hazards = []
    zone = forecast_daily.json()['location']['zone']
    if 'hazard' in forecast_daily.json()['data']:

        hazards_data = requests.get(
                f"https://api.weather.gov/alerts/active?zone={zone}",
                expire_after=seconds_until_hour(),
                )

        for h in hazards_data.json()['features']:
            if 'properties' in h:
                h = h['properties']
                # print(h)
                if h['severity'] != "Minor":
                    app.logger.info(f"hazard severity: {h['severity']}")
                    hazards.append({
                            "effective": pd.to_datetime(h['effective']),
                            "headline": h['headline'].removesuffix(f" by {h['senderName']}"),
                            "ends": pd.to_datetime(h['ends']),
                            "expires": pd.to_datetime(h['expires']),
                            "event": h['event'],
                            "description": h['description'],
                            })

    app.logger.debug(f"{(time.perf_counter()-timer_start):.2f}: " +
                     'forecast collected'
                     )
    met_data = {
            'Water Temp': {},
            'Wind Speed': {},
            'Wind Dir': {},
            'Wind Gusts': {},
            'Air Temp': {},
            'Pressure': {},
            'Updated': {},
            }

    for st in ['NWHC3','MTKN6']:

        buoy_data = requests.get(
                f"https://www.ndbc.noaa.gov/data/realtime2/{st}.txt",
                expire_after=seconds_until_hour(),
                )

        db = pd.read_csv(
                StringIO(buoy_data.text),
                sep='\s+',
                nrows=24
                )
        db = db.replace('MM', np.nan)
        db = db.bfill()

        bl = db.loc[1]

        buoy_time = pd.to_datetime(f"{bl['#YY']}-{bl['MM']}-{bl['DD']}-{bl['hh']}:{bl['mm']}")
        buoy_time = pytz.utc.localize(buoy_time).astimezone(EASTERN)
        met_data['Updated'][st] = buoy_time.strftime("%H:%M")
        met_data['Wind Speed'][st] = f"{mps_to_kt(bl['WSPD'])} kt" if not pd.isnull(bl['WSPD']) else "-"
        met_data['Wind Dir'][st] = f"{deg_to_compass(bl['WDIR'])}" if not pd.isnull(bl['WDIR']) else "-"
        met_data['Wind Gusts'][st] = f"{mps_to_kt(bl['GST'])} kt" if not pd.isnull(bl['GST']) else "-"
        met_data['Pressure'][st] = f"{bl['PRES']} hPa" if not pd.isnull(bl['PRES']) else "-"
        met_data['Air Temp'][st] = f"{c_to_f(bl['ATMP'])}&deg;F" if not pd.isnull(bl['ATMP']) else "-"
        met_data['Water Temp'][st] = f"{c_to_f(bl['WTMP'])}&deg;F" if not pd.isnull(bl['WTMP']) else "-"


    app.logger.debug(f"{(time.perf_counter()-timer_start):.2f}: " +
            'buoy collected'
                     )

    tides = requests.get(
            f"{NOAA_TC_API}?product=predictions&application=NOS.COOPS.TAC.WL&begin_date={start_date}&end_date={end_date}&datum=MLLW&station={tide_station['id']}&time_zone=lst_ldt&units=english&interval=hilo&format=json",
            )

    dt = pd.DataFrame.from_dict(tides.json()['predictions'])
    dt = dt.rename(columns={
        't': 'Date',
        'v': 'Feet',
        'type': "Type"
        })

    dt['Type'] = dt['Type'].replace(['H'], 'high tide')
    dt['Type'] = dt['Type'].replace(['L'], 'low tide')

    if tide_offset != 0:
        dt['Date'] = dt.apply(calc_tide_offset, axis=1, offset=tide_offset)
    else:
        dt['Date'] = pd.to_datetime(dt['Date'])

    dt['Feet'] = dt['Feet'].astype("float")
    dt['Time'] = dt['Date'].dt.strftime("%H:%M")
    dt['Day'] = dt['Date'].dt.strftime("%Y-%m-%d")

    # print(dt)

    tide_by_day = {}

    for i,r in dt.iterrows():
        d = EASTERN.localize(r['Date'])
        if r['Type'] == 'high tide':
            high = tide_by_day.setdefault('high', [])
            high.append({
                'time': d,
                'feet': r['Feet']
                })
        if r['Type'] == 'low tide':
            low = tide_by_day.setdefault('low', [])
            low.append({
                'time': d,
                'feet': r['Feet']
                })

        group = tide_by_day.setdefault(r['Day'], {})
        height = group.setdefault('feet', [])
        height.append(round(r['Feet'], 1))

    tide_max = dt['Feet'].max()  # used in chart for max moon %

    fig.add_trace(
      go.Scatter(
        x=dt['Date'], y=dt['Feet'],
        text=dt['Time'],
        hovertemplate = '%{y:.1f} ft' + '<br>%{x}',
        mode='lines+markers+text',
        textposition='top center', # Adjust text position
        line_shape="spline",
        name='Tide'
      )
    )

    app.logger.debug(f"{(time.perf_counter()-timer_start):.2f}: " +
                     'tides charted'
                     )

    if current_station is not None:

        currents = requests.get(
                f"{NOAA_TC_API}?product=currents_predictions&application=NOS.COOPS.TAC.WL&begin_date={start_date}&end_date={end_date}&datum=MLLW&station={current_station['id']}&time_zone=lst_ldt&units=english&interval=MAX_SLACK&format=json",
                )

        c = currents.json()
        if 'error' not in c:
            c = c['current_predictions']['cp']
            dc = pd.DataFrame.from_dict(c)
            dc = dc.rename(columns={
                'Time': 'Date',
                'Velocity_Major': 'Knots'
                })

            dc['Date'] = pd.to_datetime(dc['Date'])
            dc['Time'] = dc['Date'].dt.strftime("%H:%M")
            dc['Event'] = dc.apply(until_next_event, df=dc, axis=1)
            dc['Type'] = dc['Type'].replace(['ebb'], 'max ebb')
            dc['Type'] = dc['Type'].replace(['flood'], 'max flood')
            dc['Type'] = dc['Type'].replace(['slack'], 'slack current')

            fig.add_trace(
                go.Scatter(
                  x=dc['Date'], y=dc['Knots'],
                  text=dc['Event'],
                  texttemplate=dc['Time'],
                  hovertemplate = "%{text}",
                  mode='lines+markers',
                  textposition='top center', # Adjust text position
                  line_shape="spline",
                  name='Current'
                )
            )

            fig.update_traces(
                    textposition=improve_text_position(dc['Time']),
                    )
        else:
            app.logger.debug(c['error']['message'])

        app.logger.debug(f"{(time.perf_counter()-timer_start):.2f}: " +
            'currents charted'
            )

    # url = "https://api.open-meteo.com/v1/forecast"
    # params = {
    #     "latitude": lat,
    #     "longitude": lon,
    #     "current": ["temperature_2m", "cloud_cover", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "pressure_msl"],
    #     "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation_probability", "precipitation", "pressure_msl", "cloud_cover", "visibility", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
    #     "daily": ["temperature_2m_max", "temperature_2m_min", "wind_speed_10m_max",
    #               "wind_gusts_10m_max", "wind_direction_10m_dominant",
    #               "precipitation_probability_max", "precipitation_sum"],
    #     "temperature_unit": "fahrenheit",
    #     "wind_speed_unit": "kn",
    #     "precipitation_unit": "inch",
    #     "timezone": "America/New_York"
    # }

    # forecast_responses = openmeteo.weather_api(url, params=params)
    # forecast_response = forecast_responses[0]
    # forecast_current = forecast_response.Current()
    # forecast_hourly = forecast_response.Hourly()
    # forecast_daily = forecast_response.Daily()

    # url = "https://marine-api.open-meteo.com/v1/marine"
    # params = {
    #     "latitude": lat,
    #     "longitude": lon,
    #     "current": ["wave_height", "wave_direction", "wave_period"],
    #     "hourly": ["wave_height", "wave_direction", "wave_period"],
    #     "daily": ["wave_height_max", "wave_direction_dominant", "wave_period_max"],
    #     "length_unit": "imperial",
    #     "wind_speed_unit": "kn",
    #     "timezone": "America/New_York"
    # }

    # marine_forecast = openmeteo.weather_api(url, params=params)
    # marine_response = marine_forecast[0]
    # marine_current = marine_response.Current()
    # marine_daily = marine_response.Daily()
    # marine_hourly = marine_response.Hourly()

    # forecast = {
    #         'current': {
    #             'temperature_2m': forecast_current.Variables(0).Value(),
    #             'cloud_cover': forecast_current.Variables(1).Value(),
    #             'wind_speed_10m': forecast_current.Variables(2).Value(),
    #             'wind_direction_10m': forecast_current.Variables(3).Value(),
    #             'wind_gusts_10m': forecast_current.Variables(4).Value(),
    #             'pressure_msl': forecast_current.Variables(5).Value(),
    #             'wave_height': marine_current.Variables(0).Value(),
    #             'wave_direction': marine_current.Variables(1).Value(),
    #             'wave_period': marine_current.Variables(2).Value(),
    #             },
    #         'hourly': {
    #             'temperature_2m': forecast_hourly.Variables(0).ValuesAsNumpy(),
    #             'relative_humidity_2m': forecast_hourly.Variables(1).ValuesAsNumpy(),
    #             'precipitation_probability': forecast_hourly.Variables(2).ValuesAsNumpy(),
    #             'precipitation': forecast_hourly.Variables(3).ValuesAsNumpy(),
    #             'pressure_msl': forecast_hourly.Variables(4).ValuesAsNumpy(),
    #             'cloud_cover': forecast_hourly.Variables(5).ValuesAsNumpy(),
    #             'visibility': forecast_hourly.Variables(6).ValuesAsNumpy(),
    #             'wind_speed_10m': forecast_hourly.Variables(7).ValuesAsNumpy(),
    #             'wind_direction_10m': forecast_hourly.Variables(8).ValuesAsNumpy(),
    #             'wind_gusts_10m': forecast_hourly.Variables(9).ValuesAsNumpy(),
    #             'wave_height': marine_hourly.Variables(0).ValuesAsNumpy(),
    #             'wave_direction': marine_hourly.Variables(1).ValuesAsNumpy(),
    #             'wave_period': marine_hourly.Variables(2).ValuesAsNumpy(),
    #             },
    #         'daily': {
    #             'temperature_2m_max': forecast_daily.Variables(0).ValuesAsNumpy(),
    #             'temperature_2m_min': forecast_daily.Variables(1).ValuesAsNumpy(),
    #             'wind_speed_10m_max': forecast_daily.Variables(2).ValuesAsNumpy(),
    #             'wind_gusts_10m_max': forecast_daily.Variables(3).ValuesAsNumpy(),
    #             'wind_direction_10m_dominant': forecast_daily.Variables(4).ValuesAsNumpy(),
    #             'precipitation_probability_max': forecast_daily.Variables(5).ValuesAsNumpy(),
    #             'precipitation_sum': forecast_daily.Variables(6).ValuesAsNumpy(),
    #             'wave_height_max': marine_daily.Variables(0).ValuesAsNumpy(),
    #             'wave_direction_dominant': marine_daily.Variables(1).ValuesAsNumpy(),
    #             'wave_period_max': marine_daily.Variables(2).ValuesAsNumpy(),
    #             },
    #         }

    # hourly_data = {"date": pd.date_range(
    #     start = pd.to_datetime(forecast_hourly.Time(), unit = "s", utc = True),
    #     end = pd.to_datetime(forecast_hourly.TimeEnd(), unit = "s", utc = True),
    #     freq = pd.Timedelta(seconds = forecast_hourly.Interval()),
    #     inclusive = "left"),
    #     **forecast['hourly']
    # }

    # hourly_dataframe = pd.DataFrame(data = hourly_data)

    # daily_data = {"date": pd.date_range(
    #     start = pd.to_datetime(forecast_daily.Time(), unit = "s", utc = True),
    #     end = pd.to_datetime(forecast_daily.TimeEnd(), unit = "s", utc = True),
    #     freq = pd.Timedelta(seconds = forecast_daily.Interval()),
    #     inclusive = "left"),
    #   **forecast['daily']
    # }

    # daily_dataframe = pd.DataFrame(data = daily_data)

    # daily_dataframe['wind_direction_10m_dominant_compass'] = daily_dataframe['wind_direction_10m_dominant'].apply(deg_to_compass)
    # daily_dataframe['wave_direction_dominant_compass'] = daily_dataframe['wave_direction_dominant'].apply(deg_to_compass)
    # daily_dataframe['date'] = pd.to_datetime(daily_dataframe['date'])
    # daily_dataframe['day'] = daily_dataframe['date'].dt.strftime("%Y-%m-%d")

    # loc = "Local"
    # met_data['Updated'][loc] = local_now.strftime("%H:%M")
    # met_data['Wind Speed'][loc] = f"{round(forecast['current']['wind_speed_10m'])} kt"
    # met_data['Wind Dir'][loc] = f"{deg_to_compass(forecast['current']['wind_direction_10m'])}"
    # met_data['Wind Gusts'][loc] = f"{round(forecast['current']['wind_gusts_10m'])} kt"
    # met_data['Pressure'][loc] = f"{round(forecast['current']['pressure_msl'], 1)} hPa"
    # met_data['Air Temp'][loc] = f"{round(forecast['current']['temperature_2m'])}&deg;F"
    # met_data['Water Temp'][loc] = "-"

    tide_station_metadata = get_metadata_for_station(tide_station['id'])
    tide_lat = tide_station_metadata['lat']
    tide_lon = tide_station_metadata['lng']

    forecast_tide_station = requests.get(
            f"https://forecast.weather.gov/MapClick.php?lat={tide_lat}&lon={tide_lon}&FcstType=digitalDWML",
            expire_after=seconds_until_hour(),
            )

    tree = ET.ElementTree(ET.fromstring(forecast_tide_station.text))
    root = tree.getroot()
    times = [x.text for x in root.findall('.//start-valid-time')]
    # wind_speeds = [x.text for x in root.findall('.//wind-speed[@type="sustained"]/value')]
    # wind_gusts = [x.text for x in root.findall('.//wind-speed[@type="gust"]/value')]
    # wind_dir = [x.text for x in root.findall('.//direction[@type="wind"]/value')]
    # waves = [x.text for x in root.findall('.//waves[@type="significant"]/value')]
    temp = [x.text for x in root.findall('.//temperature[@type="hourly"]/value')]

    wind_annots = []
    fc = []

    if current_station is not None:
        current_station_metadata = get_metadata_for_station(current_station['id'])
        current_lat = current_station_metadata['lat']
        current_lon = current_station_metadata['lng']
        forecast_marine_daily = requests.get(
                f"https://forecast.weather.gov/MapClick.php?lat={current_lat}&lon={current_lon}&FcstType=json",
                expire_after=seconds_until_hour(),
                )

        print(forecast_marine_daily.url)

        fc = process_marine_forecast(forecast_marine_daily.json())

        forecast_marine_hourly = requests.get(
               f"https://forecast.weather.gov/MapClick.php?lat={current_lat}&lon={current_lon}&FcstType=digitalDWML",
               expire_after=seconds_until_hour(),
               )

        tree = ET.ElementTree(ET.fromstring(forecast_marine_hourly.text))
        root = tree.getroot()
        times = [x.text for x in root.findall('.//start-valid-time')]
        wind_speeds = [x.text for x in root.findall('.//wind-speed[@type="sustained"]/value')]
        wind_gusts = [x.text for x in root.findall('.//wind-speed[@type="gust"]/value')]
        wind_dir = [x.text for x in root.findall('.//direction[@type="wind"]/value')]
        waves = [x.text for x in root.findall('.//waves[@type="significant"]/value')]


        for t, ws, wg, wd, wv in zip(times, wind_speeds, wind_gusts, wind_dir, waves):
        # for t, ws, wg, wd, wv, temp in zip(times, wind_speeds, wind_gusts, wind_dir, waves, temp):
            if wg is not None:
                cond = f"{deg_to_compass(wd)}<br>{ws}-{wg} kt<br>{wv}'"
            else:
                cond = f"{deg_to_compass(wd)}<br>{ws} kt<br>{wv}'"
            # cond = f"{cond}<br>{temp}&deg;F"
            wind_annots = wind_annots + [dict(x=t, yref="paper", y=1, yshift=55, text=cond, showarrow=False, name="wind")]


    tide_station_forecast_urls = requests.get(
            f"https://api.weather.gov/points/{tide_lat},{tide_lon}",
            )

    tide_station_forecast = requests.get(
            tide_station_forecast_urls.json()['properties']['forecast'],
            expire_after=seconds_until_hour(),
            )

    forecast = tide_station_forecast.json()['properties']['periods']

    for i,d in enumerate(forecast):
        forecast[i]['startTime'] = datetime.fromisoformat(d['startTime'])
        forecast[i]['endTime'] = datetime.fromisoformat(d['endTime'])


    app.logger.debug(f"{(time.perf_counter()-timer_start):.2f}: " +
                     'wind annotations'
                     )

    sun = []
    for date in pd.date_range(start=start_date, end=end_date):
        sun.append(solar(date, lat, lon))

    app.logger.debug(f"{(time.perf_counter()-timer_start):.2f}: " +
                     'sun data calculated'
                     )

    fig.update_layout(shapes=sun_vlines(sun))
    fig.update_layout(annotations=sun_annots(sun) + wind_annots)

    app.logger.debug(f"{(time.perf_counter()-timer_start):.2f}: " +
                     'sun and wind annotations added'
                     )

    moon_events = lunar(start_date_dt, end_date_dt, lat, lon)
    moon_phases = get_moon_phases(
            start_date_dt,
            start_date_dt + pd.offsets.MonthEnd(2)
            )


    moon_forecast = {}
    for m in moon_phases:
        d = m['date'].strftime('%Y-%m-%d')
        moon_forecast[d] = m['phase']

    moon_data = []

    for m in moon_events:
            if m['event'] == 'transit':
                value = tide_max * m['illumination']
                text = (f"Moon upper transit at {m['time'].strftime('%H:%M')}<br>"
                        + f"Phase: {m['phase_primary']}<br>"
                        + f"{round(m['degrees'])}&deg; {m['phase_intermediate']}<br>"
                        + f"Illumination: {round(m['illumination'] * 100)}% <br>"
                        + f"Elevation: {m['pos']}")
            elif m['event'] == 'offset':
                value = None
            else:
                value = 0
                text = (f"Moon {m['event']} at {m['time'].strftime('%H:%M')}<br>"
                        +f"Direction: {m['pos']}")

            moon_data.append({
                'time': m['time'],
                'phen': m['event'],
                'fracillum': m['illumination'] if 'illumination' in m else 0,
                'value': 0,
                'value': value,
                'text': text,
                  })

    moon_data = sorted(moon_data, key=lambda d: d['time'])


    # fix curve at start of day
    if moon_data[0]['phen'] == 'set':
        del moon_data[0]

    dm = pd.DataFrame(moon_data)

    fig.add_trace(
        go.Scatter(
          x=dm['time'], y=dm['value'],
          text=dm['text'],
          texttemplate=dm['text'],
          hovertemplate = "%{text}",
          mode='lines+markers',
          textposition='top center', # Adjust text position
          line_shape="spline",
          line_color="#888",
          name='Moon'
        )
    )

    app.logger.debug(f"{(time.perf_counter()-timer_start):.2f}: " +
                     'moon data charted'
                     )

    fig.update_traces(
            connectgaps=None,
            )

    fig.update_yaxes(
     fixedrange = True,
    )

    if "iPhone" in request.headers.get('User-Agent'):
        range = [
                local_now - timedelta(hours=1),
                local_now + timedelta(hours=4)
                ]
        height = 450
    else:
        range = [
                local_now - timedelta(hours=2),
                local_now + timedelta(hours=16)
                ]
        height = 600

    fig.update_xaxes(
        range=range,
        minallowed=start_date_dt,
        maxallowed=end_date_dt,
        automargin=False,
        ticks="outside",
        tickcolor= "black",
        tickangle=0,
    )

    fig.update_layout(
            template='plotly_dark',
            showlegend=False,
            height=height,
            dragmode='pan',
            margin=dict(l=0, r=0, t=50, b=25),
            xaxis_tickformatstops = [
                dict(dtickrange=[0, 60000 * 200], value="%H:%M"),
                dict(dtickrange=[60000 * 200, 86400000], value="%m/%d %H:%M"),
                ]
            )

    fig.add_vline(x=local_now, line_width=1, line_dash="dash", line_color='green')

    fig_html = fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            div_id="myPlot",
            config={
                'displayModeBar': False,
                })

    app.logger.debug(f"{(time.perf_counter()-timer_start):.2f}: " +
                     'figure updated and dumped'
                     )


    dtt = dt[dt['Date'].dt.strftime('%Y-%m-%d') == today]
    try:
        dct = dc[dc['Date'].dt.strftime('%Y-%m-%d') == today]
        dd = pd.merge(dct, dtt,  how="outer", on=['Date', 'Type'])
    except NameError as e:
        dd = dtt

    moon_rises = [x for x in moon_data if x['phen'] == "rise"]
    moon_set = [x for x in moon_data if x['phen'] == "set"]

    text = [
            {'time': local_now,
             'text': '<span style="color: green;">now</span>',
             },
            {'time': sun[1]['dawn'],
             'text': '<span style="color: blue;">dawn</span>',
             },
            {'time': sun[1]['rise'],
             'text': '<span style="color: orange;">sun rise</span>',
             },
            {'time': sun[1]['set'],
             'text': '<span style="color: orange;">sun set</span>',
             },
            {'time': sun[1]['dusk'],
             'text': '<span style="color: blue;">dusk</span>',
             },
            {'time': moon_rises[1]['time'],
             'text': "moon rise",
             },
            {'time': moon_set[1]['time'],
             'text': "moon set",
             },
            ]

    for idx,d in dd.iterrows():
            text.append({'time': EASTERN.localize(d['Date'].to_pydatetime()),
                         'text': d['Type']})

    text = sorted(text, key=lambda d: d['time'])

    app.logger.debug(f"{(time.perf_counter()-timer_start):.2f}: " +
                     'text report created'
                     )

    resp = make_response(render_template('index.html', fig_html=fig_html,
                                         current_station=current_station,
                                         tide_station=tide_station,
                                         current_station_group=current_station_group,
                                         tide_station_group=tide_station_group,
                                         met_data=met_data,
                                         tide_offset=tide_offset,
                                         text_report=text,
                                         is_fav=is_fav,
                                         favs=favs,
                                         moon_phases=moon_phases,
                                         hazards=hazards,
                                         forecast=forecast,
                                         fc=fc,
                                         tide_by_day=tide_by_day,
                                         sun=sun,
                                         moon_forecast=moon_forecast
                                         ))

    if current_param != station_defaults['current']:
        resp.set_cookie('current', current_param, max_age=157784760)

    if tide_param != station_defaults['tide']:
        resp.set_cookie('tide', tide_param, max_age=157784760)

    resp.set_cookie('station_offsets',
                        json.dumps(station_offsets,
                                   separators=(',', ':')),
                    max_age=157784760)

    app.logger.debug(f"{(time.perf_counter()-timer_start):.2f} w/ solunar cache = {CACHE_ENABLED}")

    return resp


@app.route('/manifest.json')
def manifest():
    return jsonify({
        "short_name": "Surfcast",
        "name": "Surfcast",
        "id": "/",
        "start_url": "/",
        "background_color": "#000000",
        "display": "fullscreen",
        "scope": "/",
        "theme_color": "#000000",
        "description": "Tide, current, wind and moon info for the surfcaster.",
        "icons": [
            {
                "src": "/static/icon.png",
                "type": "image/png",
                "sizes": "512x512"
                },
            ]
    })


@app.route('/service-worker.js')
def service_worker():
    return send_from_directory(app.root_path,
                                     'service-worker.js')


if __name__ == '__main__':
    app.run(debug=True)
