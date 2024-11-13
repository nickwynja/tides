from flask import Flask, request, redirect, abort, render_template, make_response, url_for, jsonify
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

app = Flask(__name__)
turbo = Turbo(app)

# https://requests-cache.readthedocs.io/en/stable/user_guide/general.html#patching
requests_cache.install_cache()

NOAA_TC_API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
CACHE_DIR = "cache"

if not app.debug:
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)


DISABLE_CACHE = True if app.debug and True else False

app.logger.info(f"Disabled file cache: {DISABLE_CACHE}")

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
    num=int(num)
    val=int((num/22.5)+.5)
    arr=["N","NNE","NE","ENE","E","ESE", "SE", "SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
    return arr[(val % 16)]

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
    if cache and not DISABLE_CACHE:
        app.logger.info("\tsun  data from cache")
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
    if cache and not DISABLE_CACHE:
        app.logger.info("\tmoon data from cache")
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
        events.append({
            'event': 'transit',
            'time': t.astimezone(et),
            'degrees': mp.degrees,
            'pos': f"{int(a)}&deg; {deg_to_compass(az)}",
            'illumination': mp.degrees / 180 if mp.degrees / 180 < 1 else 180 / mp.degrees,
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


@app.route('/', methods=["GET", "POST"])
def tides():

    app.logger.info("load initiated")
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

        return redirect(url_for('tides',
                                current=current_param if current_param != station_defaults['current'] else None,
                                tide=tide_param if tide_param != station_defaults['tide'] else None,
                                met=met_param if met_param != station_defaults['met'] else None,
                                offset=offset_param if offset_param != "" else None,
                                )
                        )

    DAYS = 7

    tz = 'US/Eastern'
    EASTERN = pytz.timezone(tz)
    local_now = datetime.now(pytz.timezone(tz))
    tz_offset = int(local_now.utcoffset().total_seconds()/60/60)
    # start_date = local_now.strftime('%Y%m%d')
    start_date_dt = (local_now - timedelta(days = 1)).replace(microsecond=0, second=0, minute=0, hour=0)
    end_date_dt = (local_now + timedelta(days = DAYS))
    start_date = start_date_dt.strftime('%Y%m%d')
    end_date = end_date_dt.strftime('%Y%m%d')
    today = local_now.strftime('%Y-%m-%d')

    bbox = {
            "lat": [
                40.4915,
                41.2743
                ],
            "lon": [
                -71.4179,
                -73.9572,
                ]
            }

    local_current_stations = get_stations_from_bbox(
            lat_coords=bbox['lat'],
            lon_coords=bbox['lon'],
            station_type="currentpredictions"
            )

    local_tide_stations = get_stations_from_bbox(
            lat_coords=bbox['lat'],
            lon_coords=bbox['lon'],
            station_type="tidepredictions"
            )

    local_met_stations = get_stations_from_bbox(
            lat_coords=bbox['lat'],
            lon_coords=bbox['lon'],
            station_type="met"
            )

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

    current_station = [x for x in local_current_stations if x['id'] == current_param][0]
    tide_station = [x for x in local_tide_stations if x['id'] == tide_param][0]
    met_station = [x for x in local_met_stations if x['id'] == met_param][0]

    app.logger.info(time.perf_counter()-timer_start)

    app.logger.info("getting station metadata")
    station_metadata = requests.get(
            f"https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{current_station['id']}.json",
            )

    app.logger.info(f"  cached: {station_metadata.from_cache}")

    station_metadata = station_metadata.json()

    lat = station_metadata['stations'][0]['lat']
    lon = station_metadata['stations'][0]['lng']

    app.logger.info(time.perf_counter()-timer_start)

    app.logger.info("getting marine forecast")
    forecast_daily = requests.get(
            f"https://marine.weather.gov/MapClick.php?lat={lat}&lon={lon}&FcstType=json",
            expire_after=seconds_until_hour(),
            )

    app.logger.info(f"  cached: {forecast_daily.from_cache}")

    forecast_periods = forecast_daily.json()['time']['startPeriodName']
    forecast_text = forecast_daily.json()['data']['text']

    forecast = []

    for idx, text in enumerate(forecast_text[:DAYS*2]):
        forecast.append({
                "when": forecast_periods[idx],
                "text": text,
                })


    app.logger.info(time.perf_counter()-timer_start)

    app.logger.info("getting met data")
    water_temp = requests.get(
            f"{NOAA_TC_API}?product=water_temperature&application=NOS.COOPS.TAC.WL&date=latest&datum=MLLW&station={met_param}&time_zone=lst_ldt&units=english&interval=&format=json",
            expire_after=seconds_until_hour(),
            )
    air_temp = requests.get(
            f"{NOAA_TC_API}?product=air_temperature&application=NOS.COOPS.TAC.WL&date=latest&datum=MLLW&station={met_param}&time_zone=lst_ldt&units=english&interval=&format=json",
            expire_after=seconds_until_hour(),
            )

    app.logger.info(f"  cached: {water_temp.from_cache}")
    app.logger.info(f"  cached: {air_temp.from_cache}")

    met_data = [
            {
                'type': 'Water temp',
                'value': water_temp.json()['data'][0]['v'],
                'time': water_temp.json()['data'][0]['t'],
                },
            {
                'type': 'Air temp',
                'value': air_temp.json()['data'][0]['v'],
                'time': air_temp.json()['data'][0]['t'],
                },
            ]

    app.logger.info(time.perf_counter()-timer_start)
    app.logger.info("getting tides")
    tides = requests.get(
            f"{NOAA_TC_API}?product=predictions&application=NOS.COOPS.TAC.WL&begin_date={start_date}&end_date={end_date}&datum=MLLW&station={tide_station['id']}&time_zone=lst_ldt&units=english&interval=hilo&format=json",
            )

    app.logger.info(f"  cached: {tides.from_cache}")

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

    tide_max = dt['Feet'].max()  # used in chart for max moon %

    app.logger.info(time.perf_counter()-timer_start)
    app.logger.info("getting currents")
    currents = requests.get(
            f"{NOAA_TC_API}?product=currents_predictions&application=NOS.COOPS.TAC.WL&begin_date={start_date}&end_date={end_date}&datum=MLLW&station={current_station['id']}&time_zone=lst_ldt&units=english&interval=MAX_SLACK&format=json",
            )

    app.logger.info(f"  cached: {currents.from_cache}")

    dc = pd.DataFrame.from_dict(currents.json()['current_predictions']['cp'])
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

    fig = go.Figure()
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

    app.logger.info(time.perf_counter()-timer_start)
    app.logger.info("getting marine wind hourly")
    forecast_marine = requests.get(
            f"https://forecast.weather.gov/MapClick.php?lat={lat}&lon={lon}&FcstType=digitalDWML",
            expire_after=seconds_until_hour(),
            )


    app.logger.info(f"  cached: {forecast_marine.from_cache}")

    tree = ET.ElementTree(ET.fromstring(forecast_marine.text))
    root = tree.getroot()
    times = root.findall('.//start-valid-time')
    wind_speeds = root.findall('.//wind-speed[@type="sustained"]/value')
    wind_dir = root.findall('.//direction[@type="wind"]/value')
    waves = root.findall('.//waves[@type="significant"]/value')
    wind_annots = []

    for idx,t in enumerate(times[:24]):
        if (idx + 1) % 2 == 0:  # skip every other hour for display purposes
            pass
        else:
            cond = f"{deg_to_compass(wind_dir[idx].text)}<br>{wind_speeds[idx].text}kt<br>{waves[idx].text}'"
            wind_annots = wind_annots + [dict(x=t.text, yref="paper", y=1.05, text=cond, showarrow=False)]


    app.logger.info(time.perf_counter()-timer_start)
    app.logger.info("calc sun/moon data")

    sun = []
    for date in pd.date_range(start=start_date, end=end_date):
        sun.append(solar(date, lat, lon))

    moon_events = lunar(start_date_dt, end_date_dt, lat, lon)

    app.logger.info('before sun loop')
    app.logger.info(time.perf_counter()-timer_start)

    fig.update_layout(shapes=sun_vlines(sun))

    app.logger.info('after sun layout')

    app.logger.info(time.perf_counter()-timer_start)

    fig.update_layout(annotations=sun_annots(sun) + wind_annots)

    app.logger.info('after sun annotations')
    app.logger.info(time.perf_counter()-timer_start)

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

    fig.update_traces(textposition=improve_text_position(dc['Time']),
                      connectgaps=None)

    fig.update_yaxes(
     fixedrange = True,
    )

    if "iPhone" in request.headers.get('User-Agent'):
        max_range = local_now + timedelta(hours=6)
    else:
        max_range = local_now + timedelta(hours=12)

    fig.update_xaxes(
        range=[
            local_now - timedelta(hours=2),
            max_range,
            ],
        minallowed=start_date_dt,
        maxallowed=end_date_dt,
        automargin=False,
        ticks="outside",
        ticklabelmode="period",
        tickcolor= "black",
        ticklen=10,
        tickangle=0,
        minor=dict(
            ticklen=4,
            dtick=60*60*1000,
            tick0=datetime.today().strftime('%Y-%m-%d'),
            griddash='dot',
            gridcolor='white'
        )
    )

    fig.update_layout(
            showlegend=False,
            height=450,
            dragmode='pan',
            margin=dict(l=0, r=0, t=20, b=50),
            )

    fig.add_vline(x=local_now, line_width=1, line_dash="dash", line_color='green')

    fig_html = fig.to_html(
            full_html=False,
            include_plotlyjs='cdn',
            config={
                'displayModeBar': False,
                })


    app.logger.info(time.perf_counter()-timer_start)

    dct = dc[dc['Date'].dt.strftime('%Y-%m-%d') == today]
    dtt = dt[dt['Date'].dt.strftime('%Y-%m-%d') == today]
    dd = pd.merge(dct, dtt,  how="outer", on=['Date', 'Type'])

    moon_rises = [x for x in moon_data if x['phen'] == "rise"]
    moon_set = [x for x in moon_data if x['phen'] == "set"]

    text = [
            {'time': EASTERN.localize(datetime.now()),
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

    app.logger.info(time.perf_counter()-timer_start)

    text = sorted(text, key=lambda d: d['time'])

    resp = make_response(render_template('index.html', fig_html=fig_html,
                           current_station=current_station,
                           tide_station=tide_station,
                           met_station=met_station,
                           local_current_stations=local_current_stations,
                           local_tide_stations=local_tide_stations,
                           local_met_stations=local_met_stations,
                           forecast=forecast,
                           met_data=met_data,
                           tide_offset=tide_offset,
                           text_report=text,
                           ))

    if current_param != station_defaults['current']:
        resp.set_cookie('current', current_param, max_age=157784760)

    if tide_param != station_defaults['tide']:
        resp.set_cookie('tide', tide_param, max_age=157784760)

    resp.set_cookie('station_offsets',
                        json.dumps(station_offsets,
                                   separators=(',', ':')),
                    max_age=157784760)

    app.logger.info(time.perf_counter()-timer_start)

    return resp


@app.route('/manifest.json')
def manifest():
    return jsonify({
        "short_name": "Forecast",
        "name": "Tide, Current, and Wind Forecast",
        "id": "/",
        "start_url": "/",
        "background_color": "#3367D6",
        "display": "standalone",
        "scope": "/",
        "theme_color": "#3367D6",
        "description": "Weather forecast information",
    })


if __name__ == '__main__':
    app.run(debug=True)
