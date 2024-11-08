from flask import Flask, request, redirect, abort, render_template, make_response, url_for
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
from concurrent.futures import ProcessPoolExecutor
import time

app = Flask(__name__)
turbo = Turbo(app)

# https://requests-cache.readthedocs.io/en/stable/user_guide/general.html#patching
requests_cache.install_cache()

NOAA_TC_API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

if not app.debug:
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

def add_sun_annot(fig, when, color="orange", shift=20):
  fig.add_vline(x=when, line_width=2, line_dash="dash", line_color=color)
  fig.add_annotation(x=when, yref="paper", y=0, text=when.strftime("%H:%M"),
                     showarrow=False,
                     xshift=shift)
  return fig

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
        until = f"{event} for {until_string} until {next_event.strftime('%H:%m')}"
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
    d = pd.to_datetime(row['Date'])
    if row['Type'] == "H":
        return d + timedelta(minutes=offset)
    else:
        return d - timedelta(minutes=offset)

def deg_to_phase(deg):
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


def solar(date, lat, lon):
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

    return {
        'dawn': times[1].astimezone(et),
        'rise': times[3].astimezone(et),
        'set': times[4].astimezone(et),
        'dusk': times[7].astimezone(et),
        }

def lunar(date, lat, lon):
    ts = load.timescale()
    eph = load('de421.bsp')
    moon = eph['Moon']
    topos = wgs84.latlon(lat, lon)
    observer = eph['Earth'] + topos
    et = pytz.timezone('US/Eastern')

    day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(hours=24)
    t0 = ts.from_datetime(et.localize(day_start))
    t1 = ts.from_datetime(et.localize(day_end))

    mr,y = almanac.find_risings(observer, moon, t0, t1)
    ms,y = almanac.find_settings(observer, moon, t0, t1)
    mtr = almanac.find_transits(observer, moon, t0, t1)
    mp = almanac.moon_phase(eph, mtr[0])

    moon_transit = mtr.astimezone(et)[0]

    try:
        moon_set = ms.astimezone(et)[0]
    except IndexError as e:
        moon_set = None

    d = {
        'times': {
            'rise': mr.astimezone(et)[0],
            'transit': moon_transit,
            'set': moon_set,
            'none': moon_transit + timedelta(hours=12),
            },
        'deg': mp.degrees,
        'illum': mp.degrees / 180 if mp.degrees / 180 < 1 else (mp.degrees / 180) - 1,
        'phase': deg_to_phase(mp.degrees),
        }

    return d


@app.route('/', methods=["GET", "POST"])
def tides():

    app.logger.info("load initiated")

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

    DAYS = 3

    tz = 'US/Eastern'
    EASTERN = pytz.timezone(tz)
    local_now = datetime.now(pytz.timezone(tz))
    tz_offset = int(local_now.utcoffset().total_seconds()/60/60)
    # start_date = local_now.strftime('%Y%m%d')
    start_date_dt = (local_now - timedelta(days = 1))
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
                    station_offsets.get(tide_param, 0)
                    )

    tide_offset = int(offset_param) if offset_param is not None else 0

    if tide_param != 0:
        station_offsets[tide_param] = tide_offset

    current_station = [x for x in local_current_stations if x['id'] == current_param][0]
    tide_station = [x for x in local_tide_stations if x['id'] == tide_param][0]
    met_station = [x for x in local_met_stations if x['id'] == met_param][0]

    app.logger.info("getting station metadata")
    station_metadata = requests.get(
            f"https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{current_station['id']}.json",
            )

    app.logger.info(f"  cached: {station_metadata.from_cache}")

    station_metadata = station_metadata.json()

    lat = station_metadata['stations'][0]['lat']
    lon = station_metadata['stations'][0]['lng']

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


    app.logger.info("getting met data")
    water_temp = requests.get(
            f"{NOAA_TC_API}?product=water_temperature&application=NOS.COOPS.TAC.WL&begin_date={start_date}&end_date={end_date}&datum=MLLW&station={met_param}&time_zone=lst_ldt&units=english&interval=6&format=json",
            expire_after=seconds_until_hour(),
            )

    app.logger.info(f"  cached: {water_temp.from_cache}")

    water_temp = water_temp.json()['data'][-1]

    app.logger.info("getting tides")
    tides = requests.get(
            f"{NOAA_TC_API}?product=predictions&application=NOS.COOPS.TAC.WL&begin_date={start_date}&end_date={end_date}&datum=MLLW&station={tide_station['id']}&time_zone=lst_ldt&units=english&interval=hilo&format=json",
            )

    dt = pd.DataFrame.from_dict(tides.json()['predictions'])
    dt = dt.rename(columns={
        't': 'Date',
        'v': 'Feet',
        'type': "Type"
        })


    if tide_offset != 0:
        dt['Date'] = dt.apply(calc_tide_offset, axis=1, offset=tide_offset)
    else:
        dt['Date'] = pd.to_datetime(dt['Date'])

    dt['Feet'] = dt['Feet'].astype("float")
    dt['Time'] = dt['Date'].dt.strftime("%H:%M")

    tide_max = dt['Feet'].max()  # used in chart for max moon %

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

    for idx,t in enumerate(times):
        if (idx + 1) % 2 == 0:  # skip every other hour
            pass
        else:
            cond = f"{deg_to_compass(wind_dir[idx].text)}<br>{wind_speeds[idx].text}kt"
            fig.add_annotation(x=t.text, yref="paper", y=1.05, text=cond, showarrow=False)

    app.logger.info("calc sun/moon data")

    sun = []
    moon = []

    timer_start = time.perf_counter()


    for date in pd.date_range(start=start_date, end=end_date):
        sun.append(solar(date, lat, lon))
        moon.append(lunar(date, lat, lon))

    app.logger.info(time.perf_counter()-timer_start)

    for s in sun:
        fig = add_sun_annot(fig, s['dawn'], color="blue", shift=-20)
        fig = add_sun_annot(fig, s['rise'])
        fig = add_sun_annot(fig, s['set'], shift=-20)
        fig = add_sun_annot(fig, s['dusk'], color="blue", shift=20)


    moon_data = []
    for m in moon:
        for e,t in m['times'].items():
            if t is not None:
                if e == 'transit':
                    value = tide_max * m['illum']
                    text = (f"Moon upper transit at {t.strftime('%H:%m')}<br>"
                            + f"{m['phase']}<br>"
                            + f"{int(m['illum'] * 100)}% Illumination<br>")
                elif e == 'none':
                    value = None
                else:
                    value = 0
                    text = f"Moon {e} at {t.strftime('%H:%m')}"

                moon_data.append({
                    'time': t,
                    'phen': e,
                    'fracillum': m['illum'],
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

    fig.update_xaxes(
        range=[
            local_now - timedelta(hours=2),
            local_now + timedelta(hours=12),
            ],
        minallowed=start_date_dt,
        maxallowed=end_date_dt,
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
            height=500,
            margin=dict(l=10, r=10, t=40, b=40),
            )

    fig.add_vline(x=local_now, line_width=1, line_dash="dash", line_color='green')
    fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')


    resp = make_response(render_template('index.html', fig_html=fig_html,
                           current_station=current_station,
                           tide_station=tide_station,
                           met_station=met_station,
                           local_current_stations=local_current_stations,
                           local_tide_stations=local_tide_stations,
                           local_met_stations=local_met_stations,
                           forecast=forecast,
                           water_temp=water_temp,
                           tide_offset=tide_offset,
                           ))

    if current_param != station_defaults['current']:
        resp.set_cookie('current', current_param, max_age=157784760)

    if tide_param != station_defaults['tide']:
        resp.set_cookie('tide', tide_param, max_age=157784760)

    resp.set_cookie('station_offsets',
                        json.dumps(station_offsets,
                                   separators=(',', ':')),
                    max_age=157784760)

    return resp



if __name__ == '__main__':
    app.run(debug=True)
