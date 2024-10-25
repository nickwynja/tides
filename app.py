from flask import Flask, request, redirect, abort
from urllib.parse import urlparse
import logging
import re
import requests
import json
from datetime import datetime, timedelta
import pandas as pd
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
from requests_cache import CachedSession
import pprint
import math


app = Flask(__name__)

if not app.debug:
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

def add_sun_annot(fig, when, color="orange", y=-0.4):
  fig.add_vline(x=when, line_width=2, line_dash="dash", line_color=color)
  fig.add_annotation(x=when, y=y, text=when.strftime("%H:%M"), showarrow=False)
  return fig

def improve_text_position(x):
    """ it is more efficient if the x values are sorted """
    # fix indentation
    positions = ['top center', 'bottom center']  # you can add more: left center ...
    return [positions[i % len(positions)] for i in range(len(x))]

def seconds_until_hour():
    delta = timedelta(hours=1)
    now = datetime.now()
    next_hour = (now + delta).replace(microsecond=0, second=0, minute=0)
    wait_seconds = (next_hour - now).seconds
    return wait_seconds

def km_to_kn(speed):
    return math.floor(int(speed) * 0.53996)

def deg_to_compass(num):
    num=int(num)
    val=int((num/22.5)+.5)
    arr=["N","NNE","NE","ENE","E","ESE", "SE", "SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
    return arr[(val % 16)]


session = CachedSession()

@app.route('/')
def tides():
    DAYS = 7

    start_date = (datetime.today() - timedelta(days = 1)).strftime('%Y%m%d')
    end_date = (datetime.today() + timedelta(days = DAYS)).strftime('%Y%m%d')

    tide_station = "8512053" # Hash
    current_station = "ACT2781" # Rocky pt

    cache_expires = seconds_until_hour()

    forecast_hourly = session.get(
            f"https://api.weather.gov/gridpoints/OKX/87,63",
            expire_after=cache_expires,
            )

    forecast_daily = session.get(
            f"https://dataservice.accuweather.com/forecasts/v1/daily/5day/2274451?apikey=qhBeSntg5AV4KEv6AbtWQOOQKVnGAkY9&details=true",
            expire_after=cache_expires,
            )

    forecast_marine = session.get(
            f"https://forecast.weather.gov/MapClick.php?lat=41.1328&lon=-72.6664&FcstType=digitalDWML",
            expire_after=cache_expires,
            )

    tides_csv = session.get(
            f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=predictions&application=NOS.COOPS.TAC.WL&begin_date={start_date}&end_date={end_date}&datum=MLLW&station={tide_station}&time_zone=lst_ldt&units=english&interval=hilo&format=csv",
            expire_after=cache_expires,
            )
    currents_csv = session.get(
            f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=currents_predictions&application=NOS.COOPS.TAC.WL&begin_date={start_date}&end_date={end_date}&datum=MLLW&station={current_station}&time_zone=lst_ldt&units=english&interval=6&format=csv",
            expire_after=cache_expires,
            )

    dt = pd.read_table(StringIO(tides_csv.text), sep=",", names=["Date", "Feet", "Type"], skiprows=1)
    dt['Date'] = pd.to_datetime(dt['Date'])
    dt['Time'] = dt['Date'].dt.strftime("%H:%M")

    dc = pd.read_table(StringIO(currents_csv.text), sep=",", names=["Date", "Depth", "Knots", "meanFloodDir", "meanEbbDir", "Bin"], skiprows=1)
    dc['Date'] = pd.to_datetime(dc['Date'])
    dc['Time'] = dc['Date'].dt.strftime("%H:%M")
    dc['Knots'] = dc['Knots'].abs()

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
          text=dc['Time'],
          hovertemplate = '%{y} kt',
          mode='lines+markers+text',
          textposition='top center', # Adjust text position
          line_shape="spline",
          name='Current'
        )
    )


    for d in forecast_daily.json()['DailyForecasts']:
        fig = add_sun_annot(fig, pd.to_datetime(d['Sun']['Rise']))
        fig = add_sun_annot(fig, pd.to_datetime(d['Sun']['Set']))


    import xml.etree.ElementTree as ET

    tree = ET.ElementTree(ET.fromstring(forecast_marine.text))

    root = tree.getroot()

    times = root.findall('.//start-valid-time')
    wind_speeds = root.findall('.//wind-speed[@type="sustained"]/value')
    wind_dir = root.findall('.//direction[@type="wind"]/value')

    winds = []

    for idx,t in enumerate(times):
        winds.append({
            'time': t.text,
            'speed': wind_speeds[idx].text,
            'direction': wind_dir[idx].text
        })
    # print(len(times))
    # print(len(wind_speeds))
    # print(len(wind_dir))
    # for t in times:
    #     print(t.text)

    for w in winds[:24]:
        time = w['time']
        wind = f"{w['speed']}kt<br>{deg_to_compass(w['direction'])}"
        fig.add_annotation(x=time, y=5, text=wind, showarrow=False)


    # for d in forecast_hourly.json()['properties']['windSpeed']['values'][:-12]:
    #     valid_time = d['validTime'].split('/')
    #     time = pd.to_datetime(valid_time[0])
    #     valid_hours = int(valid_time[1].removeprefix("PT").removesuffix("H"))
    #     wind = f"{km_to_kn(d['value'])} kt"

    #     if valid_hours > 1:
    #         fig.add_annotation(x=time, y=5, text=wind, showarrow=False)
    #         # while valid_hours > 1:
    #         #     time = time + timedelta(hours=1)
    #         #     fig.add_annotation(x=time, y=5, text=wind, showarrow=False)
    #         #     valid_hours = valid_hours - 1
    #     else:
    #         fig.add_annotation(x=time, y=5, text=wind, showarrow=False)

    # for d in forecast_hourly.json()['properties']['windDirection']['values'][:-12]:
    #     valid_time = d['validTime'].split('/')
    #     time = pd.to_datetime(valid_time[0])
    #     valid_hours = int(valid_time[1].removeprefix("PT").removesuffix("H"))
    #     wind = deg_to_compass(d['value'])

    #     if valid_hours > 1:
    #         fig.add_annotation(x=time, y=4.9, text=wind, showarrow=False)
    #         # while valid_hours > 1:
    #         #     time = time + timedelta(hours=1)
    #         #     fig.add_annotation(x=time, y=4.9, text=wind, showarrow=False)
    #         #     valid_hours = valid_hours - 1
    #     else:
    #         fig.add_annotation(x=time, y=4.9, text=wind, showarrow=False)

    fig.update_traces(textposition=improve_text_position(dc['Time']))

    fig.update_yaxes(
     fixedrange = True,
    )

    fig.update_xaxes(
        range=[
            f"{datetime.today().strftime('%Y-%m-%d')} 05:00:00",
            f"{datetime.today().strftime('%Y-%m-%d')} 22:00:00"
            ],
        ticks="outside",
        ticklabelmode="period",
        tickcolor= "black",
        ticklen=10,
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
            height=800,
            )

    return fig.to_html(full_html=True, include_plotlyjs='cdn')

if __name__ == '__main__':
    app.run(debug=True)
