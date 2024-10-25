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
import xml.etree.ElementTree as ET


app = Flask(__name__)

if not app.debug:
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

def add_sun_annot(fig, when):
  fig.add_vline(x=when, line_width=2, line_dash="dash", line_color="orange")
  fig.add_annotation(x=when, yref="paper", y=0, text=when.strftime("%H:%M"), showarrow=False)
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

    if request.args.get('tides') == "bay":
        tide_station = "8512114" # Southold
        current_station = "ACT2521" # Jennings Pt
        lat = 41.0338
        lon = -72.4344
    else:
        tide_station = "8512053" # Hash
        current_station = "ACT2781" # Rocky pt
        lat = 41.1396742515397
        lon = -72.35398912476371

    cache_expires = seconds_until_hour()

    # forecast_hourly = session.get(
    #         f"https://api.weather.gov/gridpoints/OKX/87,63",
    #         expire_after=cache_expires,
    #         )

    forecast_daily = session.get(
            f"https://dataservice.accuweather.com/forecasts/v1/daily/5day/2274451?apikey=qhBeSntg5AV4KEv6AbtWQOOQKVnGAkY9&details=true",
            expire_after=cache_expires,
            )

    forecast_marine = session.get(
            f"https://forecast.weather.gov/MapClick.php?lat={lat}&lon={lon}&FcstType=digitalDWML",
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



    tree = ET.ElementTree(ET.fromstring(forecast_marine.text))

    root = tree.getroot()

    times = root.findall('.//start-valid-time')
    wind_speeds = root.findall('.//wind-speed[@type="sustained"]/value')
    wind_gusts = root.findall('.//wind-speed[@type="gust"]/value')
    wind_dir = root.findall('.//direction[@type="wind"]/value')
    waves = root.findall('.//waves[@type="significant"]/value')

    for idx,t in enumerate(times[:36]):
        time = t.text
        # gusts = f"<br>{wind_gusts[idx].text}kt gusts" if wind_gusts[idx].text else ""
        cond = f"{wind_speeds[idx].text}kt<br>{deg_to_compass(wind_dir[idx].text)}<br>{waves[idx].text}ft"
        fig.add_annotation(x=time, yref="paper", y=1, text=cond, showarrow=False)

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
            height=600,
            margin=dict(l=10, r=10, t=40, b=40),
            )

    fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
</head>

<body>
<a href="/?tides=sound">Sound</a>
<a href="/?tides=bay">Bay</a>
{ fig_html }
</body>
</html>
"""

    return html

if __name__ == '__main__':
    app.run(debug=True)
