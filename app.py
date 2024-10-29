from flask import Flask, request, redirect, abort, render_template
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
import pytz
from turbo_flask import Turbo


app = Flask(__name__)
turbo = Turbo(app)

current_station_location = {
        "ACT2781": {  # Rocky Point
            "lat":  41.1438,
            "lon": -72.3570,
            },
        "ACT2521": {  # Jennings Point
            "lat": 41.0747,
            "lon": -72.3825,
            },
        "ACT2841": {  # Horton Point
            "lat": 41.1050,
            "lon": -72.4567,
            },
        "LIS1012": {  # Plum Gut
            "lat": 41.1592,
            "lon": -72.2075,
            },
        "ACT2541": {  # Little Peconic Bay
            "lat": 41.0263,
            "lon": -72.3847,
            },
        "ACT2721": {  # Orient Pt
            "lat": 41.1667,
            "lon": -72.2517,
            }
        }

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
    # fix indentation
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


session = CachedSession()

@app.route('/')
def tides():
    DAYS = 2

    local_now = datetime.now(pytz.timezone('US/Eastern'))
    tz = 'US/Eastern'
    local_now = datetime.now(pytz.timezone(tz))
    start_date = local_now.strftime('%Y%m%d')
    # start_date = (datetime.today() - timedelta(days = 1)).strftime('%Y%m%d')
    end_date = (local_now + timedelta(days = DAYS)).strftime('%Y%m%d')
    today = local_now.strftime('%Y-%m-%d')
    # offset = int(local_now.utcoffset().total_seconds()/60/60)

    current_station = request.args.get('current', 'ACT2781')
    tide_station = request.args.get('tide', '8512053')
    lat = current_station_location[current_station]['lat']
    lon = current_station_location[current_station]['lon']

    forecast_marine = session.get(
            f"https://forecast.weather.gov/MapClick.php?lat={lat}&lon={lon}&FcstType=digitalDWML",
            expire_after=seconds_until_hour(),
            )

    forecast_daily = session.get(
            f"https://marine.weather.gov/MapClick.php?lat={lat}&lon={lon}&FcstType=json",
            expire_after=seconds_until_hour(),
            )

    app.logger.debug(f"Forcast wx cache hit: {forecast_daily.from_cache}")

    forecast_periods = forecast_daily.json()['time']['startPeriodName']
    forecast_text = forecast_daily.json()['data']['text']

    forecast = []

    for idx, text in enumerate(forecast_text[:DAYS*2]):
        forecast.append({
                "when": forecast_periods[idx],
                "text": text,
                })

    app.logger.debug(f"Forcast cache hit: {forecast_marine.from_cache}")

    MTK = "8510560"

    water_temperature = session.get(
            f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=water_temperature&application=NOS.COOPS.TAC.WL&begin_date={start_date}&end_date={end_date}&datum=MLLW&station={MTK}&time_zone=lst_ldt&units=english&interval=6&format=json",
            expire_after=seconds_until_hour(),
            )

    water_temp = water_temperature.json()['data'][-1]

    app.logger.debug(f"Forcast wx cache hit: {forecast_daily.from_cache}")
    tides = session.get(
            f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=predictions&application=NOS.COOPS.TAC.WL&begin_date={start_date}&end_date={end_date}&datum=MLLW&station={tide_station}&time_zone=lst_ldt&units=english&interval=hilo&format=json",
            )

    app.logger.debug(f"Tides cache hit: {tides.from_cache}")

    dt = pd.DataFrame.from_dict(tides.json()['predictions'])
    dt = dt.rename(columns={
        't': 'Date',
        'v': 'Feet',
        'type': "Type"
        })

    dt['Date'] = pd.to_datetime(dt['Date'])
    dt['Feet'] = dt['Feet'].astype("float")
    dt['Time'] = dt['Date'].dt.strftime("%H:%M")

    currents = session.get(
            f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=currents_predictions&application=NOS.COOPS.TAC.WL&begin_date={start_date}&end_date={end_date}&datum=MLLW&station={current_station}&time_zone=lst_ldt&units=english&interval=MAX_SLACK&format=json",
            )

    app.logger.debug(f"Currents cache hit: {currents.from_cache}")

    dc = pd.DataFrame.from_dict(currents.json()['current_predictions']['cp'])
    dc = dc.rename(columns={
        'Time': 'Date',
        'Velocity_Major': 'Knots'
        })

    dc['Date'] = pd.to_datetime(dc['Date'])
    dc['Time'] = dc['Date'].dt.strftime("%H:%M")
    # dc['Knots'] = dc['Knots'].abs()

    def until_next_event(row):
        if row['Knots'] == 0 :
            current = f"Slack at {row['Time']}"
            event = "Building"
        else:
            current = f"{row['Knots']} kt at {row['Time']}"
            event = "Waning"
        try:
            until = f"{event} until {dc.iloc[row.name + 1]['Time']}"
            return f"{current}<br>{until}"
        except IndexError as e:
            return None

    dc['Event'] = dc.apply(until_next_event, axis=1)

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


    for d in pd.date_range(start=start_date, end=end_date):
        date = d.strftime('%Y-%m-%d')
        astronomical = session.get(
                # f"https://aa.usno.navy.mil/api/rstt/oneday?date={date}&coords={lat},{lon}&tz={offset}",
                f"https://api.sunrise-sunset.org/json?date={date}&lat={lat}&lng={lon}&tzid={tz}",
                )
        app.logger.debug(f"Sun cache hit: {astronomical.from_cache}")
        sun = astronomical.json()['results']

        # fig = add_sun_annot(fig, pd.to_datetime(f"{date} {sun['astronomical_twilight_begin']}"),
        #                     color="grey", shift=-20)
        fig = add_sun_annot(fig, pd.to_datetime(f"{date} {sun['nautical_twilight_begin']}"),
                            color="blue", shift=-20)
        # fig = add_sun_annot(fig, pd.to_datetime(f"{date} {sun['civil_twilight_begin']}"),
        #                     color="grey", shift=-20)
        fig = add_sun_annot(fig, pd.to_datetime(f"{date} {sun['sunrise']}"))
        fig = add_sun_annot(fig, pd.to_datetime(f"{date} {sun['sunset']}"), shift=-20)
        fig = add_sun_annot(fig,
                            pd.to_datetime(f"{date} {sun['nautical_twilight_end']}"),
                            color="blue", shift=20)

    tree = ET.ElementTree(ET.fromstring(forecast_marine.text))

    root = tree.getroot()

    times = root.findall('.//start-valid-time')
    wind_speeds = root.findall('.//wind-speed[@type="sustained"]/value')
    wind_gusts = root.findall('.//wind-speed[@type="gust"]/value')
    wind_dir = root.findall('.//direction[@type="wind"]/value')
    waves = root.findall('.//waves[@type="significant"]/value')

    for idx,t in enumerate(times[:24]):
        if (idx + 1) % 2 == 0:  # skip every other hour
            pass
        else:
            time = t.text
            # gusts = f"<br>{wind_gusts[idx].text}kt gusts" if wind_gusts[idx].text else ""
            cond = f"{deg_to_compass(wind_dir[idx].text)}<br>{wind_speeds[idx].text}kt"
            fig.add_annotation(x=time, yref="paper", y=1.05, text=cond, showarrow=False)

    fig.update_traces(textposition=improve_text_position(dc['Time']))

    fig.update_yaxes(
     fixedrange = True,
    )

    fig.update_xaxes(
        range=[
            local_now - timedelta(hours=2),
            local_now + timedelta(hours=6),
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
            height=500,
            margin=dict(l=10, r=10, t=40, b=40),
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(label="4h",
                             method="relayout",
                             args=["xaxis.range", [local_now - timedelta(hours=2), local_now + timedelta(hours=2)]]
                             ),
                        dict(label="12h",
                             method="relayout",
                             args=["xaxis.range", [local_now - timedelta(hours=2), local_now + timedelta(hours=10)]]
                             ),
                        dict(label="24h",
                             method="relayout",
                             args=["xaxis.range", [local_now - timedelta(hours=2), local_now + timedelta(hours=22)]]
                             ),
                        ],
                    )
                ],
            )

    fig.add_vline(x=local_now, line_width=1, line_dash="dash", line_color='green')
    fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    return render_template('index.html', fig_html=fig_html,
                           current_station=current_station,
                           tide_station=tide_station,
                           forecast=forecast,
                           water_temp=water_temp,
                           )

if __name__ == '__main__':
    app.run(debug=True)
