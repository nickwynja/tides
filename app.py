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


@app.route('/')
def tides():
    DAYS = 7

    start_date = (datetime.today() - timedelta(days = 1)).strftime('%Y%m%d')
    end_date = (datetime.today() + timedelta(days = DAYS)).strftime('%Y%m%d')

    tide_station = "8512053" # Hash
    current_station = "ACT2781" # Rocky pt

    tides_csv = requests.get(f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=predictions&application=NOS.COOPS.TAC.WL&begin_date={start_date}&end_date={end_date}&datum=MLLW&station={tide_station}&time_zone=lst_ldt&units=english&interval=hilo&format=csv")
    currents_csv = requests.get(f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=currents_predictions&application=NOS.COOPS.TAC.WL&begin_date={start_date}&end_date={end_date}&datum=MLLW&station={current_station}&time_zone=lst_ldt&units=english&interval=6&format=csv")

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


    solar = requests.get("https://gml.noaa.gov/grad/solcalc/table.php?lat=41.130783&lon=-72.362957&year=2024")
    dss = pd.read_html(StringIO(solar.text))

    dr = dss[0]
    ds = dss[1]

    datelist = pd.date_range(datetime.today() - timedelta(days = 1), periods=DAYS).tolist()

    sro = dr.astype(object)
    sso = ds.astype(object)

    for single_date in datelist:
      month = single_date.strftime("%b")
      day = single_date.strftime("%d")

      sun_rise_time = sro[month][int(day)-1]
      sun_rise = pd.to_datetime(f"{single_date.strftime('%Y-%m-%d')} {sun_rise_time}")
      fig = add_sun_annot(fig, sun_rise)

      sun_set_time = sso[month][int(day)-1]
      sun_set = pd.to_datetime(f"{single_date.strftime('%Y-%m-%d')} {sun_set_time}")
      fig = add_sun_annot(fig, sun_set)


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
