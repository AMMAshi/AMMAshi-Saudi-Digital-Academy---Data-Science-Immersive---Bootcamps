# ==========================================================================================
# Arwa Ashi - HW4 - Week9 - Nov 7th, 2020
# Saudi Digital Academy
# ==========================================================================================
# Stock Ticker with Dash.
# ------------------------------------------------------------------------------------------
# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

# packages
# ------------------------------------------------------------------------------------------
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

# ------------------------------------------------------------------------------------------
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

# Data Frames Live Stock price data from finance.yahoo.com
# ------------------------------------------------------------------------------------------
from scrapingstockcode import df_05 as df

# Fix date and set it as index
df['dates'] = pd.to_datetime(df['dates']).dt.date
#print(df)

# Figures
# ------------------------------------------------------------------------------------------
trace_1 = px.line(df, x="dates", y="prices",color = 'companies')

trace_1.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)

# Layout
# ------------------------------------------------------------------------------------------
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Live Stock Prices in SAR',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.Div(children='Saudi Digital Academy - Dash: A web application framework for Python.', style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    
    html.Div(id='graphs'),

    dcc.Graph(
        id='stock-graph',
        figure=trace_1
    ),
    
    dcc.Dropdown(
        id='stock-ticker-input',
        options=[{'label': s[0], 'value': str(s[1])}
                 for s in zip(df.companies.unique(), df.companies.unique())],
        value=['Saudi Arabian Oil Company (2222.SR)', 'Alinma Bank (1150.SR)'],
        multi=True
    ),
])

# ------------------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
