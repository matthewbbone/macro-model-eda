from dash import Dash, html, dcc, callback, Output, Input, dash_table
from dash.dash_table.Format import Format, Scheme
import plotly.express as px
import pandas as pd
import numpy as np
import json
import colorlover

def country_code_switch(codes):

    c_map = pd.read_csv("raw_data/codes.csv")

    c_map.loc[len(c_map)] = [
        "Greece",
        "EL",
        "GRC",
        "300",
        "ISO 3166-2:GR",
    ]
    c_map.loc[len(c_map)] = [
        "United Kingdom",
        "UK",
        "GBR",
        "826",
        "ISO 3166-2:GB",
    ]

    for c in codes:
        try:
            c_map.loc[c_map["Alpha-3 code"] == c, "English short name lower case"].values[0]
        except:
            print(c)

    return [
        c_map.loc[c_map["Alpha-3 code"] == c, "English short name lower case"].values[0]
        for c in codes
    ]

def summarize_params(data):

    temp = data.copy()

    countries = temp.pop('countries')
    years = temp.pop('years')

    summ = np.zeros((len(countries), len(years)))
    total = len(temp)

    parameter_country = np.zeros((len(temp), len(countries)))
    parameter_years = np.zeros((len(temp), len(years)))
    
    ctr = 0
    for key, val in temp.items():

        # country_year
        v = np.array(val)
        v[~np.isnan(v)] = 1
        v[np.isnan(v)] = 0
        summ = summ + v

        # parameter_country
        parameter_country[ctr] = v.sum(axis=1)

        # parameter_year
        parameter_years[ctr] = v.sum(axis=0)

        ctr = ctr + 1

    data['country_year'] = summ / total
    data['parameter_country'] = parameter_country / len(years)
    data['parameter_years'] = parameter_years / len(countries)

    return data, list(temp.keys()) + ['country_year', 'parameter_country', 'parameter_years']

with open('test_data.json', 'r') as f:

    data, options = summarize_params(json.loads(f.read()))

app = Dash(__name__)

app.layout = html.Div([
    html.H1(children='Macro Data Quality', style={'textAlign':'center'}),
    dcc.Dropdown(options, 'country_year', id='dropdown-selection'),
    dash_table.DataTable(id="table", export_format="csv")
])

def country_year(value, d):

    temp = d.copy()

    countries =  temp.pop('countries')
    years = temp.pop('years')

    df = pd.DataFrame(temp[value])
    df.columns = years
    df['country'] = country_code_switch(countries)

    columns = [{'id': 'country', 'name': 'country', 'type': 'text'}]
    formatting = []
    color_bins = 10
    backgroundColor = colorlover.scales[str(color_bins)]['div']['RdYlGn']

    for c in years:
        columns.append({
            'id': str(c),
            'name': str(c),
            'type': 'numeric',
            'format': Format(precision=2, scheme=Scheme.decimal)
        })

        if value == 'country_year':
            for bin in range(color_bins):
                formatting.append({
                    'if': {
                        'filter_query': '{{{c}}} >= {mini} && {{{c}}} < {maxi}'.format(c=c, mini=bin / color_bins, maxi=(bin+1)/color_bins),
                        'column_id': f'{c}'
                    },
                    'backgroundColor': backgroundColor[bin]
                })
        else:
            formatting.append({
                'if': {
                    'column_id': f'{c}',
                    'filter_query': f"{{{c}}} is nil"
                },
                'backgroundColor': 'red'
            })
    
    return df.to_dict('records'), columns, formatting

def parameter_country(value, d):

    temp = d.copy()
    countries =  temp.pop('countries')
    years = temp.pop('years')

    df = pd.DataFrame(temp[value])

    if value == 'parameter_country':
        cols = countries
    else:
        cols = years

    df.columns = cols
    df['parameters'] = options[:-3]

    columns = [{'id': 'parameters', 'name': 'parameters', 'type': 'text'}]
    formatting = []
    backgroundColor = colorlover.scales[str(8)]['div']['RdYlGn']

    for c in cols:
        columns.append({
            'id': str(c),
            'name': str(c),
            'type': 'numeric',
            'format': Format(precision=2, scheme=Scheme.decimal)
        })

        for bin in range(8):
            formatting.append({
                'if': {
                    'filter_query': '{{{c}}} >= {mini} && {{{c}}} <= {maxi}'.format(c=c, mini=bin / 8, maxi=(bin+1)/8),
                    'column_id': f'{c}'
                },
                'backgroundColor': backgroundColor[bin]
            })

    return df.to_dict('records'), columns, formatting

@callback(
    Output('table', 'data'),
    Output('table', 'columns'),
    Output('table', 'style_data_conditional'),
    Input('dropdown-selection', 'value')
)
def update_graph(value):

    if value in ['parameter_years','parameter_country']:
        return parameter_country(value,data)
    else:
        return country_year(value, data)

if __name__ == '__main__':
    app.run_server(debug=True)