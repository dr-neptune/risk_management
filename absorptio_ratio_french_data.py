from cytoolz import curry
from dateutil import parser
from frds.measures import absorption_ratio
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.io as pio
import re
pio.templates.default = "plotly_dark"

from absorption_ratio import custom_rolling
from get_recessions import get_recessions


def get_real_values_df(percentage_df: pd.DataFrame, starting_capital: float = 1):
    # add $1 to each industry to start off with
    percentage_df.iloc[0] = starting_capital * percentage_df.iloc[0]

    # percolate real values throughout time
    for i in range(1, len(percentage_df)):
        percentage_df.iloc[i] = percentage_df.iloc[i-1] * percentage_df.iloc[i]

    return percentage_df


def plot_data(df, absorption_ratio, recessions_data, file_name = None):
    fig = sp.make_subplots(rows=2, cols=1, vertical_spacing=0.01, shared_xaxes=True)

    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col), row=1, col=1)

    # Calculate the 25th and 75th percentile lines for each row
    percentile_25 = df.apply(lambda x: np.percentile(x, 25), axis=1)
    percentile_75 = df.apply(lambda x: np.percentile(x, 75), axis=1)

    # Add the 25th and 75th percentile lines to the plot
    fig.add_trace(go.Scatter(x=df.index, y=percentile_25, mode='lines', name='25th percentile', line=dict(color='rgba(0,100,80,0.5)')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=percentile_75, mode='lines', name='75th percentile', line=dict(color='rgba(0,100,80,0.5)'), fill='tonexty'), row=1, col=1)

    fig.add_trace(go.Scatter(x=absorption_ratio.index, y=absorption_ratio, fill='tozeroy', name='Absorption Ratio', line_color='lightseagreen'), row=2, col=1)

    recessions = recessions_data[['Name', 'Begin_Date', 'End_Date']].to_dict('records')

    for recession in recessions:
        for i in range(1, 3):  # add to both subplots
            fig.add_shape(type='rect', xref='x', yref='paper',
                          x0=recession['Begin_Date'], x1=recession['End_Date'],
                          y0=0, y1=1, fillcolor='grey', opacity=0.5, layer='above', line_width=0,
                          row=i, col=1)

        # Calculate the middle of the recession period
        middle_recession = (pd.to_datetime(recession['Begin_Date']) + (pd.to_datetime(recession['End_Date']) - pd.to_datetime(recession['Begin_Date'])) / 2)
        # Add an annotation for the recession
        fig.add_annotation(
            x=middle_recession,
            y=1.05,  # to position it a bit above the top
            yref="paper",
            text=recession['Name'].replace('recession', ''),
            font=dict(size=10, color="white"),
            bgcolor="black",
            opacity=0.8,
            xshift=10,
            yshift=-20,
            showarrow=True,
            row=1, col=1,
        )

    fig.update_yaxes(title_text='Value in $', row=1, col=1)
    fig.update_yaxes(title_text='Absorption Ratio', row=2, col=1, range=[0.85, 1])
    fig.update_xaxes(rangeslider_visible=True, row=2, col=1)

    fig.update_layout(
        title='Industry Value and Absorption Ratio over Time',
        autosize=False,
        width=800,
        height=1000,
        template='plotly_dark')

    if file_name is not None:
        fig.write_html(f'figures/{file_name}.html')
    else:
        fig.show()


if __name__ == '__main__':
    # get French data
    # data from https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
    industry_data = pd.read_csv('data/french_industry_10.csv')
    industry_data = industry_data.rename(columns={industry_data.columns[0]: 'Date'})

    # adjust the date format
    industry_data['Date'] = pd.to_datetime(industry_data['Date'].apply(lambda x: f"{x}01"), format='%Y%m%d')

    # get absorption ratios
    num_eigenvectors = 10
    window_size = 24

    ar_frds = (custom_rolling(industry_data.set_index('Date') / 100,
                              curry(absorption_ratio.estimate, fraction_eigenvectors=0.2), window_size)
               .astype(float)
               .dropna())

    # get real values
    percentage_df = (industry_data.set_index('Date') / 100) + 1
    real_values = get_real_values_df(percentage_df)

    # get recession periods
    recessions = get_recessions()

    # plot 2000s data
    plot_data(real_values[real_values.index >= '2000-01-01'],
              ar_frds[ar_frds.index >='2000-01-01'][0],
              recessions[recessions['Begin_Date'] >= '2000-01-01'],
              'absorption_ratio_french_data_2000s')

    # plot data since 1926
    plot_data(real_values, ar_frds[0], recessions, 'absorption_ratio_french_data_full_history')
