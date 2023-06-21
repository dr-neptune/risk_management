
from cytoolz import curry, compose_left, take, identity
from datetime import datetime
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_dark"

from get_sp500 import get_sp500_table

def calc_absorption_ratio(df, num_eigenvectors=5):
    def calc_cov(df):
        calc_cov.mat = np.cov(df)
        return calc_cov.mat
    return compose_left(calc_cov,
                        np.linalg.eigvals,
                        sorted,
                        reversed,
                        curry(take, num_eigenvectors),
                        sum,
                        lambda v: v / np.trace(calc_cov.mat))(df)


def custom_rolling(df, function, window):
    num_rows = df.shape[0]
    results = []

    for i in range(num_rows):
        if i >= window - 1:
            lower, upper = i - window + 1, i + 1
            print(f"running lower: {lower} upper: {upper}")
            window_data = df.iloc[lower:upper, :]
            result = function(window_data)
            results.append(result)
        else:
            results.append(np.nan)

    return pd.DataFrame(results, index=df.index)


def plot_series_subplots(returns, absorption_ratio, y_col1, y_col2, title):
    # Create a subplot with 2 rows and 1 column
    fig = make_subplots(rows=2, cols=1)

    # Add the first series to the first subplot
    fig.add_trace(
        go.Scatter(x=returns['Date'], y=returns[y_col1], name="Absorption Ratio"),
        row=1, col=1
    )

    # Add the second series to the second subplot
    fig.add_trace(
        go.Scatter(x=absorption_ratio['Date'], y=absorption_ratio[y_col2], name="S&P 500 Returns"),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(height=600, width=800, title_text=title)

    fig.update_xaxes(range=[datetime(2002, 1, 1), datetime(2015, 4, 1)])

    # Show the plot
    fig.show()

if __name__ == '__main__':
    # grab data
    snp = pd.read_feather('data/sp500.feather')
    wiki = get_sp500_table()
    long_term_tickers = wiki[pd.to_datetime(wiki['Date added'], errors='coerce') <= datetime(2006, 1, 1)]['Symbol'].tolist()

    # calculate returns
    returns = (snp
               .pivot(index='Date', columns='Symbol', values='Close')
               .pct_change()
               [1:])
    returns = (returns
               [[col for col in long_term_tickers if col in returns.columns]]
               .dropna(axis=1))

    # using hyperparameters from the paper
    num_eigenvectors = returns.shape[1] // 5
    window_size = 500

    num_eigenvectors = 2
    window_size = 10

    ar = custom_rolling(returns, curry(calc_absorption_ratio, num_eigenvectors=num_eigenvectors), window_size)

    print('Saving to pickle!')
    ar.to_pickle('data/ar_sp500.pkl')

    # compare with package version
    from frds.measures import absorption_ratio

    absorption_ratio.estimate(df, fraction_eigenvectors=0.2)

    ar_frds = custom_rolling(returns, curry(absorption_ratio.estimate, fraction_eigenvectors=0.2), window_size)

    print('Saving to pickle!')
    ar_frds.to_pickle('data/ar_frds_sp500.pkl')

    # plot comparison
    # plot_series_subplots(ar[500:].astype(float).reset_index(), snp.pivot(index='Date', columns='Symbol', values='Close').sum(axis=1).reset_index()[499:-1], 0, 0, "Absorption Ratio ~ S&P 500 Returns")

    # updating with actual snp data (hackyness, ignore this)
    snp = pd.read_csv('data/snp.csv')

    ar_frds = custom_rolling(snp.set_index('Date') / 100, curry(absorption_ratio.estimate, fraction_eigenvectors=0.2), window_size)
    plot_series_subplots(#ar[500:].astype(float).reset_index(),
                         ar[:-1307].astype(float).reset_index(),
                         #returns[-220:].reset_index().rename(columns={'Real Price': 0}),
                         snp[-220:][24:].reset_index().rename(columns={'Real Price': 0}),
                         0, 0, "Absorption Ratio ~ S&P 500 Returns")
