import numpy as np
from cytoolz import curry
# Sharpe, Active, Tobin, Sortino, Treynor Ratio, Jensen's Alpha, Calmar, Information, M^2

# Sharpe Ratio = R_p - R_f / std(R_p)
def sharpe_ratio(returns, risk_free_rate):
    return (returns.mean() - risk_free_rate) / returns.std()

calculate_rolling_return(portfolio_pct_returns * 100, sharpe_ratio, risk_free_rate=0.05)

sharpe_ratio_rolling_return(portfolio_pct_returns, 0.05)

# sortino ratio = average returns / downside risk
def sortino_ratio(portfolio_returns, risk_free_rate=0):
    """
    :param portfolio_returns: Series of portfolio returns
    :param risk_free_rate: Risk-free rate of return
    :return: Sortino ratio
    """
    # Calculate the expected return
    expected_return = portfolio_returns.mean()
    # Calculate the standard deviation of negative returns
    negative_returns = portfolio_returns[portfolio_returns < 0]
    downside_std_dev = negative_returns.std(ddof=0)
    # Calculate the Sortino ratio
    sortino = (expected_return - risk_free_rate) / downside_std_dev
    return sortino


def calculate_rolling_return(returns, aggregate_fn, n_months=6, *args, **kwargs):
    return returns.rolling(int(n_months * 30.5)).apply(aggregate_fn, raw=True, args=args, kwargs=kwargs)


if __name__ == '__main__':
    portfolio_pct_returns *= 100
    sharpe_rolling = lambda rets, rfr: calculate_rolling_return(rets, sharpe_ratio, risk_free_rate=rfr)
    sortino_rolling = lambda rets, rfr: calculate_rolling_return(rets, sortino_ratio, risk_free_rate=rfr)

    plot_multiple_time_series({'Sharpe Ratio 5% risk-free': sharpe_rolling(portfolio_pct_returns, 0.05),
                               'Sharpe Ratio 1% risk-free': sharpe_rolling(portfolio_pct_returns, 0.01),
                               'Sharpe Ratio 10% risk-free': sharpe_rolling(portfolio_pct_returns, 0.1)},
                              file_name='sharpe_ratio',
                              title='Sharpe Ratio [1% / 5% / 10% risk-free]<br>6 month rolling window')

    plot_multiple_time_series({'Sortino Ratio 5% risk-free': sortino_rolling(portfolio_pct_returns, 0.05),
                               'Sortino Ratio 1% risk-free': sortino_rolling(portfolio_pct_returns, 0.01),
                               'Sortino Ratio 10% risk-free': sortino_rolling(portfolio_pct_returns, 0.1)},
                              {'Sortino Ratio Random': 1,
                               'Sortino Ratio Poor Baseline': 1,
                               'Sortino Ratio Strong Baseline': 1},
                              file_name='sortino_ratio',
                              title='Sortino Ratio [1% / 5% / 10% risk-free]<br>6 month rolling window')
