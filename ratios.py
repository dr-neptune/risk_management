import numpy as np
# Sharpe, Active, Tobin, Sortino, Treynor Ratio, Jensen's Alpha, Calmar, Information, M^2

# Sharpe Ratio = R_p - R_f / std(R_p)

def sharpe_ratio(returns, risk_free_rate):
    return (returns - risk_free_rate) / returns.std()

plot_multiple_time_series({'Sharpe Ratio Random': sharpe_ratio(portfolio_real_returns, baseline_real_returns),
                           'Sharpe Ratio Poor Baseline': sharpe_ratio(portfolio_real_returns, baseline_real_returns * 0.8),
                           'Sharpe Ratio Strong Baseline': sharpe_ratio(portfolio_real_returns, baseline_real_returns * 1.2)})

# sortino ratio = average returns / downside risk
# where downside risk is average negative returns
def sortino_ratio(returns, risk_free_rate):
    def downside_risk(adjusted_returns, risk_free_rate):
        sqrt_downside = np.square(np.clip(adjusted_returns, a_min=np.NINF, a_max=0))
        return np.sqrt(sqrt_downside.mean() * 252)
    adjusted_returns = returns - risk_free_rate
    return np.nanmean(adjusted_returns) * np.sqrt(252) / downside_risk(adjusted_returns, risk_free_rate)


sortino_ratio(portfolio_real_returns, baseline_real_returns)

sortino_ratio(portfolio_pct_returns, 0.05)



def calculate_sortino_ratio(portfolio_returns, risk_free_rate=0):
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
    sortino_ratio = (expected_return - risk_free_rate) / downside_std_dev

    return sortino_ratio


plot_multiple_time_series({'Sortino Ratio Random': calculate_sortino_ratio(portfolio_pct_returns, baseline_pct_returns),
                           'Sortino Ratio Poor Baseline': calculate_sortino_ratio(portfolio_pct_returns, baseline_pct_returns * 0.8),
                           'Sortino Ratio Strong Baseline': calculate_sortino_ratio(portfolio_pct_returns, baseline_pct_returns * 1.2)},
                          {'Sortino Ratio Random': 1,
                           'Sortino Ratio Poor Baseline': 1,
                           'Sortino Ratio Strong Baseline': 1})

calculate_sortino_ratio(portfolio_pct_returns, baseline_pct_returns)
