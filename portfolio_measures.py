import numpy as np
from scipy.stats import norm, t, lognorm


def calculate_portfolio_std_dev(portfolio_returns):
    return np.std(portfolio_returns)

def calculate_var(portfolio_returns, confidence_level=0.05, days=1, dist='normal'):
    # Compute the negative value at risk
    match dist:
        case 'normal':
            daily_var = -np.percentile(portfolio_returns, 100 * confidence_level)
        case 'lognormal':
            log_returns = np.log(1 + portfolio_returns)
            sigma = np.std(log_returns)
            mu = np.mean(log_returns)
            daily_var = -lognorm.ppf(confidence_level, sigma, scale=np.exp(mu))
        case 'student-t':
            df, loc, scale = t.fit(portfolio_returns)
            daily_var = -t.ppf(confidence_level, df, loc, scale)

    # Scale the daily VaR by the square root of the time horizon
    return daily_var * np.sqrt(days)


# def calculate_var(portfolio_returns, confidence_level=0.05, days=1, dist='normal'):
#     # Compute the negative value at risk
#     match dist:
#         case 'normal':
#             daily_var = -np.percentile(portfolio_returns, 100 * confidence_level)
#         case 'lognormal':
#             sigma = np.std(np.log(portfolio_returns))
#             mu = np.mean(np.log(portfolio_returns))
#             daily_var = -lognorm.ppf(confidence_level, sigma, scale=np.exp(mu))
#         case 'student-t':
#             df, loc, scale = t.fit(portfolio_returns)
#             daily_var = -t.ppf(confidence_level, df, loc, scale)

#     # Scale the daily VaR by the square root of the time horizon
#     return daily_var * np.sqrt(days)

# def calculate_var(portfolio_returns, confidence_level=0.05):
#     # Compute the negative value at risk
#     return -np.percentile(portfolio_returns, 100 * confidence_level)



def calculate_cvar(portfolio_returns, confidence_level=0.05, days=1, dist='normal'):
    # First, we need to calculate daily VaR
    daily_var = calculate_var(portfolio_returns, confidence_level, dist=dist)

    # CVaR is the average of losses greater than VaR
    match dist:
        case 'normal':
            daily_cvar = -np.mean(portfolio_returns[portfolio_returns < -daily_var])
        case 'lognormal':
            log_returns = np.log(1 + portfolio_returns)
            sigma = np.std(log_returns)
            mu = np.mean(log_returns)
            daily_cvar = -np.mean(lognorm.pdf(log_returns, sigma, scale=np.exp(mu))[log_returns < -daily_var])
        case 'student-t':
            df, loc, scale = t.fit(portfolio_returns)
            daily_cvar = -np.mean(t.pdf(portfolio_returns, df, loc, scale)[portfolio_returns < -daily_var])

    # Scale the daily CVaR by the square root of the time horizon
    return daily_cvar * np.sqrt(days)


def calculate_cvar(portfolio_returns, confidence_level=0.05, days=1, dist='normal'):
    # First, we need to calculate daily VaR
    daily_var = calculate_var(portfolio_returns, confidence_level, dist=dist)

    # CVaR is the average of losses greater than VaR
    daily_cvar = -np.mean(portfolio_returns[portfolio_returns < -daily_var])

    # Scale the daily CVaR by the square root of the time horizon
    return daily_cvar * np.sqrt(days)



calculate_cvar(portfolio_real_returns, days=21)
calculate_cvar(portfolio_real_returns, days=21, dist='lognormal')
calculate_cvar(portfolio_real_returns, days=21, dist='student-t')


# def calculate_cvar(portfolio_returns, confidence_level=0.05):
#     # First, we need to calculate VaR
#     var = calculate_var(portfolio_returns, confidence_level)

#     # CVaR is the average of losses greater than VaR
#     return -np.mean(portfolio_returns[portfolio_returns < -var])


def calculate_beta(portfolio_returns, market_returns):
    covariance = np.cov(portfolio_returns, market_returns)[0][1]
    market_variance = np.var(market_returns)

    return covariance / market_variance


calculate_portfolio_std_dev(portfolio_pct_returns)

calculate_var(portfolio_real_returns, days=21)
calculate_var(portfolio_real_returns, days=21, dist='student-t')
calculate_var(portfolio_pct_returns, days=21, dist='lognormal')
calculate_var(portfolio_pct_returns, days=1, dist='lognormal')

calculate_var(portfolio_real_returns)



from scipy.stats import genpareto

def calculate_var_gpd(portfolio_returns, confidence_level=0.05, days=1, threshold=0):
    # Filter the returns that exceed the threshold
    excess_returns = portfolio_returns[portfolio_returns > threshold] - threshold

    # Fit a GPD to the excess returns
    c, loc, scale = genpareto.fit(excess_returns)

    # Compute the VaR
    daily_var = threshold + scale / c * ((portfolio_returns.size / excess_returns.size * (1 - confidence_level)) ** -c - 1)

    # Scale the daily VaR by the square root of the time horizon
    return daily_var * np.sqrt(days)


calculate_var_gpd(portfolio_pct_returns, days=50, threshold=0.1) * portfolio_real_returns[-1]
calculate_var_gpd(portfolio_real_returns, days=50)





import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkAgg')

[calculate_var_gpd(portfolio_pct_returns, 0.05, 50, thresh) for thresh in np.linspace(0, 0.1, num=50)]
[calculate_var_gpd(portfolio_real_returns, 0.05, 50, thresh) for thresh in np.linspace(10000, 50000, num=50)]

def plot_var_gpd(portfolio_pct_returns, portfolio_real_returns, confidence_level=0.05, days=1, thresholds=None):
    if thresholds is None:
        # If no thresholds are provided, create a range of thresholds
        thresholds_pct = np.linspace(0, 0.1, num=100)
        thresholds_real = np.linspace(10000, 50000, num=100)

    # Calculate VaR for each threshold
    var_pct_returns = []
    for threshold in thresholds_pct:
        try:
            var = calculate_var_gpd(portfolio_pct_returns, confidence_level, days, threshold)
            var_pct_returns.append(var)
        except ValueError:
            var_pct_returns.append(np.nan)

    var_real_returns = []
    for threshold in thresholds_real:
        try:
            var = calculate_var_gpd(portfolio_real_returns, confidence_level, days, threshold)
            var_real_returns.append(var)
        except ValueError:
            var_real_returns.append(np.nan)

    # Create a figure with two subplots
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    # Plot VaR against threshold for percent returns
    ax[0].plot(thresholds_pct, var_pct_returns)
    ax[0].set_title('VaR vs Threshold (Percent Returns)')
    ax[0].set_xlabel('Threshold')
    ax[0].set_ylabel('VaR')

    # Plot VaR against threshold for real returns
    ax[1].plot(thresholds_real, var_real_returns)
    ax[1].set_title('VaR vs Threshold (Real Returns)')
    ax[1].set_xlabel('Threshold')
    ax[1].set_ylabel('VaR')

    # Display the plots
    plt.tight_layout()
    plt.show()



calculate_var_gpd(portfolio_pct_returns, 0.05, 50, 0.04)

# Use the function with your data
plot_var_gpd(portfolio_pct_returns, portfolio_real_returns, days=50)
plt.show()

def plot_gpd_fit(portfolio_real_returns, confidence_level=0.05, days=1, threshold=0):
    # Filter the returns that exceed the threshold
    excess_returns = portfolio_real_returns[portfolio_real_returns > threshold] - threshold

    # Fit a GPD to the excess returns
    c, loc, scale = genpareto.fit(excess_returns)

    # Calculate VaR for the fitted GPD
    var_gpd = calculate_var_gpd(portfolio_real_returns, confidence_level, days, threshold)

    # Create a range of values for the x-axis
    x_values = np.linspace(min(portfolio_real_returns), max(portfolio_real_returns), num=100)

    # Calculate the corresponding values of the fitted GPD
    gpd_values = genpareto.pdf(x_values, c, loc, scale)

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the portfolio real returns as a scatter plot
    ax.scatter(range(len(portfolio_real_returns)), portfolio_real_returns, label='Real Returns')

    # Plot the threshold
    ax.axhline(y=threshold, color='r', linestyle='--', label='Threshold')

    # Plot the fitted GPD
    ax.plot(x_values, gpd_values, label='Fitted GPD')

    # Add a legend
    ax.legend()

    # Display the plot
    plt.show()

# Use the function with your data
plot_gpd_fit(portfolio_real_returns)
