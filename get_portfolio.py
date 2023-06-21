import yfinance as yf
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
matplotlib.use('tkAgg')


# Technology: Apple (AAPL), Microsoft (MSFT), Amazon (AMZN), Google (GOOGL)
# Financials: JPMorgan Chase (JPM), Bank of America (BAC)
# Health Care: Johnson & Johnson (JNJ), Pfizer (PFE)
# Consumer Discretionary: Tesla (TSLA), McDonald's (MCD)
# Consumer Staples: Procter & Gamble (PG), Coca-Cola (KO)
# Energy: Exxon Mobil (XOM), Chevron (CVX)
# Industrials: Boeing (BA), 3M (MMM)
# Telecommunications: AT&T (T), Verizon (VZ)
# Utilities: NextEra Energy (NEE), Dominion Energy (D)

# tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'JPM', 'BAC', 'JNJ',
#            'PFE', 'TSLA', 'MCD', 'PG', 'KO', 'XOM', 'CVX', 'BA', 'MMM', 'T',
#            'VZ', 'NEE', 'D']

# Assign initial equal weights
weights = [1/20] * 20

# Create a dictionary with the sectors for each ticker
sectors = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'AMZN': 'Technology', 'GOOGL': 'Technology',
    'JPM': 'Financials', 'BAC': 'Financials',
    'JNJ': 'Healthcare', 'PFE': 'Healthcare',
    'TSLA': 'Consumer Discretionary', 'MCD': 'Consumer Discretionary',
    'PG': 'Consumer Staples', 'KO': 'Consumer Staples',
    'XOM': 'Energy', 'CVX': 'Energy',
    'BA': 'Industrials', 'MMM': 'Industrials',
    'T': 'Telecommunications', 'VZ': 'Telecommunications',
    'NEE': 'Utilities', 'D': 'Utilities'
}


def fetch_stock_data(tickers):
    def fetch_ticker(ticker, start='2013-01-01', end='2023-12-31'):
        return yf.download(ticker, start=start, end=end)['Adj Close']
    return {ticker: fetch_ticker(ticker) for ticker in tickers}


def create_portfolio(tickers_by_sector, weights):
    # Increase weights for Technology and Healthcare sectors
    for i, ticker in enumerate(tickers_by_sector):
        if sectors[ticker] in ['Technology', 'Healthcare']:
            weights[i] = weights[i] * 1.5

    # Normalize the weights so they sum to 1
    total_weight = sum(weights)
    weights = [weight/total_weight for weight in weights]

    portfolio = dict(zip(tickers_by_sector, weights))
    return portfolio

if __name__ == '__main__':
    portfolio = create_portfolio(sectors, weights)
    stock_data = pd.DataFrame(fetch_stock_data(portfolio.keys()))

    # calculate portfolio returns according to portfolio allocations
    baseline_pct_returns = stock_data.pct_change().mean(axis=1)
    portfolio_pct_returns = (stock_data.pct_change() * portfolio.values()).sum(axis=1)

    initial_investment = 1000

    # get real returns based on initial investment
    portfolio_real_returns = (stock_data * portfolio.values()).sum(axis=1) * initial_investment
    baseline_real_returns = stock_data.mean(axis=1) * initial_investment

    # plot
    from plots import plot_multiple_time_series


    plot_multiple_time_series({'portfolio percent returns': portfolio_pct_returns,
                               'benchmark percent returns': baseline_pct_returns})


    plot_multiple_time_series({'portfolio real returns': portfolio_real_returns,
                               'benchmark real returns': baseline_real_returns})
