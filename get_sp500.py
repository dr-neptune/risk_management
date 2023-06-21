import pandas as pd
import yfinance as yf
from datetime import datetime

def get_sp500_table():
    # Fetch the table of S&P 500 companies from Wikipedia
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')

    # The first table on the page is the one we want
    df = table[0]

    # Keep only the first six columns
    df = df.iloc[:, :6]

    return df

if __name__ == '__main__':
    df = get_sp500_table()

    # Get the symbols from the 'Symbol' column
    symbols = df['Symbol'].values.tolist()

    # Get the data for all symbols since 01-01-2000
    start_date = datetime(2006, 1, 1)
    end_date = datetime.now()

    # Create an empty DataFrame to store the data
    companies = {}

    for symbol in symbols:
        try:
            print(f'Now working on: {symbol}')
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date).reset_index()
            data['Symbol'] = symbol  # Add a column to the data for the symbol
            stock_data = pd.merge(data, df, on='Symbol')
            companies[symbol] = stock_data
        except Exception as e:
            print(f"Could not download data for {symbol}. Reason: {e}")

    stock_data = pd.concat(companies.values())

    stock_data.reset_index().to_feather('data/sp500.feather')
