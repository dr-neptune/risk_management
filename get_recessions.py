
from dateutil import parser
import pandas as pd
import re


def get_recessions():
    """Grabs the list of recessions from Wikipedia and returns a dataframe"""
    def parse_dates(row):
        if type(row) != str:
            return pd.Series([None, None])
        # split the range into start and end, and remove citations
        start_date, end_date = re.sub(r'\[\d+\]', '', row).split('–')

        # parse dates and format as 'YYYY-MM-DD'
        start_date = parser.parse(start_date).strftime('%Y-%m-%d')
        end_date = parser.parse(end_date).strftime('%Y-%m-%d')

        return pd.Series([start_date, end_date])

    url = "https://en.wikipedia.org/wiki/List_of_recessions_in_the_United_States#Great_Depression_onward_(1929%E2%80%93present)"
    tables = pd.read_html(url)
    df = tables[2]
    df[['Begin_Date', 'End_Date']] = df['Period Range'].apply(parse_dates)
    return df[1:]
