
from datetime import datetime as dt
from helper_functions import fullfile
from pandas import read_excel
from data_functions import getGoogleData

# Defines class for assets and some methods
DATE_FORMAT = '%d%b%Y'
MIN_DATE = dt.strptime('01Jan1900', DATE_FORMAT)
DATA_DIR = '/Users/armtiger/Documents/Data'

DEFAULT_PARAMS = {
    'interval_seconds' : 301,
    'num_days' : 10,
    'db_file' : fullfile(DATA_DIR, 'finance_db.sqlite3'),
    'earnings_dir' : fullfile(DATA_DIR, 'Earnings')
}

import sys

class Asset:
    """ Class for storing asset information

    Parameters
    ----------
    ticker : string
        asset ticker
    db_file : string (default = None)
        address of database storing price information
    earnings_file: string (default = None)
        address of .xls file storing earnings information
    interval_seconds, num_days : int
        parameters to be passed for loading Google finance intraday data

    Attributes
    ----------
    daily_data_ : pandas dataframe
        Data on daily prices
    intraday_data_ : pandas dataframe
        Data on intraday prices

    """

    def __init__(self, ticker, db_file = None, earnings_file = None, interval_seconds = 300, num_days = 20):
        # set up parameters
        self.ticker = ticker
        self.db_file = db_file
        self.earnings_file = earnings_file if earnings_file else fullfile(DATA_DIR, 'earnings_dir', ticker + '.xls')
        self.interval_seconds = interval_seconds
        self.num_days = num_days

        if db_file:
            getDataFromDb(self, self.db_file)
        else:
            self.intradayData = getGoogleData(self.ticker, 'intraday', interval_seconds = self.interval_seconds, num_days = self.num_days).sort_index()
            self.dailyData = getGoogleData(self.ticker, 'daily').sort_index()

