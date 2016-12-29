
# Does necessary imports, defines some helper functions and runs python scripts

import os, re, csv, math, datetime, helper_functions, sys
from functools import reduce
from helper_functions import fullfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LOAD_DATA = False
DATA_DIR = '/Users/armtiger/Documents/Data'
DB_FILE = fullfile(DATA_DIR, 'finance_db.sqlite3')
TICKER_FILE = fullfile(DATA_DIR, 'Tickers.csv')

#----------------------------------- RUN PYTHON FILES

# Run all file sin following directories
from data import data_functions, db_functions as data

packages = ['data', 'portfolio', 'plots']
for pkg in packages:
    iDir = fullfile(os.getcwd(), pkg)
    modules = [x.replace('.py', '') for x in os.listdir(iDir) if not(x.startswith('__'))]
    if not(iDir in sys.path):
        sys.path.append(iDir)
    for module in modules:
        exec('import {}'.format(module))

# Rename common variables
Asset = asset_class.Asset
Portfolio = portfolio_class.Portfolio
getGoogleData = data_functions.getGoogleData
getYahooData = data_functions.getYahooData
Portfolio = portfolio_class.Portfolio

#--------------------------- LOAD MARKET DATA INTO DB (OPTIONAL)
if LOAD_DATA:
    tickers = pd.read_csv(TICKER_FILE, header = None)[0]
    # For each ticker get data_setup
    for ticker in tickers:
        for mode in ['daily', 'intraday']:
            try:
                if mode == 'daily':
                    df = getGoogleData(ticker, mode)
                else:
                    df = getYahooData(ticker)
                db_functions.writeDataToDB(ticker, df, DB_FILE, mode)
                print('Updated {0} data for {1}'.format(mode, ticker))
            except Exception as e:
                print('Could not update {0} data for {1}: {2}'.format(mode, ticker, e))

