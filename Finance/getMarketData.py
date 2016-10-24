from data_setup import *
from db_functions import *

DATA_DIR = '/Users/armtiger/Documents/Data'
DB = fullfile(DATA_DIR, 'finance_db.sqlite3')

# Get tickers
tickers = []
with open(fullfile(DATA_DIR, 'Tickers.csv'), newline = '') as csvfile:
    datastream = csv.reader(csvfile)
    for row in datastream:
        tickers.append(row[0])

# for each ticker get data
for ticker in tickers:
    a = Asset(ticker)
    writeDataToDB(a, 'intraday', DB)
    writeDataToDB(a, 'daily', DB)