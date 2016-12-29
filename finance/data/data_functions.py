
import requests
from fredapi import Fred
from datetime import datetime as dt
from datetime import timedelta
from re import match
from pandas import DataFrame, to_datetime
from yahoo_finance import Share

DATE_FORMAT = '%d%b%Y'

def getGoogleData(ticker, mode = 'daily', interval_seconds = 120, num_days = 20):
    """ Functions which loads data from Google finance

    Parameters
    ----------
    ticker : string
        asset ticker for which data is to be loaded
    mode : string (default = 'daily')
        either 'intraday' or 'daily', depending on the frequency of the data we want to load
    interval_seconds : int (default = 120)
        determines sampling frequency for intraday mode
    num_days : int (default = 20)
        number of days of data for intraday mode

    Returns
    -------
    df : pandas dataframe with datetime index

    """

    mode = mode.lower()

    # Get Google Finance data:
    modeStr = 'getprices' if mode == 'intraday' else 'historical'
    url_string = 'http://www.google.com/finance/{}?q={}'.format(modeStr, ticker.upper())
    if mode == 'intraday':
        url_string += '&i={0}p={1}d&f=d,o,h,l,c,v'.format(interval_seconds, num_days)
    else:
        startdate = dt.strptime('01Jan1900', DATE_FORMAT)
        enddate = dt.now().date()
        url_string += '&startdate={}&enddate={}&output=csv'.format(startdate.strftime('%d %b %Y'), enddate.strftime('%d %b %Y'))
    r = requests.get(url_string).text.split('\n')

    # Format data and return pandas dataframe:
    if mode == 'intraday':
        # Get column names:
        prefix = 'COLUMNS='
        s = list(filter(lambda s: s.startswith(prefix), r))[0]
        colNames = s[len(prefix):].split(',')
    else:
        colNames = r[0].split(',') # columns names
    d = [l.split(',') for l in r if match('a*\d', l) != None]
    if mode == 'intraday':
        # Cannot fully vectorise due to sequential dependence for time information (i.e. 'offset' variable)
        for l in d:
            if l[0].startswith('a'):
                day = to_datetime(l[0][1:], unit='s')
                offset = 0
            else:
                offset = int(l[0]) * interval_seconds
            l[0] = day + timedelta(seconds = offset)
        d = [[l[0]] +  [float(s) for s in l[1:-1]] + [int(l[-1])] for l in d]
    else:
        d = [[dt.strptime(l[0], '%d-%b-%y')] + [float(s) for s in l[1:-1]] + [int(l[-1])] for l in d]
    df = DataFrame(data = d, columns = colNames)
    columns = df.columns
    df_indexed = df.set_index(list(filter(lambda x: x.lower() == 'date', df.columns))[0]) # allows for different capitalisations
    return df_indexed

"""
requests.get('http://ichart.finance.yahoo.com/table.csv?s=GM').text.split('\n')

"""

def getYahooData(ticker):
    """ Basic data function which loads daily data from Yahoo Finance. """
    url_string = 'http://ichart.finance.yahoo.com/table.csv?s={}'.format(ticker)
    r = requests.get(url_string).text.split('\n')
    colNames = r[0].split(',')
    d = [l.split(',') for l in r[1:-1]]
    df = DataFrame(data = d, columns = colNames)
    df['Date'] = df['Date'].map(lambda x: dt.strptime(x, '%Y-%m-%d'))
    df.set_index('Date', inplace = True)
    df = df.applymap(float)
    df['Volume'] = df['Volume'].map(int)
    return df
