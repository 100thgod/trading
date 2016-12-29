
from sqlite3 import connect
from helper_functions import fullfile
from functools import reduce
from pandas import DataFrame, to_datetime

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

def getDataFromDB(ticker, db):
    """ Loads data from database file to asset object.

    Parameters
    ----------
    ticker : str
        ticker used in database
    db_file : str
        address of database file where data is to be loaded from

    Returns
    -------
    dict with dailyData and intradayData

    """

    # Set up database objects
    conn = connect(db)
    cur = conn.cursor()

    # Get intraday data and daily data
    for mode in ['intraday', 'daily']:
        c = cur.execute('''SELECT * FROM {0}_{1}'''.format(ticker, mode))
        df = DataFrame(c.fetchall())
        df.columns = tuple(map(lambda x: x[0], c.description))
        df.index = to_datetime(df['Date'])
        del df['Date']
        if mode == 'intraday':
            intraday_data = df.sort_index()
        else:
            daily_data = df.sort_index()
    return {'daily_data': daily_data, 'intraday_data': intraday_data}

def writeDataToDB(ticker, df, db_file, mode):
    """ Appends data from asset object to database file, overwriting data if need be.

    Parameters
    ----------
    df : pandas dataframe
        price information with information about open, close, high, low and volume information
    db : address of database file where data is to be written to.
    mode : 'daily' or 'intraday'

    """

    # Set up data variables and database objects:
    types = ['REAL', 'REAL', 'REAL', 'REAL', 'INTEGER']
    db_flds = ['Open', 'Close', 'High', 'Low', 'Volume']
    conn = connect(db_file)
    cur = conn.cursor()

    # Helper function which assumes that fld is in the column names of df, modulo capitalisation:
    get_first = lambda l: l[0] if l else None
    get_fld_name = lambda fld, df: get_first(list(filter(lambda x: x.lower() == fld.lower(), df.columns)))

    df_flds = [get_fld_name(x, df) for x in db_flds]
    table_name = ticker + '_' + mode

    # Create table if not already produced
    cur.execute('''CREATE TABLE IF NOT EXISTS {} (
    Date TEXT UNIQUE NOT NULL PRIMARY KEY, {})
    '''.format(table_name, reduce(lambda x,y: x+', '+y, map(lambda x,y: x+' '+y, db_flds, types))))

    # Insert data and commit updates
    sql_input = [[i.strftime(TIME_FORMAT)] + list(map(float, df.loc[i,df_flds])) for i in df.index]
    cur.executemany('''
    INSERT OR REPLACE INTO {} (Date, {}) VALUES (?,?,?,?,?,?)
    '''.format(table_name, reduce(lambda x,y: x+', '+y, db_flds)), sql_input)
    conn.commit()