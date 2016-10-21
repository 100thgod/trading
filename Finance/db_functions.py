import sqlite3
from getMarketData import *

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

DB_DIR = '/Users/armtiger/Documents/Data/'

def writeDataToDB(asset, mode):

    # Set up data variables and check mode:
    mode = mode.lower()
    if mode == 'intraday':
        df = asset.intradayData
    elif mode == 'daily':
        df = asset.dailyData
    else:
        raise ValueError('Wrong mode input, should be either intraday or daily!')
    types = ['REAL', 'REAL', 'REAL', 'REAL', 'INTEGER']
    db_flds = ['Open', 'Close', 'High', 'Low', 'Volume']
    tableName = asset.ticker + '_' + mode

    # Helper function which assumes that fld is in the column names of df, modulo capitalisation:
    getFirst = lambda l: l[0] if l else None
    getFldName = lambda fld: getFirst(list(filter(lambda x: x.lower() == fld.lower(), df.columns)))
    df_flds = [getFldName(x) for x in db_flds]

    # Set up database objects
    conn = sqlite3.connect(fullfile(DB_DIR, 'finance_db.sqlite3'))
    cur = conn.cursor()

    # Create table if not arleady produced
    cur.execute('''CREATE TABLE {} (
    Date TEXT UNIQUE NOT NULL PRIMARY KEY, {})
    '''.format(tableName, reduce(lambda x,y: x+', '+y, map(lambda x,y: x+' '+y, db_flds, types))))

    # Insert data
    sql_input = [[i.strftime(TIME_FORMAT)] + list(map(float, df.loc[i,df_flds])) for i in df.index]
    cur.executemany('''
    INSERT OR REPLACE INTO {} (Date, {}) VALUES (?,?,?,?,?,?)
    '''.format(tableName, reduce(lambda x,y: x+', '+y, db_flds)), sql_input)
    conn.commit()