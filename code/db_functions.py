import sqlite3
from data_setup import *

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

def writeDataToDB(asset, mode, db):

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
    conn = sqlite3.connect(db)
    cur = conn.cursor()

    # Create table if not already produced
    cur.execute('''CREATE TABLE IF NOT EXISTS {} (
    Date TEXT UNIQUE NOT NULL PRIMARY KEY, {})
    '''.format(tableName, reduce(lambda x,y: x+', '+y, map(lambda x,y: x+' '+y, db_flds, types))))

    # Insert data
    sql_input = [[i.strftime(TIME_FORMAT)] + list(map(float, df.loc[i,df_flds])) for i in df.index]
    cur.executemany('''
    INSERT OR REPLACE INTO {} (Date, {}) VALUES (?,?,?,?,?,?)
    '''.format(tableName, reduce(lambda x,y: x+', '+y, db_flds)), sql_input)
    conn.commit()

def getDataFromDb(asset, db):

    # Set up database objects
    conn = sqlite3.connect(db)
    cur = conn.cursor()

    # Get dailyData
    c = cur.execute('''SELECT * FROM {}_daily'''.format(asset.ticker))
    df = pd.DataFrame(c.fetchall())
    df.columns = tuple(map(lambda x: x[0], c.description))
    df.index = pd.to_datetime(df['Date'])
    del df['Date']
    asset.dailyData  = df.sort_index()

    # Get intraday data
    c = cur.execute('''SELECT * FROM {}_intraday'''.format(asset.ticker))
    df = pd.DataFrame(c.fetchall())
    df.columns = tuple(map(lambda x: x[0], c.description))
    df.index = pd.to_datetime(df['Date'])
    del df['Date']
    asset.intradayData  = df.sort_index()
