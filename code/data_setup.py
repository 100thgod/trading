import requests
from datetime import datetime as dt
from datetime import timedelta

myDir = '/Users/armtiger/PycharmProjects/finance/'
for file in [myDir + x for x in ['helper_functions.py', 'db_functions.py']]:
    with open(file) as f:
        code = compile(f.read(), file, 'exec')
        exec(code)

DATE_FORMAT = '%d%b%Y'
MIN_DATE = dt.strptime('01Jan1900', DATE_FORMAT)

def getGoogleData(ticker, mode, **kwargs):

    # Set parameters and check inputs:
    mode = mode.lower()
    kwargs = {key.lower(): value for key, value in kwargs.items()}
    if mode == 'intraday':
        interval_seconds = kwargs['interval_seconds'] if 'interval_seconds' in kwargs.keys() else 120
        num_days = kwargs['num_days'] if 'num_days' in kwargs.keys() else 20
    elif mode == 'daily':
        startdate = kwargs['startdate'] if 'startdate' in kwargs.keys() else MIN_DATE
        enddate = kwargs['enddate'] if 'enddate' in kwargs.keys() else dt.now().date()
    else:
        raise ValueError('''Wrong input for mode, should be either 'intraday' or 'daily'!''')

    # Get Google Finance data:
    modeStr = 'getprices' if mode == 'intraday' else 'historical'
    url_string = 'http://www.google.com/finance/{}?q={}'.format(modeStr, ticker.upper())
    if mode == 'intraday':
        url_string += '&i={0}p={1}d&f=d,o,h,l,c,v'.format(interval_seconds, num_days)
    else:
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
    d = [l.split(',') for l in r if re.match('a*\d', l) != None]
    if mode == 'intraday':
        # Cannot fully parallelise due to sequential dependence for days
        for l in d:
            if l[0].startswith('a'):
                day = pd.to_datetime(l[0][1:], unit='s')
                offset = 0
            else:
                offset = int(l[0]) * interval_seconds
            l[0] = day + timedelta(seconds = offset)
        d = [[l[0]] +  [float(s) for s in l[1:-1]] + [int(l[-1])] for l in d]
    else:
        d = [[dt.strptime(l[0], '%d-%b-%y')] + [float(s) for s in l[1:-1]] + [int(l[-1])] for l in d]
    df = pd.DataFrame(data = d, columns = colNames)
    columns = df.columns
    df_indexed = df.set_index(list(filter(lambda x: x.lower() == 'date', df.columns))[0]) # allows for different capitalisations
    return df_indexed