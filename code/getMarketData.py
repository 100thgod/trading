
# Script which defines function to read earnings data and return a dataframe

# File setup
myDir = '/Users/armtiger/PycharmProjects/finance/'
pythonFiles = ['helper_functions.py', 'db_functions.py']
for file in [myDir + x for x in pythonFiles]:
    with open(file) as f:
        code = compile(f.read(), file, 'exec')
        exec(code)

DATA_DIR = '/Users/armtiger/Documents/Data'
DB = fullfile(DATA_DIR, 'finance_db.sqlite3')

# Get tickers
tickers = []
with open(fullfile(DATA_DIR, 'Tickers.csv'), newline = '') as csvfile:
    datastream = csv.reader(csvfile)
    for row in datastream:
        tickers.append(row[0])

# For each ticker get data
for ticker in tickers:
    a = Asset(ticker)
    writeDataToDB(a, 'intraday', DB)
    writeDataToDB(a, 'daily', DB)