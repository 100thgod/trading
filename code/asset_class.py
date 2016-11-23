import sys

class Asset:

    # --------------------- CONSTRUCTOR  :
    def __init__(self, ticker, mode = 'web', interval_seconds = 301, num_days = 10, startdate = MIN_DATE, enddate = dt.now().date()):
        self.ticker = ticker
        mode = mode.lower()
        if mode == 'web':
            try:
                self.intradayData = getGoogleData(\
                    self.ticker, 'intraday', interval_seconds = interval_seconds, num_days = num_days).sort_index()
            except:
                print('Could not load daily data from Google finance!')
            try:
                self.dailyData = getGoogleData(\
                    self.ticker, 'daily', startdate = startdate, enddate = enddate).sort_index()
            except:
                print('Could not load intraday data from Google finance!')
        elif mode == 'db':
            getDataFromDb(self, DB)
        else:
            raise ValueError('''mode should be either 'web' or 'db'!''')

        # compute return
        try:
            self.computeReturns()
        except:
            print('Could not compute returns, traceback message:\n {}'.format(sys.exc_info()[0]))

        # compute vol
        try:
            self.computeVol()
        except:
            print('Could not compute vol, traceback message:\n {}'.format(sys.exc_info()[0]))

        # initialise positions variable
        self.position = None

    # ------------------- METHODS :

    # Define methods to update data:
    def updateIntradayData(self, interval_seconds = 301, num_days = 10):
        df = getGoogleData(self.ticker, 'intraday', interval_seconds = interval_seconds, num_days = num_days)
        self.intradayData.append(df)
    def updateDailyData(self, startdate = MIN_DATE, enddate = dt.now().date()):
        df = getGoogleData(self.ticker, 'daily', startdate = startdate, enddate = enddate)
        self.dailyData.append(df)

    # Method to load position series
    def setPosition(self, positionSeries):
        # Takes input position series and
        self.position = positionSeries

    def computeReturns(self, method = 'arithmetic'):
        # arithmetic returns
        self.dailyData['Close-Close_arithmetic'] = self.dailyData['Close'] - self.dailyData['Close'].shift(-1)
        self.dailyData['Open-Open_arithmetic'] = self.dailyData['Open'] - self.dailyData['Open'].shift(-1)
        self.dailyData['intraday_arithmetic'] = self.dailyData['Close'] - self.dailyData['Open']
        self.dailyData['overnight_arithmetic'] = self.dailyData['Open'] - self.dailyData['Close'].shift(-1)
        # geometric returns
        self.dailyData['Close-Close_geometric'] = self.dailyData['Close-Close_arithmetic'] / self.dailyData['Close'].shift(-1)
        self.dailyData['Open-Open_geometric'] = self.dailyData['Open-Open_arithmetic'] / self.dailyData['Open'].shift(-1)
        self.dailyData['intraday_geometric'] = self.dailyData['intraday_arithmetic'] / self.dailyData['Open']
        self.dailyData['overnight_geometric'] = self.dailyData['overnight_arithmetic'] / self.dailyData['Close'].shift(-1)

    def computeVol(self, source = 'Close', alpha = 0.1):
        self.dailyVol = self.dailyData[source].ewm(alpha = alpha).std()

