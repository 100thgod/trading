
# Defines portfolio class
from pandas import DataFrame, Series, read_excel
from helper_functions import fullfile
from data.data_functions import getGoogleData
from data.db_functions import getDataFromDB
from data.data_functions import getYahooData
from numpy import setdiff1d, union1d, unique, array
from portfolio.portfolio_functions import format_transactions

# Default params
VOL_PARAMS = {
    'method' : 'exponential',
    'alpha' : 0.95,
    'window' : 20,
    'returns' : 'open'
}
COV_PARAMS = {
    'method' : 'rolling',
    'alpha' :  0.97,
    'window' : 50,
    'returns' : 'open'
}
DATA_DIR = '/Users/armtiger/Documents/Data'
DB_FILE = fullfile(DATA_DIR, 'finance_db.sqlite3')
TICKER_FILE = fullfile(DATA_DIR, 'Tickers.csv')
EARNINGS_DIR = fullfile(DATA_DIR, 'Earnings')
PORTFOLIO_DIR = fullfile(DATA_DIR, 'Portfolio')

class Portfolio:
    """ Class for storing portfolio information

        Parameters
        ----------
        tickers : string
            asset ticker
        earnings_dir: string (default = None)
            address of .xls file storing earnings information
        vol_params : dict with volatility params

        Attributes
        ----------
        open_data_, close_data_, high_data_, low_data_, volume_data_ : pandas dataframe
            Data on daily prices
        open_to_open_, open_to_close_, close_to_open_: pandas dataframes
        vol_ : pandas dataframe
            daily vol
        positions : pandas dataframe
            columns = tickers
        trades : pandas dataframe
            columns = ['date', 'ticker', 'transaction'], transaction > 0 for buy and < 0 for sell

        """

    def __init__(self, tickers, vol_params = VOL_PARAMS):
        self.tickers = array(tickers)

        #self.dates = Series(self.OpenPrices.index)

        #self.positions = DataFrame(index = self.dates, columns = tickers, data = 0)

        #self.vol = self.returns.ewm(alpha = volAlpha).std()
        #self.volNormRets = self.returns / self.vol.shift(-1)
        #self.cov = self.returns.rolling(window = covWindow)

    def load_data(self, db_file = None):
        """ Loads online data by default, or loads from database if db_file is specified. """
        open, close, high, low, volume = dict(), dict(), dict(), dict(), dict()
        tickers_loaded = []
        for ticker in self.tickers:
            try:
                if db_file:
                    data = getDataFromDB(ticker, db_file)['daily_data']
                    tickers_loaded.append(ticker)
                else:
                    data = getYahooData(ticker).rename(columns = {'Adj Close' : 'Close', 'Close' : 'Unadj Close'})
                    tickers_loaded.append(ticker)
                open[ticker], close[ticker], high[ticker], low[ticker], volume[ticker] = \
                    data['Open'], data['Close'], data['High'], data['Low'], data['Volume']
            except Exception as e:
                print('Could not load data for {0}: {1}'.format(ticker, e))
        print('Loaded data for:\n {}'.format(tickers_loaded))
        self.open_, self.close_, self.high_, self.low_, self.volume_ = \
            DataFrame(open), DataFrame(close), DataFrame(high), DataFrame(low), DataFrame(volume)
        self.dates_ = Series(self.open_.index)

    def compute_returns(self):
        """ Computes open_to_open, close_to_close, open_to_close and overnight returns. """
        self.open_to_open_ = self.open_.diff()
        self.close_to_close_ = self.close_.diff()
        self.open_to_close_ = self.close_ - self.open_
        self.overnight_ = self.open_ - self.close_.shift(1)

    def compute_vol(self, method = 'exponential', alpha = 0.05, window = 20, returns = 'open'):
        """ Computes vol of assets in portfolio. """
        if (method not in ['exponential', 'rolling']) or (returns not in ['open', 'close']):
            raise ValueError('Wrong inputs for computing vol!')
        self.vol_params_ = {'method' : method, 'returns' : returns}
        returns = self.open_to_open_ if returns == 'open' else self.close_to_close_
        if method == 'exponential':
            self.vol_ = returns.ewm(alpha = alpha).std()
            self.vol_params_['alpha'] = alpha
        elif method == 'rolling':
            self.vol_ = returns.rolling(window = window).std()
            self.vol_params_['window'] = window

    def compute_cov(self, method = 'rolling', alpha = '0.025', window = 50, returns = 'open'):
        """ Computes vol of assets in portfolio. """
        if (method not in ['exponential', 'rolling']) or (returns not in ['open', 'close']):
            raise ValueError('Wrong inputs for computing vol!')
        self.cov_params_ = {'method' : method, 'returns': returns}
        returns = self.open_to_open_ if returns == 'open' else self.close_to_close_
        if method == 'exponential':
            self.cov_ = returns.ewm(alpha = alpha).cov()
            self.corr_ = returns.ewm(alpha = alpha).corr()
            self.cov_params_['alpha'] = alpha
        elif method == 'rolling':
            self.cov_ = returns.rolling(window = window).cov()
            self.corr_ = returns.rolling(window=window).corr()
            self.cov_params_['window'] = window

    def load_earnings_data(self, earnings_dir = EARNINGS_DIR):
        """ Loads earnings data. """
        self.eps_, self.revenue_ = dict(), dict()
        tickers_loaded = []
        for ticker in self.tickers:
            try:
                earnings_file = fullfile(earnings_dir, ticker + '.xls')
                df = read_excel(earnings_file)
                # split dataframe into ETS and load info
                self.eps_[ticker] = df.loc[:'Revenue'].loc['Wall St.':'Actual'].T
                self.revenue_[ticker] = df.loc['Revenue':].loc['Wall St.':'Actual'].T
                tickers_loaded.append(ticker)
            except Exception as e:
                print('Could not load earnings data for {0}: {1}'.format(ticker, e))
        print('Loaded earnings data for:\n {}'.format(tickers_loaded))

    def load_transactions(self, transactions, augment_tickers = False, date_format = '%Y%m%d'):
        # Loads trade and dividend info from transactions dataframe
        trades, dividends, cash = format_transactions(transactions, date_format = date_format)
        # Handle ticker info
        trades_tickers = unique(trades['ticker'])
        if augment_tickers:
            self.tickers = union1d(self.tickers, trades_tickers)
            print('The portfolio was augmented with teh following tickers:')
            print(setdiff1d(trades_tickers, self.tickers))
        else:
            print('The trades for the following assets were not included in the portfolio:')
            print(setdiff1d(trades_tickers, self.tickers))

        # Compute positions from trade info.
        df = trades.groupby('ticker').apply(lambda x: x.resample('D').sum())
        self.positions_ = df['shares'].unstack(level='ticker').shift(1).fillna(0).cumsum().applymap(int)

        # Load cash flow and cash dataframes
        self.cash_flow_ = cash.resample('D').sum().fillna(0)
        self.cash_ = self.cash_flow_.cumsum()

        # Load dividend dataframe
        self.dividends_ = dividends.resample('D').sum()

