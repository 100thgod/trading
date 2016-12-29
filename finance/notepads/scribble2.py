

from imp import reload

reload(portfolio_class)
reload(portfolio_functions)
portfolio = portfolio_class.Portfolio
tickers = pd.read_csv(TICKER_FILE)
p = portfolio(tickers)
p.load_data()
p.compute_returns()
p.compute_vol()
p.load_earnings_data()
p.compute_cov()

PORTFOLIO_DIR = fullfile(DATA_DIR, 'Portfolio')
DATE_FORMAT = '%Y%m%d'
transactions = pd.read_excel(fullfile(PORTFOLIO_DIR, 'history.xls'))
p.load_transactions(transactions, date_format= DATE_FORMAT)
