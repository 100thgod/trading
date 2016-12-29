
from imp import reload

reload(portfolio_class)
reload(portfolio_functions)
portfolio = portfolio_class.Portfolio
p = portfolio(tickers)
p.load_data(db_file = DB_FILE)
p.compute_returns()
p.compute_vol()
p.load_earnings_data()
p.compute_cov()

PORTFOLIO_DIR = fullfile(DATA_DIR, 'Portfolio')
DATE_FORMAT = '%Y%m%d'
transactions = pd.read_excel(fullfile(PORTFOLIO_DIR, 'history.xls'))
p.load_transactions(transactions, date_format= DATE_FORMAT)

trades = portfolio_functions.format_transactions(transactions, date_format = DATE_FORMAT)
net_cash = df.groupby('ticker')['net cash'].sum()
share_value = pd.Series({
    'BBD-C' : 2496,
    'GM' : 2483.02,
    'AMZN' : 2026.6,
    'BRK.B' : 10866.75,
    'FSLR' : 1325.68,
    'GOOGL' : 2134.83,
    'RSX' : 8313.2,
    'SLV' : 7953.15,
    'TSLA' : 5826.16,
    'GIC_bond' : 6570 + 5000 + 4982,
    'USD_trade' : 2952.58
})
df1 = pd.DataFrame({'net cash' : net_cash, 'share value' : share_value})
df1.fillna(0, inplace = True)
df1['sum'] = df1['net cash'] + df1['share value']




summary = trades.groupby('ticker')['shares', 'cash flow'].sum()
prices = p.open_.tail(1).T
summary = summary.join(prices.rename(columns = {prices.columns[0] : 'unit price'}), how = 'left')
fx = 1.3174
summary['share value'] = summary['unit price'] * summary['shares'] * fx
summary['pnl'] = summary['share value'] + summary['cash flow']
