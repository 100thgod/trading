
from pandas import DataFrame
from re import split
from numpy import where
from datetime import datetime as dt


def format_transactions(transactions, trade_labels = ['BUY', 'SELL'], dividend_labels = ['DIV', 'NRT'], inplace = False, date_format = None):
    """ Formats trades, dividends and cash dataframes from transactions dataframe. """

    # If inplace, work with copy of transactions dataframe
    transactions = transactions.copy()

    # Handle date formatting (optional)
    if date_format:
        transactions['transaction date'] = transactions['transaction date'].map(str).map(lambda x: dt.strptime(x, date_format))

    # Backward adjust quantities for splits
    split_idx = transactions['transaction type'].map(lambda x: (type(x) == str) and (x.lower().startswith('split')))
    split_ratio = transactions['transaction type'][split_idx].str.split(':').map(lambda x: x[0]).str.split(' ').map(lambda x: x[1])
    for idx, ratio in zip(split_ratio.index, split_ratio):
        ticker = transactions.loc[idx, 'ticker']
        mask = transactions.index[(transactions.index < idx) & (transactions['ticker'] == ticker)]
        transactions.loc[mask, 'quantity'] *= int(ratio)

    # Format trades dataframe
    df = transactions[transactions['transaction type'].map(lambda x: x in trade_labels)]
    trades = df[['transaction date', 'ticker', 'commission', 'net cash']].rename(columns = {'transaction date' : 'date', 'net cash' : 'cash flow'})
    trades['shares'] = df['quantity'] * df['transaction type'].map(lambda x: 1 if x == 'BUY' else (-1 if x == 'SELL' else 0))
    trades.set_index('date', inplace = True)

    # Format dividends dataframe
    df = transactions[transactions['transaction type'].map(lambda x: x in dividend_labels)]
    dividends = df[['transaction date', 'ticker', 'quantity', 'net cash']].rename(columns = {'transaction date' : 'date', 'quantity' : 'shares', 'net cash' : 'cash flow'})
    dividends.set_index('date', inplace = True)

    # Format cash dataframe
    cash = transactions[['transaction date', 'ticker', 'net cash']].rename(columns={'transaction date': 'date', 'net cash': 'cash flow'})
    cash.set_index('date', inplace = True)

    return trades, dividends, cash

def sharpe(returns):
    """ Computes sharpe ratio of pandas series of returns. """
    return returns.mean() / returns.std()

def drawdown(returns):
    """" Computes drawdown of pandas series of returns. """
    val = returns.cumsum()
    running_max = val.expanding().max()
    drawdown_series = val - running_max
    return drawdown_series