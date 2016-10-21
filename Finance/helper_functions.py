import os, re, csv, math
from functools import reduce
import pandas as pd
import numpy as np

isCsv = lambda fileName: not (re.match('.csv', fileName[-4:]) is None)
getTicker = lambda csvFile: re.split('.csv', csvFile)[0]
subset = lambda x, idx: [x[i] for i in range(len(x)) if idx[i]]
fullfile = lambda *args: reduce(lambda x,y: x + '/' + y, args)
matlab2python_date = lambda matlab_datenum: datetime.fromordinal(int(matlab_datenum - 366)).date().isoformat()
composeFunctions = lambda f,g: (lambda x: f(g(x)))
subsetIdx = lambda x,y: list(map(lambda z: z in y, x))

def assign_subset(x, idx, y):
    if isinstance(idx[0], bool) or isinstance(idx[0], numpy.bool_):
        idx = subset(range(len(idx)),idx)
    j = 0
    if not(isinstance(y, list)):
        y = [y] * len(idx)
    elif len(y) == 1:
        y = y * len(idx)
    for i in range(len(idx)):
        x[idx[i]] = y[j]
        j = j + 1
    return x

def alignData(data, tickers):
    dataOut = pd.DataFrame()
    for i, iData in enumerate(data):
        iDate = list(map(composeFunctions(matlab2python_date, float), iData.pop('Date')))
        iPd = pd.DataFrame(iData, index = iDate)
        iPd.columns = list(map(lambda key: tickers[i] + '_' + key, iPd.columns))
        dataOut = pd.merge(dataOut, iPd, how = 'outer', left_index = True, right_index = True)
    return dataOut