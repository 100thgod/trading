
# Common helper functions
from functools import reduce

isCsv = lambda fileName: not (re.match('.csv', fileName[-4:]) is None)
getTicker = lambda csvFile: re.split('.csv', csvFile)[0]
subset = lambda x, idx: [x[i] for i in range(len(x)) if idx[i]]
fullfile = lambda *args: reduce(lambda x,y: x + '/' + y, args)
matlab2python_date = lambda matlab_datenum: datetime.fromordinal(int(matlab_datenum - 366)).date().isoformat()
composeFunctions = lambda f,g: (lambda x: f(g(x)))
subsetIdx = lambda x,y: list(map(lambda z: z in y, x))
filesInDir = lambda x: [fullfile(x, y) for y in os.listdir(x)]

def assign_subset(x, idx, y):
    """ Assigns values of x[idx] to y. """
    if isinstance(idx[0], bool) or isinstance(idx[0], np.bool_):
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

def run_files(python_files):
    """ Run python files """
    for file in python_files:
        with open(file) as f:
            code = compile(f.read(), file, 'exec')
            exec(code)