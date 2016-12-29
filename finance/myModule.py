def printBla():
    print('bla')


l = ['data.' + x for x in ['data_functions', 'db_functions']]
data = __import__(l[0], globals(), locals(), [], 0)
data1 = __import__('data', globals(), locals(), [], 0)

def get_variables(pkg):
    variable_names = list(filter(lambda x: not(x.startswith('__')), pkg.__dict__.keys()))
    return {x: pkg.__dict__[x] for x in variable_names}
