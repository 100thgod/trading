
NAME = 'data'

import os
l = [NAME + '.' + x.replace('.py', '') for x in os.listdir(os.getcwd() + '/' + NAME) if not(x.startswith('__'))]

for module in l:
    __import__(module)

del os, l, module