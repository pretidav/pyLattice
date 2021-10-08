from lattice.grid import *


p = LatticeParallel(grid=[4,4], pgrid=[1,2])

#@p.parallelize(platt=[2,3])
#def f():
#    print('ciao')

#f()