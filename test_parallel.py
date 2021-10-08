from lattice.grid import *


p = LatticeParallel(grid=[4,4], pgrid=[1,2])

@p.parallelize
def f():
    print('ciao')

f()