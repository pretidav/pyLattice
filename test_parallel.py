from lattice.grid import *


p = LatticeParallel(grid=[4,4], pgrid=[1,2])

@parallelize(platt=[2,3],N_threads=2)
def f(latt=None):
    print(latt)

f()