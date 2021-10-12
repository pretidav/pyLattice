from lattice.grid import *


#def test_realfield():
Grid = LatticeParallel(grid=[200,200,200],pgrid=[2,2,2])
RealField = LatticeReal(lattice=Grid)
RealField.fill_value(3)
RealField2 = LatticeReal(lattice=Grid)
RealField2.fill_value(2)
import time
start_time = time.time()
RealField+RealField2
print("--- %s seconds ---" % (time.time() - start_time))
