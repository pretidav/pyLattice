from lattice.grid import *


#def test_realfield():
Grid = LatticeParallel(grid=[50,50,50,50],pgrid=[2,2,1,1])
RealField = LatticeReal(lattice=Grid)
RealField.fill_value(3)
RealField2 = LatticeReal(lattice=Grid)
RealField2.fill_value(2)

MatrixField = LatticeVectorRealMatrix(lattice=Grid,Nd=2,N=3)
MatrixField2 = LatticeVectorRealMatrix(lattice=Grid,Nd=2,N=3)
V = VectorRealMatrix(Nd=2,N=3,value=np.array([[[1,2,3],[1,2,3],[1,2,3]],[[4,5,6],[4,5,6],[4,5,6]]]))
MatrixField.fill_value(V)
MatrixField2.fill_value(V)


import time
start_time = time.time()

MatrixField*MatrixField2
print("--- %s seconds ---" % (time.time() - start_time))


