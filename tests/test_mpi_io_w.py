from IO.MPI_IO import Writer, Reader
from mpi.parallel import CartesianComm
from lattice.grid import LatticeComplex, LatticeReal, LatticeComplexMatrix, LatticeRealMatrix, LatticeVectorReal, LatticeVectorComplex, LatticeVectorRealMatrix, LatticeVectorComplexMatrix
import numpy as np

tol = 10e-6
mpigrid = [1, 2, 2]
grid = [4,2,2]
CC = CartesianComm(mpigrid=mpigrid)

def test_latticereal():
    Lattice = LatticeReal(grid=grid, cartesiancomm=CC)
    Lattice.fill_value(n=np.array(
        [int(i)+1 for i in range(0,np.prod(grid))], dtype='float32')+CC.rank*100)
    
    print('{} \n {}'.format(CC.mpicoord,Lattice.value.reshape(grid)))
    W = Writer(lattice=Lattice,cartesiancomm=CC)
    W.write(filename='./test_file_'+'_'.join([str(a) for a in mpigrid]))

if __name__=='__main__': 
    test_latticereal()
