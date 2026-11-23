from IO.MPI_IO import Writer, Reader
from mpi.parallel import CartesianComm
from lattice.grid import LatticeComplex, LatticeReal, LatticeComplexMatrix, LatticeRealMatrix, LatticeVectorReal, LatticeVectorComplex, LatticeVectorRealMatrix, LatticeVectorComplexMatrix
import numpy as np

tol = 10e-6
mpigrid = [1,1,2]
grid = [4,4,2]
CC = CartesianComm(mpigrid=mpigrid)

def test_latticereal():
    R = Reader(cartesiancomm=CC)

    Lattice = R.read(filename='./test_file_1_2_2')
    print('{} \n {}'.format(CC.mpicoord,Lattice.value.reshape(grid)))
    #Lattice.moveforward(mu=1)
    #print('{} \n {}'.format(CC.mpicoord,Lattice.value.reshape(grid)))
if __name__=='__main__': 
    test_latticereal()
