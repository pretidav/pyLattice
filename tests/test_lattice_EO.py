from lattice.grid import LatticeReal, LatticeComplex, LatticeRealMatrix, LatticeComplexMatrix, LatticeVectorReal, LatticeVectorComplex, LatticeVectorRealMatrix, LatticeVectorComplexMatrix
from linalg.tensors import RealMatrix, ComplexMatrix, VectorReal, VectorComplex, VectorRealMatrix, VectorComplexMatrix
from mpi.parallel import CartesianComm
import numpy as np

mpigrid = [1, 1]
grid = [4, 4]
tol = 10e-6
CC = CartesianComm(mpigrid=mpigrid)


def test_realfield():
    Lattice = LatticeReal(grid=grid, cartesiancomm=CC)
    Lattice.fill_value( n = np.array([i for i in range(np.prod(grid))], dtype='float32') )
    Lattice_E, Lattice_O = Lattice.peek_EO_lattices()
    assert( (np.abs(Lattice_E.value - np.array([ 1.,  3.,  4.,  6.,  9., 11., 12., 14.]))<tol ).all())
    assert( (np.abs(Lattice_O.value - np.array([ 0.,  2.,  5.,  7.,  8., 10., 13., 15.]))<tol ).all())
    Lattice_E = Lattice_E*-1
    Lattice.poke_EO_lattices(E_lattice=Lattice_E, O_lattice=Lattice_O)
    assert( (np.abs(Lattice.value - np.array([0.,  -1.,   2.,  -3.,  -4.,   5.,  -6.,   
                                                7.,   8.,  -9.,  10., -11., -12.,  13.,
                                                -14.,  15.]))<tol).all())
    
    print('[+] TEST grid={} mpigrid={}: LatticeReal EO passed'.format('x'.join([str(a)
          for a in grid]), 'x'.join([str(a) for a in [1, 1]])))

test_realfield()