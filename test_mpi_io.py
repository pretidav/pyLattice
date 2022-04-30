from IO.MPI_IO import Writer, Reader
from mpi.parallel import CartesianComm
from lattice.grid import LatticeComplex, LatticeReal, LatticeComplexMatrix, LatticeRealMatrix, LatticeVectorReal, LatticeVectorComplex, LatticeVectorRealMatrix, LatticeVectorComplexMatrix
import numpy as np

tol = 10e-6
mpigrid = [2, 2]
grid = [3,3]
CC = CartesianComm(mpigrid=mpigrid)

def test_latticereal():
    Lattice = LatticeReal(grid=grid, cartesiancomm=CC)
    Lattice.fill_value(n=np.array(
        [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9], dtype='float32')+CC.rank)
    
    W = Writer(lattice=Lattice,cartesiancomm=CC)
    R = Reader(cartesiancomm=CC)
    W.write(filename='./test_file')
    Lattice2 = R.read(filename='./test_file')
    
    assert(( np.abs(Lattice.value-Lattice2.value)<tol).all()) 

def test_latticecomplex():
    Lattice = LatticeComplex(grid=grid, cartesiancomm=CC)
    Lattice.fill_value(n=np.array(
        [1.1, 1.2j, 1.3, 1.4j, 1.5, 1.6j, 1.7, 1.8j, 1.9], dtype='complex64')+CC.rank)
    
    W = Writer(lattice=Lattice,cartesiancomm=CC)
    R = Reader(cartesiancomm=CC)
    W.write(filename='./test_file')
    Lattice2 = R.read(filename='./test_file')

    assert(( np.abs(Lattice.value-Lattice2.value)<tol).all()) 

def test_latticerealmatrix():
    Lattice = LatticeRealMatrix(grid=grid, cartesiancomm=CC, N=2)
    test = np.array([[[1, 2], [3, 4]],     [[5, 6], [7, 8]],     [[9, 10], [11, 12]],
                     [[13, 14], [15, 16]], [[17, 18], [
                         19, 20]], [[21, 22], [23, 24]],
                     [[25, 26], [27, 28]], [[29, 30], [31, 32]], [[33, 34], [35, 36]]], dtype='float32')
    Lattice.fill_value(n=test)
    W = Writer(lattice=Lattice,cartesiancomm=CC)
    R = Reader(cartesiancomm=CC)
    W.write(filename='./test_file')
    Lattice2 = R.read(filename='./test_file')

    assert(( np.abs(Lattice.value-Lattice2.value)<tol).all()) 

def test_latticecomplexmatrix():
    Lattice = LatticeComplexMatrix(grid=grid, cartesiancomm=CC, N=2)
    test = np.array([[[1, 2j], [3, 4]],     [[5, 6j], [7, 8]],     [[9, 10j], [11, 12]],
                     [[13, 14j], [15j, 16]], [[17, 18j], [
                         19, 20j]], [[21, 22j], [23, 24]],
                     [[25, 26j], [27, 28]], [[29, 30j], [31, 32]], [[33, 34j], [35, 36]]], dtype='complex64')
    Lattice.fill_value(n=test)
    W = Writer(lattice=Lattice,cartesiancomm=CC)
    R = Reader(cartesiancomm=CC)
    W.write(filename='./test_file')
    Lattice2 = R.read(filename='./test_file')

    assert(( np.abs(Lattice.value-Lattice2.value)<tol).all()) 

def test_latticerealvector():
    Lattice = LatticeVectorReal(grid=grid, cartesiancomm=CC, Nd=3)
    test = np.array([1, 2, 3], dtype='float32')
    Lattice.fill_value(n=test)
    W = Writer(lattice=Lattice,cartesiancomm=CC)
    R = Reader(cartesiancomm=CC)
    W.write(filename='./test_file')
    Lattice2 = R.read(filename='./test_file')

    assert(( np.abs(Lattice.value-Lattice2.value)<tol).all()) 


def test_latticecomplexvector():
    Lattice = LatticeVectorComplex(grid=grid, cartesiancomm=CC, Nd=3)
    test = np.array([1, 2j, 3], dtype='complex64')
    Lattice.fill_value(n=test)
    W = Writer(lattice=Lattice,cartesiancomm=CC)
    R = Reader(cartesiancomm=CC)
    W.write(filename='./test_file')
    Lattice2 = R.read(filename='./test_file')

    assert(( np.abs(Lattice.value-Lattice2.value)<tol).all()) 


def test_latticerealvectormatrix():
    Lattice = LatticeVectorRealMatrix(grid=grid, cartesiancomm=CC, N=2, Nd=2)
    test = np.reshape(
        np.array([int(i) for i in range(3*3*2*2*2)], dtype='float32')+CC.rank, (9, 2, 2, 2))

    Lattice.fill_value(n=test)
    W = Writer(lattice=Lattice,cartesiancomm=CC)
    R = Reader(cartesiancomm=CC)
    W.write(filename='./test_file')
    Lattice2 = R.read(filename='./test_file')

    assert(( np.abs(Lattice.value-Lattice2.value)<tol).all()) 


def test_latticecomplexvectormatrix():
    Lattice = LatticeVectorComplexMatrix(
        grid=grid, cartesiancomm=CC, N=2, Nd=2)
    test = np.reshape(np.array([complex(i) for i in range(
        3*3*2*2*2)], dtype='complex64')+CC.rank, (9, 2, 2, 2))

    Lattice.fill_value(n=test)
    W = Writer(lattice=Lattice,cartesiancomm=CC)
    R = Reader(cartesiancomm=CC)
    W.write(filename='./test_file')
    Lattice2 = R.read(filename='./test_file')

    assert(( np.abs(Lattice.value-Lattice2.value)<tol).all()) 


if __name__=='__main__': 
    test_latticereal()
    test_latticecomplex()
    test_latticerealmatrix()
    test_latticecomplexmatrix()
    test_latticerealvector()
    test_latticecomplexvector()
    test_latticerealvectormatrix()
    test_latticecomplexvectormatrix()