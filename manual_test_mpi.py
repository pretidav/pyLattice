from mpi.parallel import CartesianComm, pprint
from lattice.grid import LatticeReal, LatticeComplex, LatticeRealMatrix
from linalg.tensors import RealMatrix
import numpy as np



def test_LatticeReal2D(grid,comm):
    tol = 10e-6
    Lattice = LatticeReal(grid=grid,cartesiancomm=comm)
    val = np.array([1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],dtype='float32') + comm.rank
    Lattice.fill_value(n=val)
    test1= Lattice.value - np.array([1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9]) - comm.rank
    assert( (np.abs(test1)<tol).all())
    Lattice.movebackward(mu=0)
    Lattice.moveforward(mu=0)
    assert( (np.abs(test1)<tol).all())
    Lattice.fill_value(n=np.array([1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],dtype='float32'))
    assert( (np.abs(Lattice.reducesum() - 13.5*np.prod(comm.mpigrid))<tol) ) 
    assert( (np.abs(Lattice.average() - 1.5)<tol) ) 
    Lattice.fill_value(2)
    Lattice2 = LatticeReal(grid=grid,cartesiancomm=comm)
    Lattice2.fill_value(3)
    assert( (np.abs(((Lattice + Lattice2).value)-np.array([5., 5., 5., 5., 5., 5., 5., 5., 5.]))<tol).all())    
    assert( (np.abs(((Lattice2 - Lattice).value)-np.array([1., 1., 1., 1., 1., 1., 1., 1., 1.]))<tol).all())
    assert( (np.abs(((Lattice*Lattice2).value)-np.array([6., 6., 6., 6., 6., 6., 6., 6., 6.]))<tol).all())
    pprint(comm=comm.comm,msg='[+] TEST grid={} mpigrid={}: LatticeReal passed'.format('x'.join([str(a) for a in grid]),'x'.join([str(a) for a in CC.mpigrid])))

def test_LatticeComplex2D(grid,comm):
    tol = 10e-6
    Lattice = LatticeComplex(grid=grid,cartesiancomm=comm)
    val = np.array([1.1,1.2j,1.3,1.4j,1.5,1.6j,1.7,1.8j,1.9],dtype='complex64') + comm.rank
    Lattice.fill_value(n=val)
    test1= Lattice.value - np.array([1.1,1.2j,1.3,1.4j,1.5,1.6j,1.7,1.8j,1.9]) - comm.rank
    assert( (np.abs(test1)<tol).all())
    Lattice.movebackward(mu=0)
    Lattice.moveforward(mu=0)
    assert( (np.abs(test1)<tol).all())
    ComplexField = LatticeComplex(grid=grid,cartesiancomm=comm)
    ComplexField.fill_value(1j)
    ComplexField2 = LatticeComplex(grid=grid, cartesiancomm=comm)
    ComplexField2.fill_value(3)
    assert( (np.abs(ComplexField.value-np.array([1j]*3*3))<tol).all())
    assert( (np.abs((ComplexField*ComplexField2).value-3j)<tol).all())
    assert( (np.abs((ComplexField+ComplexField2).value-np.array([3+1j]*3*3))<tol).all())
    assert( (np.abs((ComplexField-ComplexField2).value-np.array([-3+1j]*3*3))<tol).all())
    assert( np.abs(ComplexField.reducesum() - 9j*np.prod(comm.mpigrid))<tol)
    assert( np.abs(ComplexField.average() - 1j)<tol)
    pprint(comm=comm.comm,msg='[+] TEST grid={} mpigrid={}: LatticeComplex passed'.format('x'.join([str(a) for a in grid]),'x'.join([str(a) for a in CC.mpigrid])))

def test_LatticeRealMatrix2D(grid,comm): 
    tol = 10e-6
    Lattice = LatticeRealMatrix(grid=grid,cartesiancomm=comm, N=2)
    test = np.array([ [[1,2],[3,4]],     [[5,6],[7,8]],     [[9,10],[11,12]], 
                      [[13,14],[15,16]], [[17,18],[19,20]], [[21,22],[23,24]],
                      [[25,26],[27,28]], [[29,30],[31,32]], [[33,34],[35,36]]], dtype='float32') + comm.rank*100
    Lattice.fill_value(n=test)
    assert( (np.abs(Lattice.value - test)<tol).all())
    Lattice.moveforward(mu=0)
    Lattice.movebackward(mu=0)
    assert( (np.abs(Lattice.value - test)<tol).all())

    RealMatrixField = LatticeRealMatrix(grid=grid, cartesiancomm=CC, N=2)
    RealMatrixField2 = LatticeRealMatrix(grid=grid, cartesiancomm=CC, N=2)
    r = RealMatrix(value=np.array([[1,2],[3,4]],dtype='float32'))
    q = RealMatrix(value=np.array([[5,6],[7,8]],dtype='float32'))
    RealMatrixField.fill_value(n=r)
    RealMatrixField2.fill_value(n=q)
    assert( (np.abs((RealMatrixField.trace()-5).value)<tol).all())
    assert( (np.abs((RealMatrixField.transpose()-np.array([[1,3],[2,4]])).value)<tol).all())
    assert( (np.abs((RealMatrixField.det()+2).value)<tol).all())
    assert( (np.abs((RealMatrixField+q).value - np.array([[6,8],[10,12]]))<tol).all())
    assert( (np.abs((RealMatrixField+RealMatrixField2).value - np.array([[6,8],[10,12]]))<tol).all())
    assert( (np.abs((RealMatrixField-RealMatrixField2).value - np.array([[-4,-4],[-4,-4]]))<tol).all())
    assert( (np.abs((RealMatrixField*RealMatrixField.inv()).value - np.eye(2))<tol).all())
    assert( (np.abs((RealMatrixField.inv()*r).value - np.eye(2))<tol).all())
    assert( np.abs(RealMatrixField.reducesum() - 10*np.prod(comm.mpigrid)*np.prod(grid))<tol)
    assert( np.abs(RealMatrixField.average() - 10)<tol)
    pprint(comm=comm.comm,msg='[+] TEST grid={} mpigrid={}: LatticeRealMatrix passed'.format('x'.join([str(a) for a in grid]),'x'.join([str(a) for a in CC.mpigrid])))

def test_LatticeReal3D(grid,comm):
    tol = 10e-6
    Lattice = LatticeReal(grid=grid,cartesiancomm=comm)
    val = np.array([a for a in range(3*3*3)],dtype='float32') + comm.rank*100
    Lattice.fill_value(n=val)
    test1= Lattice.value - np.array([a for a in range(3*3*3)],dtype='float32') - comm.rank*100
    assert( (np.abs(test1)<tol).all())
    Lattice.movebackward(mu=0)
    Lattice.moveforward(mu=0)
    assert( (np.abs(test1)<tol).all())
    Lattice.movebackward(mu=1)
    Lattice.moveforward(mu=1)
    assert( (np.abs(test1)<tol).all())
    Lattice.movebackward(mu=2)
    Lattice.moveforward(mu=2)
    assert( (np.abs(test1)<tol).all())
    pprint(comm=comm.comm,msg='[+] TEST grid={} mpigrid={}: LatticeReal passed'.format('x'.join([str(a) for a in grid]),'x'.join([str(a) for a in CC.mpigrid])))

def test_LatticeComplex3D(grid,comm):
    tol = 10e-6
    Lattice = LatticeComplex(grid=grid,cartesiancomm=comm)
    val = np.array([complex(a) for a in range(3*3*3)],dtype='complex64') + comm.rank
    Lattice.fill_value(n=val)
    test1= Lattice.value - np.array([complex(a) for a in range(3*3*3)],dtype='complex64') - comm.rank
    assert( (np.abs(test1)<tol).all())
    Lattice.movebackward(mu=0)
    Lattice.moveforward(mu=0)
    assert( (np.abs(test1)<tol).all())
    Lattice.movebackward(mu=1)
    Lattice.moveforward(mu=1)
    assert( (np.abs(test1)<tol).all())

    pprint(comm=comm.comm,msg='[+] TEST grid={} mpigrid={}: LatticeComplex passed'.format('x'.join([str(a) for a in grid]),'x'.join([str(a) for a in CC.mpigrid])))


if __name__ == '__main__':
    CC = CartesianComm(mpigrid=[2,1])
    test_LatticeReal2D(grid=[3,3],comm=CC)
    test_LatticeComplex2D(grid=[3,3],comm=CC)
    test_LatticeRealMatrix2D(grid=[3,3],comm=CC)

    CC = CartesianComm(mpigrid=[2,1,1])
    test_LatticeReal3D(grid=[3,3,3],comm=CC)
    test_LatticeComplex3D(grid=[3,3,3],comm=CC)
