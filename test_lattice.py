from lattice.grid import LatticeReal, LatticeComplex, LatticeRealMatrix, LatticeComplexMatrix, LatticeVectorReal, LatticeVectorComplex, LatticeVectorRealMatrix, LatticeVectorComplexMatrix
from linalg.tensors import RealMatrix, ComplexMatrix, VectorReal, VectorComplex, VectorRealMatrix, VectorComplexMatrix
from mpi.parallel import CartesianComm
import numpy as np 

mpigrid = [1,1]
grid = [3,3]
tol = 10e-6
CC = CartesianComm(mpigrid=mpigrid)

def test_realfield():
    Lattice = LatticeReal(grid=grid,cartesiancomm=CC)  
    Lattice.fill_value(n=np.array([1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],dtype='float32'))
    assert( (np.abs(Lattice.value - np.array([1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9]))<tol).all())
    Lattice.movebackward(mu=0)
    assert((np.abs(Lattice.value - np.array([1.4,1.5,1.6,1.7,1.8,1.9,1.1,1.2,1.3]))<tol).all())
    Lattice.moveforward(mu=0)
    assert( (np.abs(Lattice.value - np.array([1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9]))<tol).all())
    assert( np.abs((Lattice.reducesum() - 13.5))<tol)
    assert( np.abs((Lattice.average() - 1.5))<tol)
    Lattice.fill_value(2)
    Lattice2 = LatticeReal(grid=grid,cartesiancomm=CC)
    Lattice2.fill_value(3)
    assert( (np.abs(((Lattice + Lattice2).value)-np.array([5., 5., 5., 5., 5., 5., 5., 5., 5.]))<tol).all())
    assert( (np.abs(((Lattice2 - Lattice).value)-np.array([1., 1., 1., 1., 1., 1., 1., 1., 1.]))<tol).all())
    assert( (np.abs(((Lattice*Lattice2).value)-np.array([6., 6., 6., 6., 6., 6., 6., 6., 6.]))<tol).all())
    print('[+] TEST grid={} mpigrid={}: LatticeReal passed'.format('x'.join([str(a) for a in grid]),'x'.join([str(a) for a in [1,1]])))

def test_complexfield():
    Lattice = LatticeComplex(grid=grid,cartesiancomm=CC)
    Lattice.fill_value(n=np.array([1.1,1.2j,1.3,1.4j,1.5,1.6j,1.7,1.8j,1.9],dtype='complex64'))
    assert( (np.abs(Lattice.value - np.array([1.1,1.2j,1.3,1.4j,1.5,1.6j,1.7,1.8j,1.9]))<tol).all())
    Lattice.movebackward(mu=0)
    assert( (np.abs(Lattice.value - np.array([1.4j,1.5,1.6j,1.7,1.8j,1.9,1.1,1.2j,1.3]))<tol).all())
    Lattice.moveforward(mu=0)
    assert( (np.abs(Lattice.value - np.array([1.1,1.2j,1.3,1.4j,1.5,1.6j,1.7,1.8j,1.9]))<tol).all())
  
    ComplexField = LatticeComplex(grid=grid,cartesiancomm=CC)
    ComplexField.fill_value(1j)
    ComplexField2 = LatticeComplex(grid=grid, cartesiancomm=CC)
    ComplexField2.fill_value(3)
    assert( (np.abs(ComplexField.value-np.array([1j]*3*3))<tol).all())
    assert( (np.abs((ComplexField*ComplexField2).value-3j)<tol).all())
    assert( (np.abs((ComplexField+ComplexField2).value-np.array([3+1j]*3*3))<tol).all())
    assert( (np.abs((ComplexField-ComplexField2).value-np.array([-3+1j]*3*3))<tol).all())
    assert( np.abs(ComplexField.reducesum() - 9j)<tol)
    assert( np.abs(ComplexField.average() - 1j)<tol)
    print('[+] TEST grid={} mpigrid={}: LatticeComplex passed'.format('x'.join([str(a) for a in grid]),'x'.join([str(a) for a in [1,1]])))
    
def test_realmatrixfield():
    Lattice = LatticeRealMatrix(grid=grid,cartesiancomm=CC, N=2)
    test = np.array([ [[1,2],[3,4]],     [[5,6],[7,8]],     [[9,10],[11,12]], 
                      [[13,14],[15,16]], [[17,18],[19,20]], [[21,22],[23,24]],
                      [[25,26],[27,28]], [[29,30],[31,32]], [[33,34],[35,36]]], dtype='float32')
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
    print('[+] TEST grid={} mpigrid={}: LatticeRealMatrix passed'.format('x'.join([str(a) for a in grid]),'x'.join([str(a) for a in [1,1]])))

def test_complexmatrixfield():
    Lattice = LatticeComplexMatrix(grid=grid,cartesiancomm=CC, N=2)
    test = np.array([ [[1,2j],[3,4]],     [[5,6j],[7,8]],     [[9,10j],[11,12]], 
                      [[13,14j],[15j,16]], [[17,18j],[19,20j]], [[21,22j],[23,24]],
                      [[25,26j],[27,28]], [[29,30j],[31,32]], [[33,34j],[35,36]]], dtype='complex64')
    Lattice.fill_value(n=test)
    assert( (np.abs(Lattice.value - test)<tol).all())
    Lattice.moveforward(mu=0)
    Lattice.movebackward(mu=0)
    assert( (np.abs(Lattice.value - test)<tol).all())

    ComplexMatrixField  = LatticeComplexMatrix(grid=grid, cartesiancomm=CC, N=2)
    ComplexMatrixField2  = LatticeComplexMatrix(grid=grid, cartesiancomm=CC, N=2)
    
    r = ComplexMatrix(value=np.array([[1,2j],[3j,4]]))
    q = ComplexMatrix(value=np.array([[5j,6],[7,8j]]))
    ComplexMatrixField.fill_value(n=r)
    ComplexMatrixField2.fill_value(n=q)
    
    assert( (np.abs([np.sum(a-np.array([[1,3j],[2j,4]])) for a in ComplexMatrixField.transpose().value])<tol).all())
    assert( (np.abs((ComplexMatrixField.trace()-5).value)<tol).all())
    assert( (np.abs((ComplexMatrixField.det()-10).value)<tol).all())
    assert( (np.abs((ComplexMatrixField+q).value - np.array([[1+5j,2j+6],[7+3j,4+8j]]))<tol).all())
    assert( (np.abs((ComplexMatrixField+ComplexMatrixField2).value - np.array([[1+5j,2j+6],[7+3j,4+8j]]))<tol).all())
    assert( (np.abs((ComplexMatrixField-ComplexMatrixField2).value - np.array([[1-5j,-6+2j],[-7+3j,4-8j]]))<tol).all())
    assert( (np.abs((ComplexMatrixField*ComplexMatrixField.inv()).value-np.eye(2))<tol).all())
    assert( (np.abs((ComplexMatrixField.inv()*r).value-np.eye(2))<tol).all())
    assert( (np.abs([np.sum(a-np.array([[0,2],[3,0]])) for a in ComplexMatrixField.im().value])<tol).all())
    assert( (np.abs([np.sum(a-np.array([[1,0],[0,4]])) for a in ComplexMatrixField.re().value])<tol).all())
    assert( (np.abs([np.sum(a-np.array([[1,-2j],[-3j,4]])) for a in ComplexMatrixField.conj().value])<tol).all())
    assert( (np.abs([np.sum(a-np.array([[1,-3j],[-2j,4]])) for a in ComplexMatrixField.conj().value])<tol).all())
    print('[+] TEST grid={} mpigrid={}: LatticeComplexMatrix passed'.format('x'.join([str(a) for a in grid]),'x'.join([str(a) for a in [1,1]])))

def test_realvectorfield():
    Lattice = LatticeVectorReal(grid=grid,cartesiancomm=CC, Nd=3)
    test = np.array([1,2,3], dtype='float32')
    Lattice.fill_value(n=test)
    assert( (np.abs(Lattice.value - test)<tol).all())
    Lattice.moveforward(mu=0)
    Lattice.movebackward(mu=0)
    assert( (np.abs(Lattice.value - test)<tol).all())

    RealVectorField = LatticeVectorReal(grid=grid,cartesiancomm=CC,Nd=3)
    RealVectorField.fill_value(VectorReal(value=np.array([1,2,3])))
    RealVectorField2 = LatticeVectorReal(grid=grid,cartesiancomm=CC,Nd=3)
    RealVectorField2.fill_value(VectorReal(value=np.array([1,0,3])))

    assert(np.sum(RealVectorField[0]-np.array([1,2,3])) <tol)
    assert(np.sum((RealVectorField+RealVectorField2)[0] -np.array([2,2,6])) <tol)
    assert(np.sum((RealVectorField-RealVectorField2)[0] -np.array([0,2,0])) <tol)
    print('[+] TEST grid={} mpigrid={}: LatticeVectorReal passed'.format('x'.join([str(a) for a in grid]),'x'.join([str(a) for a in [1,1]])))
    

def test_complexvectorfield():
    Lattice = LatticeVectorComplex(grid=grid,cartesiancomm=CC, Nd=3)
    test = np.array([1,2j,3], dtype='complex64')
    Lattice.fill_value(n=test)
    assert( (np.abs(Lattice.value - test)<tol).all())
    Lattice.moveforward(mu=0)
    Lattice.movebackward(mu=0)
    assert( (np.abs(Lattice.value - test)<tol).all())

    VectorField = LatticeVectorComplex(grid=grid,cartesiancomm=CC,Nd=3)
    VectorField.fill_value(VectorComplex(value=np.array([1,2j,3])))
    VectorField2 = LatticeVectorComplex(grid=grid,cartesiancomm=CC,Nd=3)
    VectorField2.fill_value(VectorComplex(value=np.array([1,0,3j])))

    assert(np.sum(VectorField[0]-np.array([1,2j,3])) <tol)
    assert(np.sum((VectorField+VectorField2)[0] -np.array([2,2j,3+3j])) <tol)
    assert(np.sum((VectorField-VectorField2)[0] -np.array([0,2j,3-3j])) <tol)
    print('[+] TEST grid={} mpigrid={}: LatticeVectorComplex passed'.format('x'.join([str(a) for a in grid]),'x'.join([str(a) for a in [1,1]])))
    
def test_realvectormatrixfield():
    Lattice = LatticeVectorRealMatrix(grid=grid,cartesiancomm=CC, Nd=3, N=2)
    test = np.array([1,2,3], dtype='float32')
    Lattice.fill_value(n=test)
    assert( (np.abs(Lattice.value - test)<tol).all())
    Lattice.moveforward(mu=0)
    Lattice.movebackward(mu=0)
    assert( (np.abs(Lattice.value - test)<tol).all())

    RealVectorField = LatticeVectorReal(grid=grid,cartesiancomm=CC,Nd=3)
    RealVectorField.fill_value(VectorReal(Nd=3,value=np.array([1,2,3])))
    RealVectorField2 = LatticeVectorReal(grid=grid,cartesiancomm=CC,Nd=3)
    RealVectorField2.fill_value(VectorReal(Nd=3,value=np.array([1,0,3])))

    assert(np.sum(RealVectorField[0]-np.array([1,2,3])) <tol)
    assert(np.sum((RealVectorField+RealVectorField2)[0] -np.array([2,2,6])) <tol)
    assert(np.sum((RealVectorField-RealVectorField2)[0] -np.array([0,2,0])) <tol)
    print('[+] TEST grid={} mpigrid={}: LatticeVectorReal passed'.format('x'.join([str(a) for a in grid]),'x'.join([str(a) for a in [1,1]])))
    
def test_complexvectormatrixfield():
    Lattice = LatticeVectorReal(grid=grid,cartesiancomm=CC, Nd=3)
    test = np.array([1,2,3], dtype='float32')
    Lattice.fill_value(n=test)
    assert( (np.abs(Lattice.value - test)<tol).all())
    Lattice.moveforward(mu=0)
    Lattice.movebackward(mu=0)
    assert( (np.abs(Lattice.value - test)<tol).all())

    RealVectorField = LatticeVectorReal(grid=grid,cartesiancomm=CC,Nd=3)
    RealVectorField.fill_value(VectorReal(Nd=3,value=np.array([1,2,3])))
    RealVectorField2 = LatticeVectorReal(grid=grid,cartesiancomm=CC,Nd=3)
    RealVectorField2.fill_value(VectorReal(Nd=3,value=np.array([1,0,3])))

    assert(np.sum(RealVectorField[0]-np.array([1,2,3])) <tol)
    assert(np.sum((RealVectorField+RealVectorField2)[0] -np.array([2,2,6])) <tol)
    assert(np.sum((RealVectorField-RealVectorField2)[0] -np.array([0,2,0])) <tol)
    print('[+] TEST grid={} mpigrid={}: LatticeVectorReal passed'.format('x'.join([str(a) for a in grid]),'x'.join([str(a) for a in [1,1]])))
    
test_realfield()
test_complexfield()
test_realmatrixfield()
test_complexmatrixfield()
test_realvectorfield()
test_complexvectorfield()
#test_realvectormatrixfield()  #TBD
#test_complexvectormatrixfield() #TBD