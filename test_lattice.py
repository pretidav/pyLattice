from lattice.grid import *
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
    
# def test_realmatrixfield():
#     Lattice = LatticeRealMatrix(grid=grid,cartesiancomm=CC, N=2)
#     test = np.array([ [[1,2],[3,4]],     [[5,6],[7,8]],     [[9,10],[11,12]], 
#                       [[13,14],[15,16]], [[17,18],[19,20]], [[21,22],[23,24]],
#                       [[25,26],[27,28]], [[29,30],[31,32]], [[33,34],[35,36]]], dtype='float32')
#     Lattice.fill_value(n=test)
#     assert(np.sum(Lattice.value - test)<tol)
#     assert( np.sum(Lattice.tensor_idx - np.array([[0,1,2],[3,4,5],[6,7,8]]))<tol )
#     Lattice.movebackward(mu=0)
#     assert( np.sum(Lattice.tensor_idx - np.array([[3,4,5],[6,7,8],[0,1,2]]))<tol)
#     test_shift = np.array([[[13,14],[15,16]], [[17,18],[19,20]], [[21,22],[23,24]],
#                            [[25,26],[27,28]], [[29,30],[31,32]], [[33,34],[35,36]],
#                            [[1,2],[3,4]],     [[5,6],[7,8]],     [[9,10],[11,12]]], dtype='float32')

#     assert( np.sum(Lattice.value - test_shift)<tol)
#     Lattice.moveforward(mu=0)
#     assert( np.abs(np.sum(Lattice.value - test))<tol)
#     assert( np.sum(Lattice.tensor_idx - np.array([[0,1,2],[3,4,5],[6,7,8]]))<tol)

    # RealMatrixField = LatticeRealMatrix(grid=grid, cartesiancomm=CC, N=2)
    # RealMatrixField2 = LatticeRealMatrix(grid=grid, cartesiancomm=CC, N=2)
    # r = RealMatrix(value=np.array([[1,2],[3,4]],dtype='float32'))
    # q = RealMatrix(value=np.array([[5,6],[7,8]],dtype='float32'))
    # RealMatrixField.fill_value(n=r)
    # RealMatrixField2.fill_value(n=q)
    # assert(np.sum((RealMatrixField.trace()-5).value)<tol)
    # assert(np.sum((RealMatrixField.transpose()-np.array([[1,3],[2,4]])).value)<tol)
    # assert(np.sum((RealMatrixField.det()+2).value)<tol)
    # assert(np.sum((RealMatrixField+q).value - np.array([[6,8],[10,12]]))<tol)
    # assert(np.sum((RealMatrixField+RealMatrixField2).value - np.array([[6,8],[10,12]]))<tol)
    # assert(np.sum((RealMatrixField-RealMatrixField2).value - np.array([[-4,-4],[-4,-4]]))<tol)
    # assert(np.sum((RealMatrixField*RealMatrixField.inv()).value-2)<tol)
    # assert(np.sum((RealMatrixField.inv()*r).value-2)<tol)
#test_realmatrixfield() 

# # def test_complexmatrixfield():
# #     Grid = LatticeParallel(grid=[4,4],pgrid=[2,1])
# #     ComplexMatrixField  = LatticeComplexMatrix(lattice=Grid,N=2)
# #     ComplexMatrixField2 = LatticeComplexMatrix(lattice=Grid,N=2)
    
# #     r = ComplexMatrix(value=np.array([[1,2j],[3j,4]]))
# #     q = ComplexMatrix(value=np.array([[5j,6],[7,8j]]))
# #     ComplexMatrixField.fill_value(n=r)
# #     ComplexMatrixField2.fill_value(n=q)
    
# #     assert(np.sum([np.sum(a-np.array([[1,3j],[2j,4]])) for a in ComplexMatrixField.transpose().value])<tol)
# #     assert(np.sum((ComplexMatrixField.trace()-5).value)<tol)
# #     assert(np.sum((ComplexMatrixField.det()-10).value)<tol)
# #     assert(np.sum((ComplexMatrixField+q).value - np.array([[1+5j,2j+6],[7+3j,4+8j]]))<tol)
# #     assert(np.sum((ComplexMatrixField+ComplexMatrixField2).value - np.array([[1+5j,2j+6],[7+3j,4+8j]]))<tol)
# #     assert(np.sum((ComplexMatrixField-ComplexMatrixField2).value - np.array([[1-5j,-6+2j],[7-3j,4-8j]]))<tol)
# #     assert(np.sum((ComplexMatrixField*ComplexMatrixField.inv()).value-2)<tol)
# #     assert(np.sum((ComplexMatrixField.inv()*r).value-2)<tol)
# #     assert(np.sum([np.sum(a-np.array([[0,2],[3,0]])) for a in ComplexMatrixField.im().value])<tol)
# #     assert(np.sum([np.sum(a-np.array([[1,0],[0,4]])) for a in ComplexMatrixField.re().value])<tol)
# #     assert(np.sum([np.sum(a-np.array([[1,-2j],[-3j,4]])) for a in ComplexMatrixField.conj().value])<tol)
# #     assert(np.sum([np.sum(a-np.array([[1,-3j],[-2j,4]])) for a in ComplexMatrixField.conj().value])<tol)

# # def test_realvectorfield():
# #     Grid = LatticeParallel(grid=[4,4],pgrid=[2,1])
# #     Nd = 3
# #     RealVectorField = LatticeVectorReal(lattice=Grid,Nd=Nd)
# #     RealVectorField.fill_value(VectorReal(Nd,value=np.array([1,2,3])))
# #     RealVectorField2 = LatticeVectorReal(lattice=Grid,Nd=Nd)
# #     RealVectorField2.fill_value(VectorReal(Nd,value=np.array([1,2,3])))

# #     assert(np.sum(RealVectorField[0] -np.array([1,2,3])) <tol)
# #     assert(np.sum((RealVectorField+RealVectorField2)[0] -np.array([2,4,6])) <tol)
    

# # def test_complexvectorfield():
# #     Grid = LatticeParallel(grid=[4,4],pgrid=[2,1])
# #     Nd = 3
# #     ComplexVectorField = LatticeVectorComplex(lattice=Grid,Nd=Nd)
# #     ComplexVectorField.fill_value(VectorComplex(Nd,value=np.array([1j,2,3])))
# #     ComplexVectorField2 = LatticeVectorComplex(lattice=Grid,Nd=Nd)
# #     ComplexVectorField2.fill_value(VectorComplex(Nd,value=np.array([1j,2,3])))

# #     assert(np.sum(ComplexVectorField[0] -np.array([1j,2,3])) <tol)
# #     assert(np.sum((ComplexVectorField+ComplexVectorField2)[0] -np.array([2j,4,6])) <tol)

