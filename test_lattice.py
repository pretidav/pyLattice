from lattice.grid import *
tol = 10e-6

def test_moveforward():
    Grid = LatticeBase(size=[4,4])
    Grid.moveforward(mu=1)
    Grid.moveforward(mu=1)
    assert(np.sum(Grid.coor-np.array([0,2]))<tol)

def test_movebackward():
    Grid = LatticeBase(size=[4,4])
    Grid.movebackward(mu=0)
    assert(np.sum(Grid.coor-np.array([3,0]))<tol)

def test_idx():
    Grid = LatticeBase(size=[4,4])
    Grid.moveforward(mu=1)
    Grid.moveforward(mu=0)
    assert(Grid.get_idx(Grid.coor)==5)

def test_realfield():
    Grid = LatticeBase(size=[4,4])
    RealField = LatticeReal(lattice=Grid)
    RealField.fill_grid(3)
    assert(np.sum(RealField.value-np.array([3]*4*4))<tol)

def test_complexfield():
    Grid = LatticeBase(size=[4,4])
    ComplexField = LatticeComplex(lattice=Grid)
    ComplexField.fill_grid(1j)
    assert(np.sum(ComplexField.value-np.array([1j]*4*4))<tol)

def test_realmatrixfield():
    Grid = LatticeBase(size=[4,4])
    RealMatrixField  = LatticeRealMatrix(lattice=Grid,N=2)
    RealMatrixField2 = LatticeRealMatrix(lattice=Grid,N=2)
    
    r = RealMatrix(value=np.array([[1,2],[3,4]]))
    q = RealMatrix(value=np.array([[5,6],[7,8]]))
    RealMatrixField.fill_grid(n=r)
    RealMatrixField2.fill_grid(n=q)
    
    assert(np.sum([np.sum(a-np.array([[1,3],[2,4]])) for a in RealMatrixField.transpose().value])<tol)
    assert(np.sum([(a-5).value for a in RealMatrixField.trace().grid])<tol)
    assert(np.sum([(a+2).value for a in RealMatrixField.det().grid])<tol)
    assert(np.sum([np.sum((a*r).value - 2) for a in RealMatrixField.inv().grid])<tol)
    assert(np.sum((RealMatrixField+q).value - np.array([[6,8],[10,12]]))<tol)
    assert(np.sum((RealMatrixField+RealMatrixField2).value - np.array([[6,8],[10,12]]))<tol)
    assert(np.sum((RealMatrixField-RealMatrixField2).value - np.array([[-4,-4],[-4,-4]]))<tol)
    assert(np.sum((RealMatrixField*RealMatrixField.inv()).value-2)<tol)
    assert(np.sum((RealMatrixField.inv()*r).value-2)<tol)
    
def test_complexmatrixfield():
    Grid = LatticeBase(size=[4,4])
    ComplexMatrixField  = LatticeComplexMatrix(lattice=Grid,N=2)
    ComplexMatrixField2 = LatticeComplexMatrix(lattice=Grid,N=2)
    
    r = ComplexMatrix(value=np.array([[1,2j],[3j,4]]))
    q = ComplexMatrix(value=np.array([[5j,6],[7,8j]]))
    ComplexMatrixField.fill_grid(n=r)
    ComplexMatrixField2.fill_grid(n=q)
    
    assert(np.sum([np.sum(a-np.array([[1,3j],[2j,4]])) for a in ComplexMatrixField.transpose().value])<tol)
    assert(np.sum([(a-5).value for a in ComplexMatrixField.trace().grid])<tol)
    assert(np.sum([(a-10).value for a in ComplexMatrixField.det().grid])<tol)
    assert(np.sum([np.sum((a*r).value - 2) for a in ComplexMatrixField.inv().grid])<tol)
    assert(np.sum((ComplexMatrixField+q).value - np.array([[1+5j,2j+6],[7+3j,4+8j]]))<tol)
    assert(np.sum((ComplexMatrixField+ComplexMatrixField2).value - np.array([[1+5j,2j+6],[7+3j,4+8j]]))<tol)
    assert(np.sum((ComplexMatrixField-ComplexMatrixField2).value - np.array([[1-5j,-6+2j],[7-3j,4-8j]]))<tol)
    assert(np.sum((ComplexMatrixField*ComplexMatrixField.inv()).value-2)<tol)
    assert(np.sum((ComplexMatrixField.inv()*r).value-2)<tol)
#add tests for adj, conj ,im , re 