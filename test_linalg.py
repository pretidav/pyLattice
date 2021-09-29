from linalg.tensors import *
    
tol = 10e-6
a  = Real(10)
b  = Real(-3)
c  = 10 
ac = Complex(3+5j)
bc = Complex(2-8j)

A  = RealMatrix(value=np.array([[0,1],[3,4]]))
B  = RealMatrix(value=np.array([[2,1],[0,2]]))
C = RealMatrix(N=2)
C.value = np.array([[3,0],[0,6]])


Ac  = ComplexMatrix(value=np.array([[0,1j],[3j,4]],dtype=complex))
Bc  = ComplexMatrix(N=2)
Bc.value = np.array([[2,1+7j],[0,2+9j]],dtype=complex)





#Real tests 
def test_sum_real_real():
    assert( (a+b).value  ==  7)
def test_diff_real_real():
    assert( (a-b).value  ==  13)
def test_mul_real_real():
    assert( (a*b).value  ==  -30)
def test_mul_real_float():
    assert( (a*c).value  ==  100)
def test_pow_real_float():
    assert( (a**c).value ==  10000000000)
def test_pow_real_real():
    assert( (a**b).value ==  0.001)
def test_mul_real_complex():
    assert((a*bc).value  == 20-80j)

#Complex tests
def test_sum_complex_complex():
    assert((ac+bc).value == 5-3j)
def test_diff_complex_complex():
    assert((ac-bc).value == 1+13j)
def test_mul_complex_complex():
    assert((ac*bc).value == 46-14j)
def test_mul_complex_real():
    assert((ac*a).value  == 30+50j)
def test_conj_complex():
    assert(ac.conj().value == 3-5j)

#RealMatrix tests
def test_sum_realmatrix():
    assert(np.sum((A+B).value-np.array([[2,2],[3,6]]))<tol)
def test_diff_realmatrix():
    assert(np.sum((A-B).value-np.array([[-2,0],[3,2]]))<tol)
def test_prod_realmatrix():
    assert(np.sum((A*B).value-np.array([[0,2],[6,11]]))<tol)
def test_trace_realmatrix():
    assert(B.trace().value == 4)
def test_det_realmatrix():
    assert(B.det().value   == 4)
def test_inv_realmatrix():
    assert(np.sum((B.inv()*B).value)-3.0 < tol) 

#ComplexMatrix tests
def test_im_complexmatrix():
    assert(np.sum(Ac.im().value-np.array([[0,1],[3,0]]))<tol)  
def test_re_complexmatrix():
    assert(np.sum(Ac.re().value-np.array([[0,0],[0,4]]))<tol)  
def test_sum_complexmatrix():
    assert(np.sum((Ac+Bc).value-np.array([[2,1+8j],[3j,6+9j]]))<tol)
def test_diff_complexmatrix():
    assert(np.sum((Ac-Bc).value-np.array([[-2,-1-6j],[3j,2-9j]]))<tol)
def test_prod_complexmatrix():
    assert(np.sum((Ac*Bc).value-np.array([[0,-6],[6j,-13+9j]]))<tol)
def test_prod_complexrealmatrix():
    assert(np.sum((Ac*B).value-np.array([[0,2j],[6j,8+3j]]))<tol)
def test_prod_realcomplexmatrix():
    assert(np.sum((A*Bc).value-np.array([[0,2+9j],[6,11+57j]]))<tol)
def test_trace_complexmatrix():
    assert(Bc.trace().value - 4+9j < tol)
def test_det_complexmatrix():
    assert(Bc.det().value - 4+18j < tol)
def test_inv_complexmatrix():
    assert(np.sum((Bc.inv()*Bc).value)-3.0 < tol) 
def test_adj_complexmatrix():
    assert(np.sum(Bc.adj().value - np.array([[2,0],[1-7j,2-9j]])) < tol) 

def test_vectorreal():
    v = VectorReal(value=np.array([2,3]))
    w = VectorReal(value=np.array([5,7]))
    M = RealMatrix(value=np.array([[0,1],[3,4]]))
    assert(np.sum((v+w).value-np.array([7,10]))<tol)    
    assert(np.sum((v-w).value-np.array([-3,-4]))<tol)    
    assert(np.sum((v*w).value-np.array([10,21]))<tol)    
    assert(np.sum((M*v).value-np.array([3,18]))<tol)    
    assert(((v.dot(w)).value-31)<tol)    
    
    