from linalg.tensors import *
    
tol = 10e-6
a  = Real(10)
b  = Real(-3)
c  = 10 
ac = Complex(3+5j)
bc = Complex(2-8j)

A  = RealMatrix(N=2)
A.value = np.array([[0,1],[3,4]])
B  = RealMatrix(N=2)
B.value = np.array([[2,1],[0,2]])

Ac  = ComplexMatrix(N=2)
Ac.value = np.array([[0,1j],[3j,4]])
Bc  = ComplexMatrix(N=2)
Bc.value = np.array([[2,1+7j],[0,2+9j]])

U  = Link(N=2,mu=2)



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

#RealMatrix tests
def test_sum_realmatrix():
    assert(np.sum((A+B).value-np.array([[2,2],[3,6]]))<tol)
def test_diff_realmatrix():
    assert(np.sum((A-B).value-np.array([[-2,0],[3,2]]))<tol)
def test_prod_realmatrix():
    assert(np.sum((A*B).value-np.array([[0,2],[6,11]]))<tol)

#ComplexMatrix tests
def test_im_complexmatrix():
    assert(np.sum(Ac.im().value-np.array([[0,1],[3,0]]))<tol)  
def test_re_complexmatrix():
    assert(np.sum(Ac.re().value-np.array([[0,0],[0,4]]))<tol)  
