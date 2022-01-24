from linalg.tensors import *

#basic definitions
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
def test_real():
    assert( (a+b).value  ==  [7.])
    assert( (a-b).value  ==  13)
    assert( (a*b).value  ==  -30)
    assert( (a*c).value  ==  100)
    assert( (a**c).value ==  10000000000)
    assert( (a**b).value ==  0.001)
    assert((a*bc).value  == 20-80j)

#Complex tests
def test_complex():
    assert((ac+bc).value == 5-3j)
    assert((ac-bc).value == 1+13j)
    assert((ac*bc).value == 46-14j)
    assert((ac*a).value  == 30+50j)
    assert(ac.conj().value == 3-5j)

#RealMatrix tests
def test_realmatrix():
    assert( (np.abs((A+B).value-np.array([[2,2],[3,6]]))<tol).all())
    assert( (np.abs((A-B).value-np.array([[-2,0],[3,2]]))<tol).all())
    assert( (np.abs((A*B).value-np.array([[0,2],[6,11]]))<tol).all())
    assert( B.trace().value == 4)
    assert( B.det().value   == 4)
    assert( (np.abs((B.inv()*B).value)-3.0 < tol).all() )

#ComplexMatrix tests
def test_complexmatrix():
    assert( (np.abs(Ac.im().value-np.array([[0,1],[3,0]]))<tol).all())  
    assert( (np.abs(Ac.re().value-np.array([[0,0],[0,4]]))<tol).all())  
    assert( (np.abs((Ac+Bc).value-np.array([[2,1+8j],[3j,6+9j]]))<tol).all())
    assert( (np.abs((Ac-Bc).value-np.array([[-2,-1-6j],[3j,2-9j]]))<tol).all())
    assert( (np.abs((Ac*Bc).value-np.array([[0,-9+2j],[6j,-13+39j]]))<tol).all())
    assert( (np.abs((Ac*B).value-np.array([[0,2j],[6j,8+3j]]))<tol).all())
    assert( (np.abs((A*Bc).value-np.array([[0,2+9j],[6,11+57j]]))<tol).all())
    assert(Bc.trace().value - 4+9j < tol)
    assert(Bc.det().value - 4+18j < tol)
    assert( (np.abs((Bc.inv()*Bc).value- np.eye(2)) < tol).all()) 
    assert( (np.abs(Bc.adj().value - np.array([[2,0],[1-7j,2-9j]])) < tol).all()) 

def test_vectorreal():
    v = VectorReal(value=np.array([2,3]))
    w = VectorReal(value=np.array([5,7]))
    M = RealMatrix(value=np.array([[0,1],[3,4]]))
    assert( (np.abs((v+w).value-np.array([7,10]))<tol).all())    
    assert( (np.abs((v-w).value-np.array([-3,-4]))<tol).all())    
    assert( (np.abs((v*w).value-np.array([10,21]))<tol).all())    
    assert( (np.abs((M*v).value-np.array([3,18]))<tol).all())    
    assert(((v.dot(w)).value-31)<tol)    
    
def test_vectorcomplex():
    v = VectorComplex(value=np.array([2,3j]))
    w = VectorComplex(value=np.array([5j,7]))
    M = ComplexMatrix(value=np.array([[0,1j],[3,4]]))
    assert( (np.abs((v+w).value-np.array([2+5j,7+3j]))<tol).all())    
    assert( (np.abs((v-w).value-np.array([2-5j,3j-7]))<tol).all())    
    assert( (np.abs((v*w).value-np.array([10j,21j]))<tol).all())    
    assert( (np.abs((M*v).value-np.array([-3,6+12j]))<tol).all())    
    assert(((v.dot(w)).value-31j)<tol)    

test_vectorcomplex()
def test_vectorrealmatrix(): 
    A = VectorRealMatrix(value=np.array([[[1,2],[3,4]],[[5,6],[7,8]]],dtype='float32'))
    B = VectorRealMatrix(value=np.array([[[1,-2],[-3,4]],[[5,-6],[7,8]]],dtype='float32'))
    assert( (np.abs((A+B).value-np.array([[[2,0],[0,8]],[[10,0],[14,16]]],dtype='float32'))<tol ).all()) 
    assert( (np.abs((A-B).value-np.array([[[0,4],[6,0]],[[0,12],[0,0]]],dtype='float32'))<tol ).all()) 
    assert( (np.abs((A*B).value-np.array([[[-5,6],[-9,10]],[[67,18],[91,22]]],dtype='float32'))<tol ).all()) 
    assert( (np.abs((A.transpose()).value - np.array([[[1,3],[2,4]],[[5,7],[6,8]]]))<tol ).all())   
    assert( (np.abs((A.det()).value - np.array([-2,-2]))<tol ).all())   
    assert( (np.abs((A.trace()).value - np.array([5,13]))<tol ).all())   
    assert( (np.abs((A.inv()).value - np.array([[[-2,1],[1.5,-0.5]],[[-4,3],[3.5,-2.5]]]))<tol ).all())   

def test_vectorcomplexmatrix(): 
    A = VectorComplexMatrix(value=np.array([[[1,2j],[3j,4]],[[5j,6],[7,8j]]],dtype='complex64'))
    B = VectorComplexMatrix(value=np.array([[[1j,-2],[-3,4j]],[[5j,-6],[7j,8]]],dtype='complex64'))
    assert( (np.abs((A+B).value-np.array([[[1+1j,-2+2j],[-3+3j,4+4j]],[[10j,0],[7+7j,8+8j]]],dtype='complex64'))<tol ).all()) 
    assert( (np.abs((A-B).value-np.array([[[1-1j,2+2j],[3+3j,4-4j]],[[0,12],[7-7j,-8+8j]]],dtype='complex64'))<tol ).all()) 
    assert( (np.abs((A*B).value-np.array([[[-5j,-10],[-15,10j]],[[-25+42j,48-30j],[-56+35j,-42+64j]]],dtype='complex64'))<tol ).all()) 
    assert( (np.abs((A.transpose()).value - np.array([[[1,3j],[2j,4]],[[5j,7],[6,8j]]]))<tol ).all())       
    assert( (np.abs((A.det()).value - np.array([10,-82]))<tol ).all())   
    assert( (np.abs((A.trace()).value - np.array([5,13j]))<tol ).all())   
    assert( (np.abs((A.inv()).value - np.array([[[0.40000001,-0.2j],[-0.30000001j,0.1]],[[-0.09756097j, 0.07317073],[0.08536585,-0.06097561j]]]))<tol ).all())   
    assert( (np.abs((A.conj()).value-np.array([[[1,-2j],[-3j,4]],[[-5j,6],[7,-8j]]],dtype='complex64'))<tol ).all())   
    assert( (np.abs((A.adj()).value-np.array([[[1,-3j],[-2j,4]],[[-5j,7],[6,-8j]]],dtype='complex64'))<tol ).all()) 
    assert( (np.abs((A.re()).value-np.array([[[1,0],[0,4]],[[0,6],[7,0]]],dtype='float32'))<tol ).all()) 
    assert( (np.abs((A.im()).value-np.array([[[0,2],[3,0]],[[5,0],[0,8]]],dtype='float32'))<tol ).all()) 
    