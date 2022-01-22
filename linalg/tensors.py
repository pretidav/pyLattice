import numpy as np


class Real():
    def __init__(self,value: float = 0):
        self.value = np.array([value],dtype=float)

    def __add__(self,rhs):
        out = Real()
        if isinstance(rhs, Real):
            out.value = self.value + rhs.value
        else : 
            out.value = self.value + rhs
        return out 

    def __radd__(self,lhs):
        out = Real()
        if isinstance(lhs, Real):
            out.value = lhs.values + self.value 
        else : 
            out.value = lhs + self.value  
        return out 

    def __sub__(self,rhs):
        out = Real()
        if isinstance(rhs, Real):
            out.value = self.value - rhs.value
        else : 
            out.value = self.value - rhs
        return out 

    def __rsub__(self,lhs):
        out = Real()
        if isinstance(lhs, Real):
            out.value = lhs.value - self.value 
        else : 
            out.value = lhs - self.value 
        return out 
        
    def __mul__(self,rhs):
        out = Real()
        if isinstance(rhs, (Real,Complex,RealMatrix,ComplexMatrix)):
            out.value = self.value*rhs.value
        elif isinstance(rhs, (float,int,complex)) : 
            out.value = self.value*rhs
        return out

    def __rmul__(self,lhs):
        out = Real()
        if isinstance(lhs, (Real,Complex,RealMatrix,ComplexMatrix)):
            out.value = lhs.value*self.value
        elif isinstance(lhs, (float,int,complex)) : 
            out.value = lhs*self.value
        return out

    def __pow__(self,n):
        out = Real()
        if isinstance(n, (float,int)):
            out.value = self.value**n
        else : 
            out.value = self.value**n.value
        return out

class Complex(Real):
    def __init__(self, value: complex = 1j):
        super().__init__()
        self.value = np.array([value],dtype=complex)

    def re(self):
        out = Real()
        out.value = np.real(self.value)
        return out 

    def im(self):
        out = Real()
        out.value = np.imag(self.value)
        return out 

    def conj(self):
        out = Complex()
        out.value = np.conj(self.value)
        return out 

class RealMatrix():
    def __init__(self, N: int = None, value: np.ndarray = None):
        if N!=None:
            self.N = N 
            self.value = np.zeros((N,N), dtype=float)
        else :
            self.N = len(value)
            self.value = value

    def transpose(self):
        out = RealMatrix(self.N)
        out.value = np.transpose(self.value)
        return out 

    def trace(self):
        tr = np.trace(self.value)
        return Real(tr)
    
    def det(self):
        d = np.linalg.det(self.value)
        return Real(d)

    def inv(self):
        out = RealMatrix(self.N)
        out.value = np.linalg.inv(self.value)
        return out

    def __add__(self,rhs):
        if isinstance(rhs, RealMatrix):
            out = RealMatrix(self.N)
        elif isinstance(rhs, ComplexMatrix):
            out = ComplexMatrix(self.N)
        assert(self.value.shape==rhs.value.shape)
        out.value = self.value + rhs.value
        return out

    def __radd__(self,lhs):
        if isinstance(lhs, RealMatrix):
            out = RealMatrix(self.N)
        if isinstance(lhs, ComplexMatrix):
            out = ComplexMatrix(self.N)
        assert(self.value.shape==lhs.value.shape)
        out.value = self.value + lhs.value
        return out

    def __sub__(self,rhs):
        if isinstance(rhs, RealMatrix):
            out = RealMatrix(self.N)
        if isinstance(rhs, ComplexMatrix):
            out = ComplexMatrix(self.N)
        assert(self.value.shape==rhs.value.shape)
        out.value = self.value - rhs.value
        return out

    def __rsub__(self,lhs):
        if isinstance(lhs, RealMatrix):
            out = RealMatrix(self.N)
        if isinstance(lhs, ComplexMatrix):
            out = ComplexMatrix(self.N)
        assert(self.value.shape==lhs.value.shape)
        out.value = lhs.value - self.value 
        return out

    def __mul__(self,rhs):
        if isinstance(rhs, RealMatrix):
            out = RealMatrix(self.N)
            assert(self.value.shape[1]==rhs.value.shape[0])
            out.value = np.dot(self.value,rhs.value)
        elif isinstance(rhs, Real):
            out = RealMatrix(self.N)
            out.value = self.value*rhs.value
        elif isinstance(rhs, Complex):
            out = ComplexMatrix(self.N)
            out.value = self.value*rhs.value    
        elif isinstance(rhs, VectorComplex):
            out = VectorComplex(Nd=self.N)
            assert(self.value.shape[1]==rhs.value.shape[0])
            out.value = np.dot(self.value,rhs.value)
        elif isinstance(rhs, VectorReal):
            out = VectorReal(Nd=self.N)
            assert(self.value.shape[1]==rhs.value.shape[0])
            out.value = np.dot(self.value,rhs.value)
        return out

class Identity(RealMatrix):
    def __init__(self, N: int):
        super().__init__(N)
        self.value = np.diag([1]*self.N)

class ComplexMatrix(RealMatrix):
    def __init__(self,  N: int = None, value: np.ndarray = None):
        if N!=None: 
            self.N = N 
            self.value = np.zeros((N,N), dtype=complex)
        else:
            self.N = len(value)
            self.value = value

    def transpose(self):
        out = ComplexMatrix(self.N)
        out.value = np.transpose(self.value)
        return out 

    def conj(self):
        out = ComplexMatrix(self.N)
        out.value = np.conj(self.value) 
        return out 

    def adj(self):
        tmp = ComplexMatrix(self.N)
        tmp = self.conj()
        return tmp.transpose()

    def re(self):
        out = RealMatrix(self.N)
        out.value = np.real(self.value)
        return out

    def im(self):
        out = RealMatrix(self.N)
        out.value = np.imag(self.value)
        return out

    def trace(self):
        tr = np.trace(self.value)
        return Complex(tr)

    def det(self):
        d = np.linalg.det(self.value)
        return Complex(d)

    def inv(self):
        out = ComplexMatrix(self.N)
        out.value = np.linalg.inv(self.value)
        return out

    def __add__(self,rhs):
        out = ComplexMatrix(self.N)
        if isinstance(rhs, (RealMatrix,ComplexMatrix)):
            assert(self.value.shape==rhs.value.shape)
            out.value = self.value + rhs.value
        return out

    def __radd__(self,lhs):
        out = ComplexMatrix(self.N)
        if isinstance(lhs, (RealMatrix,ComplexMatrix)):
            assert(self.value.shape==lhs.value.shape)
            out.value = self.value + lhs.value
        return out

    def __sub__(self,rhs):
        out = ComplexMatrix(self.N)
        if isinstance(rhs, (RealMatrix,ComplexMatrix)):
            assert(self.value.shape==rhs.value.shape)
            out.value = self.value - rhs.value
        return out

    def __rsub__(self,lhs):
        out = ComplexMatrix(self.N)
        if isinstance(lhs, (RealMatrix,ComplexMatrix)):
            assert(self.value.shape==lhs.value.shape)
            out.value = lhs.value - self.value 
        return out

    def __mul__(self,rhs):
        if isinstance(rhs, RealMatrix):
            out = RealMatrix(self.N)
            assert(self.value.shape[1]==rhs.value.shape[0])
            out.value = np.dot(self.value,rhs.value)
        elif isinstance(rhs, (Complex,Real)):
            out = RealMatrix(self.N)
            out.value = self.value*rhs.value
        elif isinstance(rhs, VectorComplex):
            out = VectorComplex(Nd=self.N)
            assert(self.value.shape[1]==rhs.value.shape[0])
            out.value = np.dot(self.value,rhs.value)
        return out

class VectorReal():
    def __init__(self, Nd:int = None, value: np.ndarray = None):
        if Nd!=None: 
            self.Nd = Nd 
            self.value = np.array([0.]*self.Nd, dtype=float)
        else:
            self.Nd = len(value)
            self.value = value

    def __getitem__(self,mu:int):
        return Real(self.value[mu])

    def poke_component(self, mu: int, m):
        if isinstance(m,Real):
            self.value[mu] = m.value
        elif isinstance(m,(int,float)):
            self.value[mu] = m
    
    def __add__(self,rhs):
        out = VectorReal(Nd=self.Nd)
        if isinstance(rhs, VectorReal):
            assert(self.value.shape==rhs.value.shape)
            out.value = self.value + rhs.value
        elif isinstance(rhs, Real):
            out.value = self.value + rhs.value
        return out

    def __radd__(self,lhs):
        out = VectorReal(Nd=self.Nd)
        if isinstance(lhs, VectorReal):
            assert(self.value.shape==lhs.value.shape)
            out.value = self.value + lhs.value
        elif isinstance(lhs, Real):
            out.value = self.value + lhs.value
        return out

    def __sub__(self,rhs):
        out = VectorReal(Nd=self.Nd)
        if isinstance(rhs, VectorReal):
            assert(self.value.shape==rhs.value.shape)
            out.value = self.value - rhs.value
        elif isinstance(rhs, Real):
            out.value = self.value - rhs.value
        return out

    def __rsub__(self,lhs):
        out = VectorReal(Nd=self.Nd)
        if isinstance(lhs, VectorReal):
            assert(self.value.shape==lhs.value.shape)
            out.value = lhs.value - self.value 
        elif isinstance(lhs, Real):
            out.value = lhs.value - self.value 
        return out

    def __mul__(self,rhs):
        out = VectorReal(Nd=self.Nd)
        if isinstance(rhs, VectorReal):
            assert(self.value.shape==rhs.value.shape)
            out.value = self.value * rhs.value
        elif isinstance(rhs, Real):
            out.value = self.value * rhs.value
        return out

    def dot(self,rhs):
        out = VectorReal(Nd=self.Nd)
        if isinstance(rhs, VectorReal):
            assert(self.value.shape==rhs.value.shape)
            out.value = np.dot(self.value,rhs.value)
        elif isinstance(rhs, Real):
            out.value = self.value*rhs.value
        return out
    
    def transpose(self):
        out = VectorReal(Nd=self.Nd)
        out.value = self.value[:]
        return out


class VectorComplex():
    def __init__(self, Nd:int = None, value: np.ndarray = None):
        if Nd!=None: 
            self.Nd = Nd 
            self.value = np.array([1j]*self.Nd, dtype=complex)
        else:
            self.Nd = len(value)
            self.value = value

    def __getitem__(self,mu:int):
        return Complex(self.value[mu])

    def poke_component(self, mu: int, m):
        if isinstance(m,Complex):
            self.value[mu] = m.value
        elif isinstance(m,(int,float)):
            self.value[mu] = m
    
    def __add__(self,rhs):
        out = VectorComplex(Nd=self.Nd)
        if isinstance(rhs, VectorComplex):
            assert(self.value.shape==rhs.value.shape)
            out.value = self.value + rhs.value
        elif isinstance(rhs, (Real,Complex)):
            out.value = self.value + rhs.value
        return out

    def __radd__(self,lhs):
        out = VectorComplex(Nd=self.Nd)
        if isinstance(lhs, VectorComplex):
            assert(self.value.shape==lhs.value.shape)
            out.value = self.value + lhs.value
        elif isinstance(lhs, (Real,Complex)):
            out.value = self.value + lhs.value
        return out

    def __sub__(self,rhs):
        out = VectorComplex(Nd=self.Nd)
        if isinstance(rhs, VectorComplex):
            assert(self.value.shape==rhs.value.shape)
            out.value = self.value - rhs.value
        elif isinstance(rhs, (Real,Complex)):
            out.value = self.value - rhs.value
        return out

    def __rsub__(self,lhs):
        out = VectorComplex(Nd=self.Nd)
        if isinstance(lhs, VectorComplex):
            assert(self.value.shape==lhs.value.shape)
            out.value = lhs.value - self.value 
        elif isinstance(lhs, (Real,Complex)):
            out.value = lhs.value - self.value
        return out

    def __mul__(self,rhs):
        out = VectorComplex(Nd=self.Nd)
        if isinstance(rhs, VectorComplex):
            assert(self.value.shape==rhs.value.shape)
            out.value = self.value * rhs.value
        elif isinstance(rhs, (Real,Complex)):
            out.value = self.value * rhs.value
        return out

    def dot(self,rhs):
        out = VectorComplex(Nd=self.Nd)
        if isinstance(rhs, VectorComplex):
            assert(self.value.shape==rhs.value.shape)
            out.value = np.dot(self.value,rhs.value)
        elif isinstance(rhs, (Real,Complex)):
            out.value = self.value*rhs.value
        return out
    
    def transpose(self):
        out = VectorComplex(Nd=self.Nd)
        out.value = self.value[:]
        return out


class VectorRealMatrix():
    def __init__(self, Nd:int = None, N:int = None, value: np.ndarray = None):
        self.Nd = Nd 
        self.N  = N
        if N!= None and Nd!=None:
            self.value = np.zeros(shape=(Nd,N,N), dtype=float)
        else: 
            self.value = value
            self.Nd = value.shape[0]
            self.N  = value.shape[1]

    def __getitem__(self,mu:int):
        out = RealMatrix(N=self.N)
        out.value = self.value[mu]
        return out

    def poke_component(self, mu: int, m):
        if isinstance(m,RealMatrix):
            self.value[mu] = m.value
        elif isinstance(m,np.ndarray):
            self.value[mu] = m
    
    def __add__(self,rhs):
        out = VectorRealMatrix(Nd=self.Nd,N=self.N)
        if isinstance(rhs, VectorRealMatrix):
            assert(self.value.shape==rhs.value.shape)
            for mu in range(self.Nd):
                out.value[mu] = self.value[mu] + rhs.value[mu]
        elif isinstance(rhs, RealMatrix):
            for mu in range(self.Nd):
                out.value[mu] = self.value[mu] + rhs.value
        elif isinstance(rhs, Real):
            out.value = self.value + rhs.value
        elif isinstance(rhs, (int,float)):
            out.value = self.value + rhs    
        return out

    def __radd__(self,lhs):
        out = VectorRealMatrix(Nd=self.Nd,N=self.N)
        if isinstance(lhs, VectorRealMatrix):
            assert(self.value.shape==lhs.value.shape)
            for mu in range(self.Nd):
                out.value[mu] = self.value[mu] + lhs.value[mu]
        elif isinstance(lhs, RealMatrix):
            for mu in range(self.Nd):
                out.value[mu] = self.value[mu] + lhs.value
        elif isinstance(lhs, Real):
            out.value = self.value + lhs.value
        elif isinstance(lhs, (float,int)):
            out.value = self.value + lhs   
        return out

    def __sub__(self,rhs):
        out = VectorRealMatrix(Nd=self.Nd,N=self.N)
        if isinstance(rhs, VectorRealMatrix):
            assert(self.value.shape==rhs.value.shape)
            for mu in range(self.Nd):
                out.value[mu] = self.value[mu] - rhs.value[mu]
        elif isinstance(rhs, RealMatrix):
            for mu in range(self.Nd):
                out.value[mu] = self.value[mu] - rhs.value
        elif isinstance(rhs, Real):
            out.value = self.value - rhs.value
        elif isinstance(rhs, (int,float)):
            out.value = self.value - rhs    
        return out

    def __rsub__(self,lhs):
        out = VectorRealMatrix(Nd=self.Nd,N=self.N)
        if isinstance(lhs, VectorRealMatrix):
            assert(self.value.shape==lhs.value.shape)
            for mu in range(self.Nd):
                out.value[mu] = lhs.value[mu] - self.value[mu]
        if isinstance(lhs, RealMatrix):
            for mu in range(self.Nd):
                out.value[mu] = lhs.value - self.value[mu]
        elif isinstance(lhs, Real):
            out.value = lhs.value - self.value
        elif isinstance(lhs, (float,int)):
            out.value = lhs - self.value   
        return out

    def __mul__(self,rhs):
        out = VectorRealMatrix(Nd=self.Nd,N=self.N)
        if isinstance(rhs, VectorRealMatrix):
            assert(self.value.shape==rhs.value.shape)
            for mu in range(self.Nd):
                out.value[mu] = np.dot(self.value[mu] , rhs.value[mu])
        elif isinstance(rhs, RealMatrix):
            for mu in range(self.Nd):
                out.value[mu] = np.dot(self.value[mu] , rhs.value)
        elif isinstance(rhs, Real):
            out.value = self.value * rhs.value
        elif isinstance(rhs, (float,int)):
            out.value = self.value * rhs    
        return out

    def transpose(self):
        out = VectorRealMatrix(Nd=self.Nd,N=self.N)
        for i in range(self.Nd):
            out.value[i]=np.transpose(self.value[i,:,:])
        return out 

    def trace(self):
        out = VectorReal(Nd=self.Nd)
        for i in range(self.Nd):
            out.value[i]=np.trace(self.value[i,:,:])
        return out 
    
    def det(self):
        out = VectorReal(Nd=self.Nd)
        for i in range(self.Nd):
            out.value[i]=np.linalg.det(self.value[i,:,:])
        return out 

    def inv(self):
        out = VectorRealMatrix(Nd=self.Nd,N=self.N)
        for i in range(self.Nd):
            out.value[i]=np.linalg.inv(self.value[i,:,:])
        return out 

class VectorComplexMatrix():
    def __init__(self, Nd:int = None, N:int = None, value: np.ndarray = None):
        self.Nd = Nd 
        self.N  = N
        if Nd!=None and N!=None:
            self.value = np.zeros(shape=(Nd,N,N), dtype=complex)
        else : 
            self.value = value 
            self.Nd = value.shape[0]
            self.N = value.shape[1]

    def __getitem__(self,mu:int):
        out = ComplexMatrix(N=self.N)
        out.value = self.value[mu]
        return out

    def poke_component(self, mu: int, m):
        if isinstance(m,ComplexMatrix):
            self.value[mu] = m.value
        elif isinstance(m,np.ndarray):
            self.value[mu] = m
    
    def __add__(self,rhs):
        out = VectorComplexMatrix(Nd=self.Nd,N=self.N)
        if isinstance(rhs, (VectorComplexMatrix,VectorRealMatrix)):
            assert(self.value.shape==rhs.value.shape)
            for mu in range(self.Nd):
                out.value[mu] = self.value[mu] + rhs.value[mu]
        elif isinstance(rhs, (ComplexMatrix,RealMatrix)):
            for mu in range(self.Nd):
                out.value[mu] = self.value[mu] + rhs.value
        elif isinstance(rhs, (Complex,Real)):
            out.value = self.value + rhs.value
        elif isinstance(rhs, (int,float,complex)):
            out.value = self.value + rhs    
        return out

    def __radd__(self,lhs):
        out = VectorComplexMatrix(Nd=self.Nd,N=self.N)
        if isinstance(lhs, (VectorComplexMatrix,VectorRealMatrix)):
            assert(self.value.shape==lhs.value.shape)
            for mu in range(self.Nd):
                out.value[mu] = self.value[mu] + lhs.value[mu]
        elif isinstance(lhs, (ComplexMatrix,RealMatrix)):
            for mu in range(self.Nd):
                out.value[mu] = self.value[mu] + lhs.value
        elif isinstance(lhs, (Complex,Real)):
            out.value = self.value + lhs.value
        elif isinstance(lhs, (float,int,complex)):
            out.value = self.value + lhs   
        return out

    def __sub__(self,rhs):
        out = VectorComplexMatrix(Nd=self.Nd,N=self.N)
        if isinstance(rhs, (VectorComplexMatrix,VectorRealMatrix)):
            assert(self.value.shape==rhs.value.shape)
            for mu in range(self.Nd):
                out.value[mu] = self.value[mu] - rhs.value[mu]
        elif isinstance(rhs, (ComplexMatrix,RealMatrix)):
            for mu in range(self.Nd):
                out.value[mu] = self.value[mu] - rhs.value
        elif isinstance(rhs, (Complex,Real)):
            out.value = self.value - rhs.value
        elif isinstance(rhs, (int,float,complex)):
            out.value = self.value - rhs    
        return out

    def __rsub__(self,lhs):
        out = VectorComplexMatrix(Nd=self.Nd,N=self.N)
        if isinstance(lhs, (VectorComplexMatrix,VectorRealMatrix)):
            assert(self.value.shape==lhs.value.shape)
            for mu in range(self.Nd):
                out.value[mu] = lhs.value[mu] - self.value[mu]
        if isinstance(lhs, (ComplexMatrix,RealMatrix)):
            for mu in range(self.Nd):
                out.value[mu] = lhs.value - self.value[mu]
        elif isinstance(lhs, (Complex,Real)):
            out.value = lhs.value - self.value
        elif isinstance(lhs, (float,int,complex)):
            out.value = lhs - self.value   
        return out

    def __mul__(self,rhs):
        out = VectorComplexMatrix(Nd=self.Nd,N=self.N)
        if isinstance(rhs, (VectorComplexMatrix,VectorRealMatrix)):
            assert(self.value.shape==rhs.value.shape)
            for mu in range(self.Nd):
                out.value[mu] = np.dot(self.value[mu] , rhs.value[mu])
        elif isinstance(rhs, (ComplexMatrix,RealMatrix)):
            for mu in range(self.Nd):
                out.value[mu] = np.dot(self.value[mu] , rhs.value)
        elif isinstance(rhs, (Complex,Real)):
            out.value = self.value * rhs.value
        elif isinstance(rhs, (float,int,complex)):
            out.value = self.value * rhs    
        return out

    def transpose(self):
        out = VectorComplexMatrix(Nd=self.Nd,N=self.N)
        for i in range(self.Nd):
            out.value[i]=np.transpose(self.value[i,:,:])
        return out 

    def trace(self):
        out = VectorComplex(Nd=self.Nd)
        for i in range(self.Nd):
            out.value[i]=np.trace(self.value[i,:,:])
        return out 
    
    def det(self):
        out = VectorComplex(Nd=self.Nd)
        for i in range(self.Nd):
            out.value[i]=np.linalg.det(self.value[i,:,:])
        return out 

    def inv(self):
        out = VectorComplexMatrix(Nd=self.Nd,N=self.N)
        for i in range(self.Nd):
            out.value[i]=np.linalg.inv(self.value[i,:,:])
        return out 

    def conj(self):
        out = VectorComplexMatrix(Nd=self.Nd, N=self.N)
        out.value = np.conj(self.value) 
        return out 

    def adj(self):
        tmp = VectorComplexMatrix(Nd=self.Nd, N=self.N)
        tmp = self.conj()
        return tmp.transpose()

    def re(self):
        out = VectorRealMatrix(Nd=self.Nd, N=self.N)
        out.value = np.real(self.value)
        return out

    def im(self):
        out = VectorRealMatrix(Nd=self.Nd, N=self.N)
        out.value = np.imag(self.value)
        return out
