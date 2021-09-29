import numpy as np
import unittest

class Real():
    def __init__(self,value: float = 0):
        self.value = value

    def __add__(self,X):
        out = Real()
        if isinstance(X, Real):
            out.value = self.value + X.value
        else : 
            out.value = self.value + X
        return out 

    def __sub__(self,X):
        out = Real()
        if isinstance(X, Real):
            out.value = self.value - X.value
        else : 
            out.value = self.value - X
        return out 

    def __mul__(self,X):
        out = Real()
        if isinstance(X, (Real,Complex,RealMatrix,ComplexMatrix)):
            out.value = self.value*X.value
        elif isinstance(X, float) or isinstance(X,int) or isinstance(X,complex) : 
            out.value = self.value*X
        return out
        
    def __pow__(self,n):
        out = Real()
        if isinstance(n, (float,int)):
            out.value = self.value**n
        else : 
            out.value = self.value**n.value
        return out

    def __str__(self):
        return str(self.value)

class Complex(Real):
    def __init__(self, value: complex = 1j):
        super().__init__()
        self.value = value

    def re(self):
        out = Real()
        out.value = self.value.re
        return out 

    def im(self):
        out = Real()
        out.value = self.value.imag
        return out 

    def conj(self):
        out = Complex()
        out.value = self.value - 1j*2.0*self.value.imag
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
        out.value = self.value.transpose()
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

    def __add__(self,X):
        out = RealMatrix(self.N)
        if isinstance(X, (RealMatrix,ComplexMatrix)):
            assert(self.value.shape==X.value.shape)
            out.value = self.value + X.value
        return out

    def __sub__(self,X):
        out = RealMatrix(self.N)
        if isinstance(X, (RealMatrix,ComplexMatrix)):
            assert(self.value.shape==X.value.shape)
            out.value = self.value - X.value
        return out

    def __mul__(self,X):
        if isinstance(X, RealMatrix):
            out = RealMatrix(self.N)
            assert(self.value.shape[1]==X.value.shape[0])
            out.value = np.dot(self.value,X.value)
        elif isinstance(X, Real):
            out = RealMatrix(self.N)
            out.value = self.value*X.value
        elif isinstance(X, Complex):
            out = ComplexMatrix(self.N)
            out.value = self.value*X.value    
        elif isinstance(X, VectorComplex):
            out = VectorComplex(Nd=self.N)
            assert(self.value.shape[1]==X.value.shape[0])
            out.value = np.dot(self.value,X.value)
        elif isinstance(X, VectorReal):
            out = VectorReal(Nd=self.N)
            assert(self.value.shape[1]==X.value.shape[0])
            out.value = np.dot(self.value,X.value)
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
        out.value = self.value.transpose()
        return out 

    def conj(self):
        out = ComplexMatrix(self.N)
        out.value = self.value -1j*2.0*np.imag(self.value)
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

    def __add__(self,X):
        out = ComplexMatrix(self.N)
        if isinstance(X, (RealMatrix,ComplexMatrix)):
            assert(self.value.shape==X.value.shape)
            out.value = self.value + X.value
        return out

    def __sub__(self,X):
        out = ComplexMatrix(self.N)
        if isinstance(X, (RealMatrix,ComplexMatrix)):
            assert(self.value.shape==X.value.shape)
            out.value = self.value - X.value
        return out

    def __mul__(self,X):
        if isinstance(X, RealMatrix):
            out = RealMatrix(self.N)
            assert(self.value.shape[1]==X.value.shape[0])
            out.value = np.dot(self.value,X.value)
        elif isinstance(X, (Complex,Real)):
            out = RealMatrix(self.N)
            out.value = self.value*X.value
        elif isinstance(X, VectorComplex):
            out = VectorComplex(Nd=self.N)
            assert(self.value.shape[1]==X.value.shape[0])
            out.value = np.dot(self.value,X.value)
        return out

class VectorReal():
    def __init__(self, Nd:int = None, value: np.ndarray = None):
        if Nd!=None: 
            self.Nd = Nd 
            self.value = np.array([0.]*self.Nd)
        else:
            self.Nd = len(value)
            self.value = value

    def peek_component(self,mu):
        return self.value[mu]

    def poke_component(self, mu: int, m):
        if isinstance(m,Real):
            self.value[mu] = m.value
        elif isinstance(m,(int,float)):
            self.value[mu] = m
    
    def __add__(self,X):
        out = VectorReal(Nd=self.Nd)
        if isinstance(X, VectorReal):
            assert(self.value.shape==X.value.shape)
            out.value = self.value + X.value
        elif isinstance(X, Real):
            out.value = self.value + X.value
        return out

    def __sub__(self,X):
        out = VectorReal(Nd=self.Nd)
        if isinstance(X, VectorReal):
            assert(self.value.shape==X.value.shape)
            out.value = self.value - X.value
        elif isinstance(X, Real):
            out.value = self.value - X.value
        return out

    def __mul__(self,X):
        out = VectorReal(Nd=self.Nd)
        if isinstance(X, VectorReal):
            assert(self.value.shape==X.value.shape)
            out.value = self.value * X.value
        elif isinstance(X, Real):
            out.value = self.value * X.value
        return out

    def dot(self,X):
        out = VectorReal(Nd=self.Nd)
        if isinstance(X, VectorReal):
            assert(self.value.shape==X.value.shape)
            out.value = np.dot(self.value,X.value)
        elif isinstance(X, Real):
            out.value = self.value*X.value
        return out


class VectorComplex():
    def __init__(self, Nd:int = None, value: np.ndarray = None):
        if Nd!=None: 
            self.Nd = Nd 
            self.value = np.array([1j]*self.Nd)
        else:
            self.Nd = len(value)
            self.value = value

    def peek_component(self,mu):
        return self.value[mu]

    def poke_component(self, mu: int, m):
        if isinstance(m,Complex):
            self.value[mu] = m.value
        elif isinstance(m,(int,float)):
            self.value[mu] = m
    
    def __add__(self,X):
        out = VectorComplex(Nd=self.Nd)
        if isinstance(X, VectorComplex):
            assert(self.value.shape==X.value.shape)
            out.value = self.value + X.value
        elif isinstance(X, (Real,Complex)):
            out.value = self.value + X.value
        return out

    def __sub__(self,X):
        out = VectorComplex(Nd=self.Nd)
        if isinstance(X, VectorComplex):
            assert(self.value.shape==X.value.shape)
            out.value = self.value - X.value
        elif isinstance(X, (Real,Complex)):
            out.value = self.value - X.value
        return out

    def __mul__(self,X):
        out = VectorComplex(Nd=self.Nd)
        if isinstance(X, VectorComplex):
            assert(self.value.shape==X.value.shape)
            out.value = self.value * X.value
        elif isinstance(X, (Real,Complex)):
            out.value = self.value * X.value
        return out

    def dot(self,X):
        out = VectorComplex(Nd=self.Nd)
        if isinstance(X, VectorComplex):
            assert(self.value.shape==X.value.shape)
            out.value = np.dot(self.value,X.value)
        elif isinstance(X, (Real,Complex)):
            out.value = self.value*X.value
        return out