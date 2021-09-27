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
        out = RealMatrix()
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
        out = RealMatrix(self.N)
        if isinstance(X, RealMatrix):
            assert(self.value.shape[1]==X.value.shape[0])
            out.value = np.dot(self.value,X.value)
        elif isinstance(X, (Complex,Real)):
            out.value = self.value*X.value
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
        out = RealMatrix(self.N)
        if isinstance(X, RealMatrix):
            assert(self.value.shape[1]==X.value.shape[0])
            out.value = np.dot(self.value,X.value)
        elif isinstance(X, (Complex,Real)):
            out.value = self.value*X.value
        return out

class Link(ComplexMatrix):
    def __init__(self,N: int, mu:int):
        super().__init__(N)
        self.value = np.array([self.value]*mu)
    
    def peek_lorentz(self,mu):
        out = ComplexMatrix()
        out.value = self.value[mu]
        return out

    def poke_lorentz(self,mu,m):
        if isinstance(m,np.ndarray):
            assert(m.shape==self.value[0].shape)
            self.value[mu] = m
       