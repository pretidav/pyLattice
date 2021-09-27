import numpy as np
import unittest

# Types: 
# Real     - real number 
# Complex  - Real,Real
# RealMatrix 
# ComplexMatrix
# RealLattice
# ComplexLattice
# RealLatticeMatrix
# ComplexLatticeMatrix
# LCMatrix
# LSCMatrix
# LatticeCMatrix
# LatticeSCMatrix
# LatticeLCMatrix 
# LatticeLSCMatrix

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

class RealMatrix():
    def __init__(self,N: int):
        self.N = N 
        self.value = np.zeros((N,N), dtype=float)

    def transpose(self):
        self.value = self.value.transpose()
   
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

class ComplexMatrix(RealMatrix):
    def __init__(self, N: int):
        super().__init__(N)
        self.value = np.zeros((N,N), dtype=complex)
    
    def re(self):
        out = RealMatrix(self.N)
        out.value = np.real(self.value)
        return out

    def im(self):
        out = RealMatrix(self.N)
        out.value = np.imag(self.value)
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
       