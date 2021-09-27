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
# CMatrix
# SCMatrix 
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
        if isinstance(X, Real) or isinstance(X,Complex):
            out.value = self.value*X.value
        elif isinstance(X, float) or isinstance(X,int) or isinstance(X,complex) : 
            out.value = self.value*X
        return out
        
    def __pow__(self,n):
        out = Real()
        if isinstance(n, float) or isinstance(n, int):
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
        if isinstance(X, RealMatrix) or isinstance(X,ComplexMatrix):
            out.value = self.value + X.value
        elif isinstance(X, complex) or isinstance(X,Real):
            print('here')
            out.value = self.value + X.value
        return out

    def __mul__(self,X):
        out = RealMatrix(self.N)
        if isinstance(X, RealMatrix):
            out.value = np.dot(self.value,X.value)
        elif isinstance(X, complex) or isinstance(X,Real):
            out.value = self.value*X.value
        return out



class ComplexMatrix(RealMatrix):
    def __init__(self, N: int):
        super().__init__(N)
        self.value = np.zeros((N,N),dtype=float)
    
    def re(self):
        out = RealMatrix(self.N)
        out.value = np.real(self.value)
        return out

    def im(self):
        out = RealMatrix(self.N)
        out.value = np.imag(self.value)
        return out
