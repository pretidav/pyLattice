import numpy as np
from linalg.tensors import *

class LatticeBase():
    def __init__(self, size):
        if isinstance(size,np.ndarray):
            self.size = size
        else :
            self.size = np.array(size)
        self.dimensions = len(size)
        self.coor   = np.array([0]*self.size,dtype=int)
        self.idx    = self.get_idx(self.coor) 
        self.length = np.prod(self.size)

    def moveforward(self,mu,step=1): 
        self.coor[mu] = (self.coor[mu] + step)%self.size[mu]
        self.idx = self.get_idx(self.coor)

    def movebackward(self,mu,step=1): 
        self.coor[mu] = (self.coor[mu] - step)%self.size[mu]
        self.idx = self.get_idx(self.coor)
        
    def get_idx(self,x):
        idx = x[-1]
        for d in reversed(range(len(self.size)-1)):
            idx *= self.size[d]
            idx += x[d]
        return idx
    
class LatticeReal():
    def __init__(self,lattice: LatticeBase):
        self.lattice = lattice
        self.fill_grid() 
        self.value = self.get_value()

    def fill_grid(self, n=0):
        if isinstance(n,Real):
            self.grid = [n]*self.lattice.length
        elif isinstance(n, (float,int)):
            self.grid = [Real(n)]*self.lattice.length
        self.value = self.get_value()
    
    def get_value(self):
        return np.array([a.value for a in self.grid])

class LatticeComplex():
    def __init__(self,lattice: LatticeBase):
        self.lattice = lattice
        self.fill_grid()
        self.value = self.get_value()

    def fill_grid(self, n=0):
        if isinstance(n,Complex):
            self.grid = [n]*self.lattice.length
        elif isinstance(n, (float,int,complex)):
            self.grid = [Complex(n)]*self.lattice.length
        self.value = self.get_value()

    def get_value(self):
        return np.array([a.value for a in self.grid])
        
class LatticeRealMatrix(): 
    def __init__(self, lattice: LatticeBase, N: int):
        self.N = N
        self.lattice = lattice
        self.fill_grid(RealMatrix(N=self.N))
        self.value = self.get_value()

    def fill_grid(self, n:RealMatrix):
        if isinstance(n,RealMatrix):
            self.grid = [n]*self.lattice.length
        self.value = self.get_value()

    def get_value(self):
        return np.array([a.value for a in self.grid])
    
    def transpose(self):
        out = LatticeRealMatrix(lattice=self.lattice, N=self.N)
        out.grid = [a.transpose() for a in self.grid]
        out.value = out.get_value()
        return out 

    def trace(self):
        out = LatticeReal(lattice=self.lattice)
        out.grid = np.array([a.trace() for a in self.grid])
        out.value = out.get_value()
        return out
    
    def det(self):
        out = LatticeReal(lattice=self.lattice)
        out.grid = np.array([a.det() for a in self.grid])
        out.value = out.get_value()
        return out

    def inv(self):
        out = LatticeRealMatrix(lattice=self.lattice, N=self.N)
        out.grid = np.array([a.inv() for a in self.grid])
        out.value = out.get_value()
        return out

    def __add__(self,X):
        out = LatticeRealMatrix(lattice=self.lattice, N=self.N)
        if isinstance(X, LatticeRealMatrix):
            assert(len(self.grid)==len(X.grid))
            assert(self.grid[0].value.shape==X.grid[0].value.shape)
            out.value = self.value + X.value
        if isinstance(X, RealMatrix):
            assert(self.grid[0].value.shape==X.value.shape)
            out.value = self.value + X.value
        return out

    def __sub__(self,X):
        out = LatticeRealMatrix(lattice=self.lattice, N=self.N)
        if isinstance(X, LatticeRealMatrix):
            assert(len(self.grid)==len(X.grid))
            assert(self.grid[0].value.shape==X.grid[0].value.shape)
            out.value = self.value - X.value
        if isinstance(X, RealMatrix):
            assert(self.grid[0].value.shape==X.value.shape)
            out.value = self.value - X.value
        return out

    def __mul__(self,X):
        out = LatticeRealMatrix(lattice=self.lattice, N=self.N)
        if isinstance(X, LatticeRealMatrix):
            assert(len(self.grid)==len(X.grid))
            assert(self.grid[0].value.shape==X.grid[0].value.shape)
            out.value = np.dot(self.value , X.value)
        if isinstance(X, RealMatrix):
            assert(self.grid[0].value.shape==X.value.shape)
            out.value = np.dot(self.value, X.value)
        if isinstance(X, Real):
            out.value = self.value * X.value
        return out

class LatticeComplexMatrix(): 
    def __init__(self, lattice: LatticeBase, N: int):
        self.N = N
        self.lattice = lattice
        self.fill_grid(ComplexMatrix(N=self.N))
        self.value = self.get_value()

    def fill_grid(self, n:ComplexMatrix):
        if isinstance(n,ComplexMatrix):
            self.grid = [n]*self.lattice.length
        self.value = self.get_value()

    def get_value(self):
        return np.array([a.value for a in self.grid])
    
    def transpose(self):
        out = LatticeComplexMatrix(lattice=self.lattice, N=self.N)
        out.grid = [a.transpose() for a in self.grid]
        out.value = out.get_value()
        return out 

    def trace(self):
        out = LatticeComplex(lattice=self.lattice)
        out.grid = np.array([a.trace() for a in self.grid])
        out.value = out.get_value()
        return out
    
    def det(self):
        out = LatticeComplex(lattice=self.lattice)
        out.grid = np.array([a.det() for a in self.grid])
        out.value = out.get_value()
        return out

    def inv(self):
        out = LatticeComplexMatrix(lattice=self.lattice, N=self.N)
        out.grid = np.array([a.inv() for a in self.grid])
        out.value = out.get_value()
        return out

    def conj(self):
        out = LatticeComplexMatrix(lattice=self.lattice, N=self.N)
        out.grid = np.array([a.conj() for a in self.grid])
        out.value = out.get_value()
        return out 

    def adj(self):
        tmp = LatticeComplexMatrix(lattice=self.lattice, N=self.N)
        tmp = self.conj()
        return tmp.transpose()

    def re(self):
        out = LatticeRealMatrix(lattice=self.lattice, N=self.N)
        out.grid = np.array([a.re() for a in self.grid])
        out.value = out.get_value()
        return out

    def im(self):
        out = LatticeRealMatrix(lattice=self.lattice, N=self.N)
        out.grid = np.array([a.im() for a in self.grid])
        out.value = out.get_value()
        return out

    def __add__(self,X):
        out = LatticeComplexMatrix(lattice=self.lattice, N=self.N)
        if isinstance(X, (LatticeComplexMatrix,LatticeRealMatrix)):
            assert(len(self.grid)==len(X.grid))
            assert(self.grid[0].value.shape==X.grid[0].value.shape)
            out.value = self.value + X.value
        if isinstance(X, (ComplexMatrix,RealMatrix)):
            assert(self.grid[0].value.shape==X.value.shape)
            out.value = self.value + X.value
        return out

    def __sub__(self,X):
        out = LatticeRealMatrix(lattice=self.lattice, N=self.N)
        if isinstance(X, (LatticeRealMatrix,LatticeComplexMatrix)):
            assert(len(self.grid)==len(X.grid))
            assert(self.grid[0].value.shape==X.grid[0].value.shape)
            out.value = self.value - X.value
        if isinstance(X, (RealMatrix,ComplexMatrix)):
            assert(self.grid[0].value.shape==X.value.shape)
            out.value = self.value - X.value
        return out

    def __mul__(self,X):
        out = LatticeRealMatrix(lattice=self.lattice, N=self.N)
        if isinstance(X, (LatticeRealMatrix,LatticeComplexMatrix)):
            assert(len(self.grid)==len(X.grid))
            assert(self.grid[0].value.shape==X.grid[0].value.shape)
            out.value = np.dot(self.value , X.value)
        if isinstance(X, (RealMatrix,ComplexMatrix)):
            assert(self.grid[0].value.shape==X.value.shape)
            out.value = np.dot(self.value, X.value)
        if isinstance(X, (Real,Complex)):
            out.value = self.value * X.value
        return out
