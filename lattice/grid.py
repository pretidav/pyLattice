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
    
class LatticeReal(Real):
    def __init__(self,lattice: LatticeBase):
        super().__init__()
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


class LatticeComplex(Complex):
    def __init__(self,lattice: LatticeBase):
        super().__init__()
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
        
class LatticeRealMatrix(RealMatrix): 
    def __init__(self,lattice: LatticeBase):
        super().__init__()
        self.lattice = lattice
        self.fill_grid()
        self.value = self.get_value()

    def fill_grid(self, n:RealMatrix):
        if isinstance(n,RealMatrix):
            self.grid = [n]*self.lattice.length
        self.value = self.get_value()

    def get_value(self):
        return np.array([a.value for a in self.grid])
    