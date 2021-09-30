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
        self.value = np.zeros(shape=(self.lattice.length,1),dtype=float)

    def fill_value(self, n=0):
        if isinstance(n,Real):
            self.value[:] = n.value
        elif isinstance(n, (float,int)):
            self.value = n
        
    def __getitem__(self, idx:int):
            return self.value[idx,:]

    def moveforward(self,mu,step=1): 
        self.lattice.moveforward(mu=mu,step=1)

    def movebackward(self,mu,step=1): 
        self.lattice.movebackward(mu=mu,step=1)
        
    def get_idx(self,coor):
        return self.lattice.get_idx(x=coor)

    def __add__(self,rhs):
        out = LatticeReal(lattice=self.lattice)
        if isinstance(rhs, LatticeReal):
            assert(self.value.shape==rhs.value.shape)
            out.value = self.value + rhs.value
        elif isinstance(rhs, Real):
            out.value = self.value + rhs.value
        elif isinstance(rhs, (int,float)):
            out.value = self.value + rhs    
        return out

    def __sub__(self,rhs):
        out = LatticeReal(lattice=self.lattice)
        if isinstance(rhs, LatticeReal):
            assert(self.value.shape==rhs.value.shape)
            out.value = self.value - rhs.value
        elif isinstance(rhs, Real):
            out.value = self.value - rhs.value
        elif isinstance(rhs, (int,float)):
            out.value = self.value - rhs    
        return out

    def __mul__(self,rhs):
        out = LatticeReal(lattice=self.lattice)
        if isinstance(rhs, LatticeReal):
            assert(self.value.shape==rhs.value.shape)
            for i in range(out.lattice.length):
                out.value[i] = self.value[i] * rhs.value[i]
        elif isinstance(rhs, Real):
            for i in range(out.lattice.length):
                out.value[i] = self.value[i] * rhs.value
        elif isinstance(rhs, (int,float)):
            for i in range(out.lattice.length):
                out.value[i] = self.value[i] * rhs
        return out

class LatticeComplex():
    def __init__(self,lattice: LatticeBase):
        self.lattice = lattice 
        self.value = np.zeros(shape=(self.lattice.length,1),dtype=complex)

    def fill_value(self, n=0):
        if isinstance(n,(Real,Complex)):
            self.value[:] = n.value
        elif isinstance(n, (complex,float,int)):
            self.value = n

    def __getitem__(self, idx:int):
            return self.value[idx,:]

    def moveforward(self,mu,step=1): 
        self.lattice.moveforward(mu=mu,step=1)

    def movebackward(self,mu,step=1): 
        self.lattice.movebackward(mu=mu,step=1)
        
    def get_idx(self,x):
        return self.lattice.get_idx(x=x)
       
    
    def __add__(self,rhs):
        out = LatticeComplex(lattice=self.lattice)
        if isinstance(rhs, (LatticeReal,LatticeComplex)):
            assert(self.value.shape==rhs.value.shape)
            out.value = self.value + rhs.value
        elif isinstance(rhs, (Complex,Real)):
            out.value = self.value + rhs.value
        elif isinstance(rhs, (int,float,complex)):
            out.value = self.value + rhs    
        return out

    def __sub__(self,rhs):
        out = LatticeComplex(lattice=self.lattice)
        if isinstance(rhs, (LatticeComplex,LatticeReal)):
            assert(self.value.shape==rhs.value.shape)
            out.value = self.value - rhs.value
        elif isinstance(rhs, (Complex,Real)):
            out.value = self.value - rhs.value
        elif isinstance(rhs, (int,float,complex)):
            out.value = self.value - rhs    
        return out

    def __mul__(self,rhs):
        out = LatticeComplex(lattice=self.lattice)
        if isinstance(rhs, (LatticeComplex,LatticeReal)):
            assert(self.value.shape==rhs.value.shape)
            for i in range(out.lattice.length):
                out.value[i] = self.value[i] * rhs.value[i]
        elif isinstance(rhs, (Complex,Real)):
            for i in range(out.lattice.length):
                out.value[i] = self.value[i] * rhs.value
        elif isinstance(rhs, (int,float,complex)):
            for i in range(out.lattice.length):
                out.value[i] = self.value[i] * rhs
        return out
class LatticeRealMatrix(): 
    def __init__(self, lattice: LatticeBase, N: int):
        self.N = N
        self.lattice = lattice 
        self.value = np.zeros(shape=(self.lattice.length,N,N),dtype=float)

    def fill_value(self, n:RealMatrix):
        if isinstance(n,RealMatrix):
            self.value[:] = n.value

    def moveforward(self,mu,step=1): 
        self.lattice.moveforward(mu=mu,step=1)

    def movebackward(self,mu,step=1): 
        self.lattice.movebackward(mu=mu,step=1)
        
    def get_idx(self,x):
        return self.lattice.get_idx(x=x)

    def __getitem__(self, idx:int):
            return self.value[idx,:]
    
    def transpose(self):
        out = LatticeRealMatrix(lattice=self.lattice, N=self.N)
        for i in range(self.lattice.length):
            out.value[i]=np.transpose(self.value[i,:,:])
        return out 

    def trace(self):
        out = LatticeReal(lattice=self.lattice)
        for i in range(self.lattice.length):
            out.value[i]=np.trace(self.value[i,:,:])
        return out 

    def det(self):
        out = LatticeReal(lattice=self.lattice)
        for i in range(self.lattice.length):
            out.value[i]=np.linalg.det(self.value[i,:,:])
        return out 

    def inv(self):
        out = LatticeRealMatrix(lattice=self.lattice, N=self.N)
        for i in range(self.lattice.length):
            out.value[i]=np.linalg.inv(self.value[i,:,:])
        return out 

    def __add__(self,rhs):
        out = LatticeRealMatrix(lattice=self.lattice, N=self.N)
        if isinstance(rhs, LatticeRealMatrix):
            assert(self.value.shape==rhs.value.shape)
            out.value = self.value + rhs.value
        elif isinstance(rhs, RealMatrix):
            assert(self.value[0].shape==rhs.value.shape)
            out.value = self.value + rhs.value
        return out

    def __sub__(self,rhs):
        out = LatticeRealMatrix(lattice=self.lattice, N=self.N)
        if isinstance(rhs, LatticeRealMatrix):
            assert(self.value.shape==rhs.value.shape)
            out.value = self.value - rhs.value
        elif isinstance(rhs, RealMatrix):
            assert(self.value[0].shape==rhs.value.shape)
            out.value = self.value - rhs.value
        return out

    def __mul__(self,rhs):
        out = LatticeRealMatrix(lattice=self.lattice, N=self.N)
        if isinstance(rhs, LatticeRealMatrix):
            assert(self.value.shape==rhs.value.shape)
            for i in range(out.lattice.length):
                out.value[i] = np.dot(self.value[i] , rhs.value[i])
        elif isinstance(rhs, RealMatrix):
            assert(self.value[0].shape==rhs.value.shape)
            for i in range(out.lattice.length):
                out.value[i] = np.dot(self.value[i], rhs.value)
        elif isinstance(rhs, Real):
            out.value = self.value * rhs.value
        return out

class LatticeComplexMatrix(): 
    def __init__(self, lattice: LatticeBase, N: int):
        self.N = N
        self.lattice = lattice 
        self.value = np.zeros(shape=(self.lattice.length,N,N),dtype=complex)

    def fill_value(self, n:ComplexMatrix):
        if isinstance(n,ComplexMatrix):
            self.value[:] = n.value

    def moveforward(self,mu,step=1): 
        self.lattice.moveforward(mu=mu,step=1)

    def movebackward(self,mu,step=1): 
        self.lattice.movebackward(mu=mu,step=1)
    
    def __getitem__(self, idx:int):
            return self.value[idx,:]
    
    def get_idx(self,x):
        return self.lattice.get_idx(x=x)

    def transpose(self):
        out = LatticeComplexMatrix(lattice=self.lattice, N=self.N)
        for i in range(self.lattice.length):
            out.value[i]=np.transpose(self.value[i,:,:])
        return out 

    def trace(self):
        out = LatticeComplexMatrix(lattice=self.lattice, N=self.N)
        for i in range(self.lattice.length):
            out.value[i]=np.trace(self.value[i,:,:])
        return out 
    
    def det(self):
        out = LatticeComplexMatrix(lattice=self.lattice, N=self.N)
        for i in range(self.lattice.length):
            out.value[i]=np.linalg.det(self.value[i,:,:])
        return out 

    def inv(self):
        out = LatticeComplexMatrix(lattice=self.lattice, N=self.N)
        for i in range(self.lattice.length):
            out.value[i]=np.linalg.inv(self.value[i,:,:])
        return out 

    def conj(self):
        out = LatticeComplexMatrix(lattice=self.lattice, N=self.N)
        for i in range(self.lattice.length):
            out.value[i]=np.conj(self.value[i,:,:])
        return out 

    def adj(self):
        tmp = LatticeComplexMatrix(lattice=self.lattice, N=self.N)
        tmp = self.conj()
        return tmp.transpose()

    def re(self):
        out = LatticeRealMatrix(lattice=self.lattice, N=self.N)
        out.value = np.real(self.value)
        return out

    def im(self):
        out = LatticeRealMatrix(lattice=self.lattice, N=self.N)
        out.value = np.imag(self.value)
        return out

    def __add__(self,rhs):
        out = LatticeComplexMatrix(lattice=self.lattice, N=self.N)
        if isinstance(rhs, (LatticeComplexMatrix,LatticeRealMatrix)):
            assert(self.value.shape==rhs.value.shape)
            out.value = self.value + rhs.value
        elif isinstance(rhs, (ComplexMatrix,RealMatrix)):
            assert(self.value[0].shape==rhs.value.shape)
            out.value = self.value + rhs.value
        return out

    def __sub__(self,rhs):
        out = LatticeComplexMatrix(lattice=self.lattice, N=self.N)
        if isinstance(rhs, (LatticeRealMatrix,LatticeComplexMatrix)):
            assert(self.value.shape==rhs.value.shape)
            out.value = self.value - rhs.value
        elif isinstance(rhs, (RealMatrix,ComplexMatrix)):
            assert(self.value[0].shape==rhs.value.shape)
            out.value = self.value - rhs.value
        return out

    def __mul__(self,rhs):
        out = LatticeComplexMatrix(lattice=self.lattice, N=self.N)
        if isinstance(rhs, (LatticeRealMatrix,LatticeComplexMatrix)):
            assert(self.value.shape==rhs.value.shape)
            for i in range(self.lattice.length):
                out.value[i] = np.dot(self.value[i] , rhs.value[i])
        elif isinstance(rhs, (RealMatrix,ComplexMatrix)):
            assert(self.value[0].shape==rhs.value.shape)
            for i in range(self.lattice.length):
                out.value[i] = np.dot(self.value[i], rhs.value)
        elif isinstance(rhs, (Real,Complex)):
            out.value = self.value * rhs.value
        elif isinstance(rhs, (float,int,complex)):
            out.value = self.value * rhs    
        return out
