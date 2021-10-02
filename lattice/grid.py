import numpy as np
from linalg.tensors import *

class LatticeBase():
    def __init__(self, size):
        if isinstance(size,np.ndarray):
            self.size = size
        else :
            self.size    = np.array(size)
        self.dimensions  = len(size)
        self.length      = np.prod(self.size)   
        self.flat_idx    = self.get_flat_idx() 
        self.tensor_idx  = self.get_tensor_idx(idx=self.flat_idx)

    def moveforward(self,mu,step=1): 
        self.tensor_idx = np.roll(self.tensor_idx,shift=step,axis=mu)
        self.update_flat_idx()

    def movebackward(self,mu,step=1): 
        self.tensor_idx = np.roll(self.tensor_idx,shift=-step,axis=mu)
        self.update_flat_idx()

    def get_idx(self,x):
        idx = x[-1]
        for d in reversed(range(len(self.size)-1)):
            idx *= self.size[d]
            idx += x[d]
        return idx

    def get_tensor_idx(self,idx: np.ndarray):
        out = np.reshape(idx,self.size)
        return out

    def get_flat_idx(self):
        return np.array([i for i in range(self.length)])
        
    def update_flat_idx(self):
        self.flat_idx = np.ndarray.flatten(self.tensor_idx)

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
        self.value = self.value[self.lattice.flat_idx]

    def movebackward(self,mu,step=1): 
        self.lattice.movebackward(mu=mu,step=1)
        self.value = self.value[self.lattice.flat_idx]
        
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
        self.value = self.value[self.lattice.flat_idx]

    def movebackward(self,mu,step=1): 
        self.lattice.movebackward(mu=mu,step=1)
        self.value = self.value[self.lattice.flat_idx]
        
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
        self.value = self.value[self.lattice.flat_idx]

    def movebackward(self,mu,step=1): 
        self.lattice.movebackward(mu=mu,step=1)
        self.value = self.value[self.lattice.flat_idx]
        
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
        self.value = self.value[self.lattice.flat_idx]

    def movebackward(self,mu,step=1): 
        self.lattice.movebackward(mu=mu,step=1)
        self.value = self.value[self.lattice.flat_idx]

    def __getitem__(self, idx:int):
            return self.value[idx,:]
    
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


class LatticeVectorReal(): 
    def __init__(self, lattice: LatticeBase, Nd: int):
        self.Nd = Nd
        self.lattice = lattice 
        self.value = np.zeros(shape=(self.lattice.length,Nd),dtype=float)

    def fill_value(self, n:VectorReal):
        if isinstance(n,VectorReal):
            self.value[:] = n.value

    def moveforward(self,mu,step=1): 
        self.lattice.moveforward(mu=mu,step=1)
        self.value = self.value[self.lattice.flat_idx]

    def movebackward(self,mu,step=1): 
        self.lattice.movebackward(mu=mu,step=1)
        self.value = self.value[self.lattice.flat_idx]
        
    def __getitem__(self, idx:int):
            return self.value[idx,:]
    
    def transpose(self):
        out = LatticeVectorReal(lattice=self.lattice, Nd=self.Nd)
        for i in range(self.lattice.length):
            out.value[i]=np.transpose(self.value[i,:])
        return out 

    def __add__(self,rhs):
        out = LatticeVectorReal(lattice=self.lattice, Nd=self.Nd)
        if isinstance(rhs, LatticeVectorReal):
            assert(self.value.shape==rhs.value.shape)
            out.value = self.value + rhs.value
        return out

    def __sub__(self,rhs):
        out = LatticeVectorReal(lattice=self.lattice, Nd=self.Nd)
        if isinstance(rhs, LatticeVectorReal):
            assert(self.value.shape==rhs.value.shape)
            out.value = self.value - rhs.value
        
    def __mul__(self,rhs):
        if isinstance(rhs, LatticeVectorReal):
            out = LatticeReal(lattice=self.lattice)
            assert(self.value.shape==rhs.value.shape)
            for i in range(out.lattice.length):
                out.value[i] = np.dot(self.value[i] , rhs.value[i])
        elif isinstance(rhs, LatticeReal):
            out = LatticeVectorReal(lattice=self.lattice, Nd=self.Nd)
            for i in range(out.lattice.length):
                out.value[i] = self.value[i]*rhs.value[i]
        elif isinstance(rhs, Real):
            out = LatticeVectorReal(lattice=self.lattice, Nd=self.Nd)
            out.value = self.value * rhs.value
        return out


class LatticeVectorComplex(): 
    def __init__(self, lattice: LatticeBase, Nd: int):
        self.Nd = Nd
        self.lattice = lattice 
        self.value = np.zeros(shape=(self.lattice.length,Nd),dtype=complex)

    def fill_value(self, n:VectorComplex):
        if isinstance(n,VectorComplex):
            self.value[:] = n.value

    def moveforward(self,mu,step=1): 
        self.lattice.moveforward(mu=mu,step=1)
        self.value = self.value[self.lattice.flat_idx]

    def movebackward(self,mu,step=1): 
        self.lattice.movebackward(mu=mu,step=1)
        self.value = self.value[self.lattice.flat_idx]
        
    def __getitem__(self, idx:int):
            return self.value[idx,:]
    
    def transpose(self):
        out = LatticeVectorComplex(lattice=self.lattice, Nd=self.Nd)
        for i in range(self.lattice.length):
            out.value[i]=np.transpose(self.value[i,:])
        return out 

    def __add__(self,rhs):
        out = LatticeVectorComplex(lattice=self.lattice, Nd=self.Nd)
        if isinstance(rhs, (LatticeVectorReal,LatticeVectorComplex)):
            assert(self.value.shape==rhs.value.shape)
            out.value = self.value + rhs.value
        return out

    def __sub__(self,rhs):
        out = LatticeVectorComplex(lattice=self.lattice, Nd=self.Nd)
        if isinstance(rhs, (LatticeVectorReal,LatticeVectorComplex)):
            assert(self.value.shape==rhs.value.shape)
            out.value = self.value - rhs.value
        
    def __mul__(self,rhs):
        if isinstance(rhs, LatticeVectorReal):
            out = LatticeReal(lattice=self.lattice)
            assert(self.value.shape==rhs.value.shape)
            for i in range(out.lattice.length):
                out.value[i] = np.dot(self.value[i] , rhs.value[i])
        elif isinstance(rhs, (LatticeReal,LatticeComplex)):
            out = LatticeVectorComplex(lattice=self.lattice,Nd=self.Nd)
            for i in range(out.lattice.length):
                out.value[i] = self.value[i]*rhs.value[i]
        elif isinstance(rhs, (Complex,Real)):
            out = LatticeVectorComplex(lattice=self.lattice,Nd=self.Nd)
            out.value = self.value * rhs.value
        return out
 
    def conj(self):
        out = LatticeVectorComplex(lattice=self.lattice, Nd=self.Nd)
        for i in range(self.lattice.length):
            out.value=np.conj(self.value)
        return out 

    def re(self):
        out = LatticeVectorReal(lattice=self.lattice, Nd=self.Nd)
        out.value = np.real(self.value)
        return out

    def im(self):
        out = LatticeVectorReal(lattice=self.lattice, Nd=self.Nd)
        out.value = np.imag(self.value)
        return out


class LatticeVectorRealMatrix(): 
    def __init__(self, lattice: LatticeBase, Nd: int, N:int):
        self.Nd = Nd
        self.N  = N
        self.lattice = lattice 
        self.value = np.zeros(shape=(self.lattice.length,Nd,N,N),dtype=float)

    def fill_value(self, n:VectorRealMatrix):
        if isinstance(n,VectorRealMatrix):
            self.value[:] = n.value

    def moveforward(self,mu,step=1): 
        self.lattice.moveforward(mu=mu,step=1)
        self.value = self.value[self.lattice.flat_idx]

    def movebackward(self,mu,step=1): 
        self.lattice.movebackward(mu=mu,step=1)
        self.value = self.value[self.lattice.flat_idx]
        
    def __getitem__(self, idx:int):
            return self.value[idx,:]
    
    def transpose(self):
        out = LatticeVectorRealMatrix(lattice=self.lattice, N=self.N, Nd=self.Nd)
        for i in range(self.lattice.length):
            for n in range(self.Nd):
                out.value[i,n]=np.transpose(self.value[i,n,:,:])
        return out 

    def trace(self):
        out = LatticeVectorReal(lattice=self.lattice,Nd=self.Nd)
        for i in range(self.lattice.length):
            for n in range(self.Nd):
                out.value[i,n]=np.trace(self.value[i,n,:,:])
        return out 

    def det(self):
        out = LatticeVectorReal(lattice=self.lattice,Nd=self.Nd)
        for i in range(self.lattice.length):
            for n in range(self.Nd):
                out.value[i,n]=np.linalg.det(self.value[i,n,:,:])
        return out 

    def inv(self):
        out = LatticeVectorRealMatrix(lattice=self.lattice, N=self.N, Nd=self.Nd)
        for i in range(self.lattice.length):
            for n in range(self.Nd):
                out.value[i,n]=np.linalg.inv(self.value[i,n,:,:])
        return out 

    def __add__(self,rhs):
        out = LatticeVectorRealMatrix(lattice=self.lattice, N=self.N, Nd=self.Nd)
        if isinstance(rhs, LatticeVectorRealMatrix):
            assert(self.value.shape==rhs.value.shape)
            out.value = self.value + rhs.value
        elif isinstance(rhs, LatticeRealMatrix):
            assert(self.value[0,0].shape==rhs.value[0].shape)
            for i in range(self.lattice.length):
                for n in range(self.Nd):
                    out.value[i,n] = self.value[i,n] + rhs.value[i]
        elif isinstance(rhs, (Real,RealMatrix)):
            assert(self.value[0,0].shape==rhs.value[0].shape)
            for i in range(self.lattice.length):
                for n in range(self.Nd):
                    out.value[i,n] = self.value[i,n] + rhs.value
        elif isinstance(rhs, (float,int)):
            for i in range(self.lattice.length):
                for n in range(self.Nd):
                    out.value[i,n] = self.value[i,n] + rhs
        return out

    def __sub__(self,rhs):
        out = LatticeVectorRealMatrix(lattice=self.lattice, N=self.N, Nd=self.Nd)
        if isinstance(rhs, LatticeVectorRealMatrix):
            assert(self.value.shape==rhs.value.shape)
            out.value = self.value - rhs.value
        elif isinstance(rhs, LatticeRealMatrix):
            assert(self.value[0,0].shape==rhs.value[0].shape)
            for i in range(self.lattice.length):
                for n in range(self.Nd):
                    out.value[i,n] = self.value[i,n] - rhs.value[i]
        elif isinstance(rhs, (Real,RealMatrix)):
            assert(self.value[0,0].shape==rhs.value[0].shape)
            for i in range(self.lattice.length):
                for n in range(self.Nd):
                    out.value[i,n] = self.value[i,n] - rhs.value
        elif isinstance(rhs, (float,int)):
            for i in range(self.lattice.length):
                for n in range(self.Nd):
                    out.value[i,n] = self.value[i,n] - rhs
        return out

    def __mul__(self,rhs):
        out = LatticeVectorRealMatrix(lattice=self.lattice, N=self.N, Nd=self.Nd)
        if isinstance(rhs, LatticeVectorRealMatrix):
            assert(self.value.shape==rhs.value.shape)
            for i in range(out.lattice.length):
                for n in range(self.Nd):
                    out.value[i,n] = np.dot(self.value[i,n] , rhs.value[i,n])
        elif isinstance(rhs, LatticeRealMatrix):
            for i in range(out.lattice.length):
                for n in range(self.Nd):
                    out.value[i,n] = np.dot(self.value[i,n] , rhs.value[i])
        elif isinstance(rhs, RealMatrix):
            for i in range(out.lattice.length):
                for n in range(self.Nd):
                    out.value[i,n] = np.dot(self.value[i,n] , rhs.value)
        elif isinstance(rhs, Real):
            for i in range(out.lattice.length):
                for n in range(self.Nd):
                    out.value[i,n] = self.value[i,n] * rhs.value
        elif isinstance(rhs, (float,int)):
            assert(self.value.shape==rhs.value.shape)
            for i in range(out.lattice.length):
                for n in range(self.Nd):
                    out.value[i,n] = self.value[i,n] * rhs
        return out



class LatticeVectorComplexMatrix(): 
    def __init__(self, lattice: LatticeBase, Nd: int, N:int):
        self.Nd = Nd
        self.N  = N
        self.lattice = lattice 
        self.value = np.zeros(shape=(self.lattice.length,Nd,N,N),dtype=complex)

    def fill_value(self, n:VectorComplexMatrix):
        if isinstance(n,VectorComplexMatrix):
            self.value[:] = n.value

    def moveforward(self,mu,step=1): 
        self.lattice.moveforward(mu=mu,step=1)
        self.value = self.value[self.lattice.flat_idx]

    def movebackward(self,mu,step=1): 
        self.lattice.movebackward(mu=mu,step=1)
        self.value = self.value[self.lattice.flat_idx]
        
    def __getitem__(self, idx:int):
            return self.value[idx,:]
    
    def transpose(self):
        out = LatticeVectorComplexMatrix(lattice=self.lattice, N=self.N, Nd=self.Nd)
        for i in range(self.lattice.length):
            for n in range(self.Nd):
                out.value[i,n]=np.transpose(self.value[i,n,:,:])
        return out 

    def trace(self):
        out = LatticeVectorComplex(lattice=self.lattice,Nd=self.Nd)
        for i in range(self.lattice.length):
            for n in range(self.Nd):
                out.value[i,n]=np.trace(self.value[i,n,:,:])
        return out 

    def det(self):
        out = LatticeVectorComplex(lattice=self.lattice,Nd=self.Nd)
        for i in range(self.lattice.length):
            for n in range(self.Nd):
                out.value[i,n]=np.linalg.det(self.value[i,n,:,:])
        return out 

    def inv(self):
        out = LatticeVectorComplexMatrix(lattice=self.lattice, N=self.N, Nd=self.Nd)
        for i in range(self.lattice.length):
            for n in range(self.Nd):
                out.value[i,n]=np.linalg.inv(self.value[i,n,:,:])
        return out 

    def conj(self):
        out = LatticeVectorComplexMatrix(lattice=self.lattice, N=self.N, Nd=self.Nd)
        for i in range(self.lattice.length):
            for n in range(self.Nd):
                out.value[i,n]=np.conj(self.value[i,n,:,:])
        return out 

    def adj(self):
        tmp = LatticeVectorComplexMatrix(lattice=self.lattice, N=self.N, Nd=self.Nd)
        tmp = self.conj()
        return tmp.transpose()

    def re(self):
        out = LatticeVectorRealMatrix(lattice=self.lattice, N=self.N, Nd=self.Nd)
        out.value = np.real(self.value)
        return out

    def im(self):
        out = LatticeVectorRealMatrix(lattice=self.lattice, N=self.N, Nd=self.Nd)
        out.value = np.imag(self.value)
        return out
        
    def __add__(self,rhs):
        out = LatticeVectorComplexMatrix(lattice=self.lattice, N=self.N, Nd=self.Nd)
        if isinstance(rhs, (LatticeVectorComplexMatrix,LatticeVectorRealMatrix)):
            assert(self.value.shape==rhs.value.shape)
            out.value = self.value + rhs.value
        elif isinstance(rhs, (LatticeComplexMatrix,LatticeRealMatrix)):
            assert(self.value[0,0].shape==rhs.value[0].shape)
            for i in range(self.lattice.length):
                for n in range(self.Nd):
                    out.value[i,n] = self.value[i,n] + rhs.value[i]
        elif isinstance(rhs, (Real,RealMatrix,Complex,ComplexMatrix)):
            assert(self.value[0,0].shape==rhs.value[0].shape)
            for i in range(self.lattice.length):
                for n in range(self.Nd):
                    out.value[i,n] = self.value[i,n] + rhs.value
        elif isinstance(rhs, (float,int,complex)):
            for i in range(self.lattice.length):
                for n in range(self.Nd):
                    out.value[i,n] = self.value[i,n] + rhs
        return out

    def __sub__(self,rhs):
        out = LatticeVectorComplexMatrix(lattice=self.lattice, N=self.N, Nd=self.Nd)
        if isinstance(rhs, (LatticeVectorComplexMatrix,LatticeVectorRealMatrix)):
            assert(self.value.shape==rhs.value.shape)
            out.value = self.value - rhs.value
        elif isinstance(rhs, (LatticeComplexMatrix,LatticeRealMatrix)):
            assert(self.value[0,0].shape==rhs.value[0].shape)
            for i in range(self.lattice.length):
                for n in range(self.Nd):
                    out.value[i,n] = self.value[i,n] - rhs.value[i]
        elif isinstance(rhs, (Real,RealMatrix,Complex,ComplexMatrix)):
            assert(self.value[0,0].shape==rhs.value[0].shape)
            for i in range(self.lattice.length):
                for n in range(self.Nd):
                    out.value[i,n] = self.value[i,n] - rhs.value
        elif isinstance(rhs, (float,int,complex)):
            for i in range(self.lattice.length):
                for n in range(self.Nd):
                    out.value[i,n] = self.value[i,n] - rhs
        return out

    def __mul__(self,rhs):
        out = LatticeVectorComplexMatrix(lattice=self.lattice, N=self.N, Nd=self.Nd)
        if isinstance(rhs, (LatticeVectorComplexMatrix,LatticeVectorRealMatrix)):
            assert(self.value.shape==rhs.value.shape)
            for i in range(out.lattice.length):
                for n in range(self.Nd):
                    out.value[i,n] = np.dot(self.value[i,n] , rhs.value[i,n])
        elif isinstance(rhs, (LatticeRealMatrix,LatticeComplexMatrix)):
            for i in range(out.lattice.length):
                for n in range(self.Nd):
                    out.value[i,n] = np.dot(self.value[i,n] , rhs.value[i])
        elif isinstance(rhs, (ComplexMatrix,RealMatrix)):
            for i in range(out.lattice.length):
                for n in range(self.Nd):
                    out.value[i,n] = np.dot(self.value[i,n] , rhs.value)
        elif isinstance(rhs, (Complex,Real)):
            for i in range(out.lattice.length):
                for n in range(self.Nd):
                    out.value[i,n] = self.value[i,n] * rhs.value
        elif isinstance(rhs, (float,int,complex)):
            assert(self.value.shape==rhs.value.shape)
            for i in range(out.lattice.length):
                for n in range(self.Nd):
                    out.value[i,n] = self.value[i,n] * rhs
        return out