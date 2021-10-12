import numpy as np
from multiprocessing import cpu_count
from threading import Thread
from multiprocessing import Process
from linalg.tensors import *
import time

class LatticeBase():
    def __init__(self, grid):
        if isinstance(grid,np.ndarray):
            self.grid = grid
        else :
            self.grid    = np.array(grid)
        self.dimensions  = len(grid)
        self.length      = np.prod(self.grid)   
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
        for d in reversed(range(len(self.grid)-1)):
            idx *= self.grid[d]
            idx += x[d]
        return idx

    def get_tensor_idx(self,idx: np.ndarray):
        out = np.reshape(idx,self.grid)
        return out

    def get_flat_idx(self):
        return np.array([i for i in range(self.length)])
        
    def update_flat_idx(self):
        self.flat_idx = np.ndarray.flatten(self.tensor_idx)

class LatticeParallel(LatticeBase):
    def __init__(self,grid,pgrid):
        super().__init__(grid)
        self.pgrid  = pgrid
        self.ptensor_idx = None 
        self.pflat_idx   = None

        self.N_threads = np.prod(self.pgrid)
        self.MAX_threads = cpu_count()
        self.check_cpu_count()
        self.update_pidx()

    def get_pvalue(self,value):
        return value[self.pflat_idx]
       
    # def get_value_from_pvalue(self,value,pvalue):
    #     out = np.copy(value)
    #     for n in range(pvalue.shape[0]):
    #         out[self.pflat_idx[n]]=pvalue[n]

    def moveforward(self,mu,step=1): 
        super().moveforward(mu,step)
        self.update_pidx()

    def movebackward(self,mu,step=1): 
        super().movebackward(mu,step)
        self.update_pidx()
        
    def update_pidx(self):
        self.ptensor_idx = self.local_grid()
        self.pflat_idx = self.get_pflat_idx()

    def get_pflat_idx(self):    
        return np.array([np.ndarray.flatten(a) for a in self.ptensor_idx[:] ])

    def local_grid(self):
        return np.reshape(self.tensor_idx,newshape=[np.prod(self.pgrid)]+[int(a/self.pgrid[i]) for i,a in enumerate(self.tensor_idx.shape)])  


    #https://realpython.com/primer-on-python-decorators/

    def check_cpu_count(self):
        if self.N_threads>self.MAX_threads:
            print('## ERROR ##')
            print('## parallelization grid {} requires {} processes'.format(self.pgrid,self.N_threads))
            print('## Max threads available are {} '.format(self.MAX_threads))
            exit(1)

class Worker(Thread):
    def __init__(self, target, index):
      Thread.__init__(self)
      self.target = target
      self.index = index
    def run(self):
        self.target(i=self.index)

class LatticeReal():
    def __init__(self,lattice: LatticeParallel):
        self.lattice = lattice 
        self.value = np.zeros(shape=(self.lattice.length,1),dtype=float)
        self.pvalue = self.lattice.get_pvalue(value=self.value)

    def fill_value(self, n=0):
        if isinstance(n,Real):
            self.value[:] = n.value
            self.pvalue = self.lattice.get_pvalue(value=self.value)
        elif isinstance(n, (float,int)):
            self.value[:] = n
            self.pvalue = self.lattice.get_pvalue(value=self.value)
        
    def __getitem__(self, idx:int):
            return self.value[idx,:]

    def moveforward(self,mu,step=1): 
        self.lattice.moveforward(mu=mu,step=1)
        self.value = self.value[self.lattice.flat_idx]
        self.pvalue = self.lattice.get_pvalue(value=self.value)

    def movebackward(self,mu,step=1): 
        self.lattice.movebackward(mu=mu,step=1)
        self.value = self.value[self.lattice.flat_idx]
        self.pvalue = self.lattice.get_pvalue(value=self.value)

    def threading(self,fn):
        threads = [Worker(target=fn,index=i) for i in range(self.lattice.N_threads)]
        for t in threads:
            t.start()
        for t in threads: 
            t.join()  

    def parallel(self,fn):
        procs = [Process(target=fn,args=(i,)) for i in range(self.lattice.N_threads)]
        for p in procs:
            p.start()
        for p in procs: 
            p.join()  

    def __add__PROCESS(self,rhs):
        out = LatticeReal(lattice=self.lattice)
        def fn(i):
            x = self.lattice.pflat_idx[i]
            if isinstance(rhs, LatticeReal):
                assert(self.value[x].shape==rhs.value[x].shape)
                out.value[x] = self.value[x] + rhs.value[x]
            elif isinstance(rhs, Real):
                out.value[x] = self.value[x] + rhs.value[x]
            elif isinstance(rhs, (int,float)):
                out.value[x] = self.value[x] + rhs    
        self.parallel(fn=fn)
        return out

    def __add__THREAD(self,rhs):
        out = LatticeReal(lattice=self.lattice)
        def fn(i):
            x = self.lattice.pflat_idx[i]
            if isinstance(rhs, LatticeReal):
                assert(self.value[x].shape==rhs.value[x].shape)
                out.value[x] = self.value[x] + rhs.value[x]
            elif isinstance(rhs, Real):
                out.value[x] = self.value[x] + rhs.value[x]
            elif isinstance(rhs, (int,float)):
                out.value[x] = self.value[x] + rhs    
        self.threading(fn=fn)
        return out

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
    def __init__(self,lattice: LatticeParallel):
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
    def __init__(self, lattice: LatticeParallel, N: int):
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
    def __init__(self, lattice: LatticeParallel, N: int):
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
    def __init__(self, lattice: LatticeParallel, Nd: int):
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
    def __init__(self, lattice: LatticeParallel, Nd: int):
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
    def __init__(self, lattice: LatticeParallel, Nd: int, N:int):
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
    def __init__(self, lattice: LatticeParallel, Nd: int, N:int):
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