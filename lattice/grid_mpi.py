from tkinter import Grid
from mpi.parallel import * 
from linalg.tensors import *
import numpy as np 


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

    def pick_last_slice(self,tensor,mu): 
        return np.take(tensor,indices=-1,axis=mu)

    def pick_first_slice(self,tensor,mu): 
        return np.take(tensor,indices=0,axis=mu)
    

class LatticeMPI(LatticeBase):
    def __init__(self, grid, cartesiancomm):
        super().__init__(grid)
        self.cartesiancomm  = cartesiancomm 

    def moveforward(self,mu,step=1,value=None):
        out = value 
        snd_idx = np.ndarray.flatten(self.pick_last_slice(tensor=self.tensor_idx,mu=mu))
        snd_halo = out[snd_idx]
        tensor_idx_tmp = self.tensor_idx
        super().moveforward(mu=mu,step=step)
        out = out[self.flat_idx]
        rcv_halo = self.cartesiancomm.forwardshift(mu, snd_buf=snd_halo)
        rcv_idx = np.ndarray.flatten(self.pick_first_slice(tensor=tensor_idx_tmp,mu=mu))
        out[rcv_idx]=rcv_halo
        return out 
        
    def movebackward(self,mu,step=1,value=None): 
        out = value
        snd_idx = np.ndarray.flatten(self.pick_first_slice(tensor=self.tensor_idx, mu=mu))
        snd_halo = out[snd_idx]
        tensor_idx_tmp = self.tensor_idx
        super().movebackward(mu=mu,step=step)
        out = out[self.flat_idx]
        rcv_halo = self.cartesiancomm.backwardshift(mu, snd_buf=snd_halo)
        rcv_idx = np.ndarray.flatten(self.pick_last_slice(tensor=tensor_idx_tmp,mu=mu))
        out[rcv_idx]=rcv_halo
        return out

    def ReduceSum(self,value): 
        out = np.array(0,dtype=np.float)
        snd_buf = np.array(np.sum(self.value),dtype=np.float)
        self.cartesiancomm.comm.Allreduce([snd_buf, MPI.FLOAT], [out, MPI.FLOAT], op=MPI.SUM)
        print(out)
        return out

    def Average(self,value): 
        return self.ReduceSum(value)/(np.prod(self.grid)*np.prod(self.cartesiancomm.mpigrid))

class LatticeReal(LatticeMPI):
    def __init__(self,grid,cartesiancomm):
        super().__init__(grid=grid,cartesiancomm=cartesiancomm) 
        self.value = np.zeros(shape=(self.length,1),dtype=float)
       
    def fill_value(self, n=0):
        if isinstance(n,Real):
            self.value[:] = n.value
        elif isinstance(n, (float,int)):
            self.value[:] = n
        
    def __getitem__(self, idx:int):
            return self.value[idx,:]

    def moveforward(self,mu,step=1): 
        self.value = super().moveforward(mu=mu,step=1,value=self.value)
        
    def movebackward(self,mu,step=1): 
        self.value = super().movebackward(mu=mu,step=1,value=self.value)

    def reducesum(self): 
        return super().ReduceSum(value=self.value)

    def average(self): 
        return super().Average(value=self.value)

    def __add__(self,rhs):
        out = LatticeReal(grid=self.grid,cartesiancomm=self.cartesiancomm)
        if isinstance(rhs, LatticeReal):
            assert(self.value.shape==rhs.value.shape)
            out.value = self.value + rhs.value
        elif isinstance(rhs, Real):
            out.value = self.value + rhs.value
        elif isinstance(rhs, (int,float)):
            out.value = self.value + rhs    
        return out

    def __sub__(self,rhs):
        out = LatticeReal(grid=self.grid,cartesiancomm=self.cartesiancomm)
        if isinstance(rhs, LatticeReal):
            assert(self.value.shape==rhs.value.shape)
            out.value = self.value - rhs.value
        elif isinstance(rhs, Real):
            out.value = self.value - rhs.value
        elif isinstance(rhs, (int,float)):
            out.value = self.value - rhs    
        return out

    def __mul__(self,rhs):
        out = LatticeReal(grid=self.grid,cartesiancomm=self.cartesiancomm)
        if isinstance(rhs, LatticeReal):
            assert(self.value.shape==rhs.value.shape)
            out.value = self.value * rhs.value
        elif isinstance(rhs, Real):
            out.value = self.value * rhs.value
        elif isinstance(rhs, (int,float)):
            out.value = self.value * rhs
        return out