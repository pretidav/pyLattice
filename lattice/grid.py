import numpy as np

class LatticeBase():
    def __init__(self, size):
        if isinstance(size,np.ndarray):
            self.size = size
        else :
            self.size = np.array(size)
        self.dimensions = len(size)
        self.coor  = np.array([0]*self.size,dtype=int)
        self.grid  = np.zeros(shape=np.prod(self.size))
        self.idx   = self.get_idx(self.coor) 

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
    
# class LatticeObject(LatticeBase):
#     def __init__(self):

