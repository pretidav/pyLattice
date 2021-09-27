import numpy as np

class LatticeBase():
    def __init__(self, size):
        if isinstance(size,np.ndarray):
            self.size = size
        else :
            self.size = np.array(size)
        self.dimensions = len(size)
        self.coor = np.array([0]*self.size,dtype=int)

    def moveforward(self,mu,step=1): 
        self.coor[mu] = (self.coor[mu] + step)%self.size[mu]

    def movebackward(self,mu,step=1): 
        self.coor[mu] = (self.coor[mu] - step)%self.size[mu]

