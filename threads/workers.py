from multiprocessing import cpu_count
from threading import Thread
import numpy as np
from numpy.lib.index_tricks import AxisConcatenator

class Worker():
    def __init__(self, f):
      Thread.__init__(self)
      self.f = f
    def run(self):
        self.f

class Parallel():
    def __init__(self, pgrid, tensor_idx, worker = None):
        self.pgrid  = pgrid 
        self.worker = worker
        self.tensor_idx = tensor_idx
        self.local_grid()

    def recursive_split(self,t,factor,n):
        nn=n
        out = [np.split(tt,factor[nn],axis=nn) for tt in t][0] #this is wrong
        nn+=1
        if nn<len(factor):
            out = self.recursive_split(t=out,factor=factor,n=nn)
        return out

    def local_grid(self):
        tmp = np.copy(self.tensor_idx)
        out =  self.recursive_split(t=[tmp],factor=self.pgrid,n=0)
        print(out)

    


        

