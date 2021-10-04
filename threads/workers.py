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
        print(n)
        print(t)
        print(factor)
        nn=n
        out = []
        for tt in t:
            print(tt)
            out.append( np.split(tt,factor[nn],axis=nn) )
        print('$'*10)
        nn+=1
        if nn>=len(factor):
            return 0
        self.recursive_split(t=out,factor=factor,n=nn)
        #return out 

    def local_grid(self):
        tmp = np.copy(self.tensor_idx)
        self.recursive_split(t=[tmp],factor=self.pgrid,n=0)
        #print(tmp)
        #for n in range(len(self.pgrid)):
        #t = np.array(np.split(tmp,[self.pgrid[0],self.pgrid[1]],axis=(0,1)))
        #print(t)  
        #    break
        #print([a for a in tmp])
            #tmp = np.array([np.split(a,self.pgrid[n],axis=n) for a in tmp])
            #print(tmp)
    
    


        

