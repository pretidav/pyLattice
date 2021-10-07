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
        self.dim = len(pgrid)
        self.ptensor_idx = self.local_grid()

    def local_grid(self):
        if self.dim>=1:
            s1 = np.split(self.tensor_idx, self.pgrid[0], axis=0)
            out = s1
        if self.dim>=2:
            s2 = [np.split(ss, self.pgrid[1], axis=1) for ss in s1]
            out = s2
        if self.dim>=3:
            s3 = []
            for s in s2 : 
                s3.append([np.split(ss, self.pgrid[2], axis=2) for ss in s])
            out = s3
        if self.dim>=4:
            s4 = [] 
            for s in s3: 
                s4j = []
                for sj in s:
                    s4j.append([np.split(ss, self.pgrid[3], axis=3) for ss in sj])
                s4.append(s4j)
            out = s4
        if self.dim>=5: 
            s5 = [] 
            for s in s4: 
                s5j = []
                for sj in s:
                    s5jj = []
                    for sjj in sj:
                        s5jj.append([np.split(ss, self.pgrid[4], axis=4) for ss in sjj])
                    s5j.append(s5jj)
                s5.append(s5j)
            out = s5
        if self.dim>=6: 
            s6 = [] 
            for s in s5: 
                s6j = []
                for sj in s:
                    s6jj = []
                    for sjj in sj:
                        s6jjj = []
                        for sjjj in sjj:
                            s6jjj.append([np.split(ss, self.pgrid[5], axis=5) for ss in sjjj])
                        s6jj.append(s6jjj)
                    s6j.append(s6jj)
                s6.append(s6j)
            out = s6        

        return np.array(out) 


        

