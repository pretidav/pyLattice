from mpi.parallel import CartesianComm
from utils.inputargs import Parser
from lattice.grid import LatticeRealMatrix
import numpy as np 
import time

if __name__=='__main__': 

    CC = CartesianComm(mpigrid=[2,2,1,1])
    test = np.reshape(
        np.array([int(i) for i in range(20*20*40*40*2*2)], dtype='float32'), (20*20*40*40,  2, 2))    
    Lattice = LatticeRealMatrix(grid=[20,20,40,40], cartesiancomm=CC, N=2)
    Lattice2 = LatticeRealMatrix(grid=[20,20,40,40], cartesiancomm=CC, N=2)
    Lattice.fill_value(n=test)
    Lattice2.fill_value(n=test)
    start = time.time()
    result = (Lattice*Lattice2).reducesum()
    end = time.time()
    print(end - start)

