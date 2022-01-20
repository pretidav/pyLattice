from utils.inputargs import Parser
from mpi.parallel import CartesianComm, pprint
from lattice.grid_mpi import LatticeMPI, LatticeReal
import numpy as np

PP = Parser()
CC = CartesianComm(mpigrid=PP.mpigrid)
Lattice = LatticeReal(grid=PP.grid,cartesiancomm=CC)
Lattice.fill_value(n=1)
#print('rank {} \n{}'.format(CC.comm.rank,np.reshape(Lattice.value,PP.grid)))
#print('rank {} \n{}'.format(CC.comm.rank,Lattice.tensor_idx))
#Lattice.movebackward(mu=0)
#print('shift rank {} \n{}'.format(CC.comm.rank,Lattice.tensor_idx))
#print('shift rank {} \n{}'.format(CC.comm.rank,np.reshape(Lattice.value,PP.grid)))
#Lattice.movebackward(mu=0)
#print(Lattice.tensor_idx)

print(Lattice.reducesum())

