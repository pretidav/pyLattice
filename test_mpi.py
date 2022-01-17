from utils.inputargs import Parser
from mpi.parallel import CartesianComm, pprint

PP = Parser()
CC = CartesianComm(mpigrid=PP.mpigrid)

