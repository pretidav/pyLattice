from model.ising import IsingField, IsingModel
from utils.inputargs import Parser, Logger
from mpi.parallel import CartesianComm
import numpy as np 

if __name__ == '__main__': 
    PP = Parser() 
    CC = CartesianComm(mpigrid=PP.mpigrid)
    LL = Logger(debug=PP.debug,logfile=PP.logfile)
    local_seed = CC.parallel_RNG(seed=PP.seed)

    field = IsingField(grid=PP.grid,cartesiancomm=CC,initialization='random',seed=local_seed)
    model = IsingModel(field=field,logger=LL.log, beta=PP.beta)

    model.run_mc(steps=PP.steps,online_obs=True)
    model.plot_mc()