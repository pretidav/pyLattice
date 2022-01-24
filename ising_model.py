from model.ising import IsingField, IsingModel
from utils.inputargs import Parser
from mpi.parallel import CartesianComm, pprint

if __name__ == '__main__': 
    PP = Parser() 
    CC = CartesianComm(mpigrid=PP.mpigrid)
    local_seed = CC.parallel_RNG(seed=PP.seed)

    field = IsingField(grid=PP.grid,cartesiancomm=CC,initialization='random',seed=local_seed)
    model = IsingModel(field=field)
    pprint(comm=CC.comm, msg=model.field.value)
    pprint(comm=CC.comm, msg=model.magnetization())
    pprint(comm=CC.comm, msg=model.energy())
    
    
