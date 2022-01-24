from model.ising import IsingField, IsingModel
from utils.inputargs import Parser
from mpi.parallel import CartesianComm, pprint
import numpy as np 

if __name__ == '__main__': 
    PP = Parser() 
    CC = CartesianComm(mpigrid=PP.mpigrid)
    local_seed = CC.parallel_RNG(seed=PP.seed)

    field = IsingField(grid=PP.grid,cartesiancomm=CC,initialization='random',seed=local_seed)
    model = IsingModel(field=field)
    pprint(comm=CC.comm, msg=model.field.value.reshape(3,3))
    pprint(comm=CC.comm, msg=model.magnetization())
    pprint(comm=CC.comm, msg=model.energy())
    
    # print(model.nn_field.value[:,0].reshape(3,3))
    # print(model.nn_field.value[:,1].reshape(3,3))
    # print(model.nn_field.value[:,2].reshape(3,3))
    # print(model.nn_field.value[:,3].reshape(3,3))

    # print('update')
    # model.update_nn_field(idx=0)
    # model.field.value[0] = 9
    # pprint(comm=CC.comm, msg=model.field.value.reshape(3,3))
    # print(model.nn_field.value[:,0].reshape(3,3))
    # print(model.nn_field.value[:,1].reshape(3,3))
    # print(model.nn_field.value[:,2].reshape(3,3))
    # print(model.nn_field.value[:,3].reshape(3,3))
