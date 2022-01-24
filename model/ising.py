from lattice.grid import LatticeReal,LatticeVectorReal
import numpy as np 
from copy import deepcopy, copy

class IsingField(LatticeReal):
    def __init__(self,grid,cartesiancomm, initialization='random', seed = 0): 
        super().__init__(grid,cartesiancomm)
        self.seed = seed 
        self.initialization = initialization
        self.init()  

    def init(self): 
        if self.initialization=='hot': 
            self.value[:]=1
        if self.initialization=='cold': 
            self.value[:]=-1
        if self.initialization=='random': 
            np.random.seed(seed=self.seed)
            self.value[:]=np.array(np.random.randint(low=-1, high=1, size=self.value.shape),dtype='float32')


class IsingModel():
    def __init__(self,field):
        self.field = field  
        self.M = self.magnetization()
        self.E = 0
        self.nn_field = self.get_nn_field() 

    def magnetization(self): 
        return self.field.average()
    
    def energy(self): 
        self.local_energy()
        return self.local_E.reducesum()

    def get_nn_field(self): 
        out = LatticeVectorReal(grid=self.field.grid,cartesiancomm=self.field.cartesiancomm, Nd=len(self.field.grid)*2)
        for mu in range(len(self.field.grid)):
            lattice = copy(self.field)
            lattice.movebackward(mu=mu) 
            out.poke_index(mu=mu,obj=lattice)
            del lattice 
        for mu in range(len(self.field.grid)):
            lattice = copy(self.field)
            lattice.moveforward(mu=mu) 
            out.poke_index(mu=mu+len(self.field.grid),obj=lattice)
            del lattice
        return out

    def local_energy(self):
        self.local_E = IsingField(grid=self.field.grid, cartesiancomm=self.field.cartesiancomm)
        self.local_E.fill_value(n=0)
        for mu in range(len(self.field.grid)): 
            self.local_E = self.local_E + (self.nn_field.peek_index(mu) + self.nn_field.peek_index(mu+len(self.field.grid))) * self.field

    # def update_nn_field(self,idx): 
    #     for mu in range(len(self.field.grid)):
    #         if mu==0: 
    #             self.nn_field.value[idx+1,mu] = 9
    #         else: 
    #             self.nn_field.value[idx+np.prod(self.field.grid[:mu]),mu] = 9
            
class Metropolis(): 
    def __init__(self): 
        self.seed 