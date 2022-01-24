from lattice.grid import LatticeReal
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

    def magnetization(self): 
        return self.field.average()
    
    def energy(self): 
        self.local_energy()
        return self.local_E.reducesum()

    def local_energy(self):
        self.local_E = IsingField(grid=self.field.grid, cartesiancomm=self.field.cartesiancomm)
        self.local_E.fill_value(n=0)
        for mu in range(len(self.field.grid)): 
            R_lattice = copy(self.field)
            L_lattice = copy(self.field)
            R_lattice.moveforward(mu=mu)
            L_lattice.movebackward(mu=mu)
            self.local_E = self.local_E + (R_lattice + L_lattice)*self.field
            del R_lattice, L_lattice
            
class Metropolis(): 
    def __init__(self): 
        self.seed 