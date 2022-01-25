from lattice.grid import LatticeReal,LatticeVectorReal
import numpy as np 
from copy import deepcopy, copy

from mpi.parallel import pprint

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
            self.value[:]=np.array(np.random.choice([-1,1],self.value.shape),dtype='float32')

class IsingModel():
    def __init__(self,field,logger,beta):
        self.beta = beta
        self.log = logger 
        self.field = field  
        self.M = self.magnetization()
        self.E = 0
        self.nn_field = self.get_nn_field() 
        self.boundary_up, self.boundary_down = self.get_boundary_idx()
        self.local_E = self.local_energy()

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
        local_E = IsingField(grid=self.field.grid, cartesiancomm=self.field.cartesiancomm)
        local_E.fill_value(n=0)
        for mu in range(len(self.field.grid)): 
            local_E = local_E + (self.nn_field.peek_index(mu) + self.nn_field.peek_index(mu+len(self.field.grid))) * self.field
        return local_E

    def flip_site_field(self,idx): 
        self.field.value[idx] *= -1
        self.update_nn_field(idx)

    def update_nn_field(self,idx): 
        lengrid = len(self.field.grid)
        for ii in idx: 
            count = 0 
            for mu in reversed(range(lengrid)):
                shift = np.prod(self.field.grid[:mu])
                if ii not in self.boundary_up[count]:
                    self.nn_field.value[ii-shift,count] *=-1 
                if ii not in self.boundary_down[count]:
                    self.nn_field.value[ii+shift,count+lengrid] *=-1 
                count +=1
    
    def get_boundary_idx(self):
        boundary_up=[] 
        boundary_down=[]     
        dummy_tensor_idx = np.reshape(np.array([i for i in range(self.field.length)]), self.field.grid)
        for mu in range(len(self.field.grid)):
            boundary_up.append(np.ndarray.flatten(self.field.pick_first_slice(tensor=dummy_tensor_idx, mu=mu)))
            boundary_down.append(np.ndarray.flatten(self.field.pick_last_slice(tensor=dummy_tensor_idx, mu=mu)))
        return boundary_up, boundary_down

    def global_update(self):
        flip_idx = self.metropolis_test()
        self.field.value = np.where(flip_idx,self.field.value*-1,self.field.value)
        self.update_nn_field(idx=flip_idx)
        self.local_E = self.local_energy()
        
    def metropolis_test(self):
        rng = np.random.random(size=self.field.length) 
        return rng<np.exp(-self.local_E.value/(self.beta)) 
        
    def run_mc(self,steps): 
        pprint(comm=self.field.cartesiancomm.comm, msg=self.magnetization())
        pprint(comm=self.field.cartesiancomm.comm, msg=self.energy())
        for n in range(steps): 
            self.global_update()
            pprint(comm=self.field.cartesiancomm.comm, msg='epoch: {}'.format(n))
            pprint(comm=self.field.cartesiancomm.comm, msg='M: {}'.format(self.magnetization()))
            pprint(comm=self.field.cartesiancomm.comm, msg='E: {}'.format(self.energy()))
