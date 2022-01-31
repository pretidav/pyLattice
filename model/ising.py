from lattice.grid import LatticeReal,LatticeVectorReal
import numpy as np 
from copy import deepcopy, copy
from mpi.parallel import pprint
import matplotlib.pyplot as plt

class IsingField(LatticeReal):
    def __init__(self,grid,cartesiancomm, initialization='cold', seed = 0): 
        super().__init__(grid,cartesiancomm)
        self.seed = seed 
        self.initialization = initialization
        self.init()  

    def init(self): 
        if self.initialization=='hot': 
            self.value[:]=np.ones(shape=self.value.shape,dtype='float32')
        if self.initialization=='cold': 
            self.value[:]=np.ones(shape=self.value.shape,dtype='float32')-2
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
        self.local_E = self.local_energy(self.field)

    def magnetization(self): 
        return self.field.average()
    
    def energy(self): 
        self.local_E = self.local_energy(input_field=self.field)
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

    def local_energy(self,input_field):
        local_E = IsingField(grid=input_field.grid, cartesiancomm=input_field.cartesiancomm)
        local_E.fill_value(n=0)
        for mu in range(len(input_field.grid)): 
            local_E = local_E + (self.nn_field.peek_index(mu) + self.nn_field.peek_index(mu+len(input_field.grid))) * input_field
        return local_E
    
    def get_boundary_idx(self):
        boundary_up=[] 
        boundary_down=[]     
        dummy_tensor_idx = np.reshape(np.array([i for i in range(np.prod(self.field.grid))]), self.field.grid)
        for mu in range(len(self.field.grid)):
            boundary_up.append(np.ndarray.flatten(self.field.pick_first_slice(tensor=dummy_tensor_idx, mu=mu)))
            boundary_down.append(np.ndarray.flatten(self.field.pick_last_slice(tensor=dummy_tensor_idx, mu=mu)))
        return boundary_up, boundary_down

    def global_update(self):
        delta_energy = self.local_energy(self.field)*(2)
        delta_energy_E,delta_energy_O = delta_energy.peek_EO_lattices()
        field_E, field_O = self.field.peek_EO_lattices()
        
        #Even Update 
        acc_E = self.metropolis_test(deltaE = delta_energy_E)
        field_E.value[:]=np.where(acc_E,field_E.value*-1,field_E.value)
        # print(acc_E)
        # print(field_E.value)
       
        delta_energy = self.local_energy(self.field)*(2)
        delta_energy_E,delta_energy_O = delta_energy.peek_EO_lattices()
        field_E, field_O = self.field.peek_EO_lattices()
       
        #Odd Update 
        acc_O = self.metropolis_test(deltaE = delta_energy_O)
        field_O.value[:]=np.where(acc_O,field_O.value*-1,field_O.value)
        # print(acc_O)
        # print(field_O.value)
        self.field.poke_EO_lattices(E_lattice=field_E,O_lattice=field_O)
        # print(self.field.value.reshape(self.field.grid))
        self.nn_field = self.get_nn_field() 
        self.local_E = self.local_energy(self.field)

        return (np.sum(acc_E)+np.sum(acc_O))/np.prod(self.field.grid)

    def metropolis_test(self,deltaE):
        rng = np.random.random(size=len(deltaE.value)) 
        r = np.minimum(1, np.exp(deltaE.value)/self.beta)    
        acc_matrix = r>rng 
        return acc_matrix 
        
    def run_mc(self,steps): 
        mlist, elist, nlist = [],[], []
        m = self.magnetization()
        e = self.energy()
        pprint(comm=self.field.cartesiancomm.comm, msg='--- epoch: {}'.format(0))
        pprint(comm=self.field.cartesiancomm.comm, msg='M: {}'.format(m))
        pprint(comm=self.field.cartesiancomm.comm, msg='E: {}'.format(e))
        mlist.append(m)
        elist.append(e/np.prod(self.field.grid))
        nlist.append(0)
        for n in range(steps): 
            acc = self.global_update()
            m = self.magnetization()
            e = self.energy()
            pprint(comm=self.field.cartesiancomm.comm, msg='--- epoch: {}'.format(n+1))
            pprint(comm=self.field.cartesiancomm.comm, msg='acceptance: {}'.format(acc))
            pprint(comm=self.field.cartesiancomm.comm, msg='M: {}'.format(m))
            pprint(comm=self.field.cartesiancomm.comm, msg='E: {}'.format(e))
            mlist.append(m)
            elist.append(e/np.prod(self.field.grid))
            nlist.append(n)
        plt.scatter(x=nlist,y=mlist)
        #plt.scatter(x=nlist,y=elist)
        plt.savefig('./ising.png')