from lattice.grid import LatticeReal,LatticeVectorReal
import numpy as np 
from copy import copy
from mpi.parallel import pprint, pplot

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
        self.mlist, self.elist, self.nlist, self.acclist = [],[],[], []
        self.M = self.magnetization()
        self.E = 0
        self.nn_field = self.get_nn_field() 
        self.local_E = self.local_energy(self.field)
    
    def magnetization(self): 
        return self.field.average()
    
    def energy(self): 
        self.local_E = self.local_energy(input_field=self.field)
        return self.local_E.average()

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
    
    def global_update(self):
        #Even Update 
        delta_energy = self.local_energy(self.field)*(-2)
        delta_energy_E, _ = delta_energy.peek_EO_lattices()
        field_E, _ = self.field.peek_EO_lattices()
        acc_E = self.metropolis_test(deltaE = delta_energy_E)
        field_E.value[:]=np.where(acc_E,field_E.value*-1,field_E.value)
        self.field.poke_EO_lattices(E_lattice=field_E)

        #Odd Update
        delta_energy = self.local_energy(self.field)*(-2)
        _,delta_energy_O = delta_energy.peek_EO_lattices()
        _, field_O = self.field.peek_EO_lattices()
        acc_O = self.metropolis_test(deltaE = delta_energy_O)
        field_O.value[:]=np.where(acc_O,field_O.value*-1,field_O.value)
        self.field.poke_EO_lattices(O_lattice=field_O)
        
        self.nn_field = self.get_nn_field() 
        self.local_E = self.local_energy(self.field)

        return (np.sum(acc_E)+np.sum(acc_O))

    def metropolis_test(self,deltaE):
        rng = np.random.random(size=len(deltaE.value)) 
        r = np.minimum(1, np.exp(-deltaE.value)/self.beta)    
        acc_matrix = rng <= r 
        return acc_matrix 
        
    def run_mc(self,steps,online_obs=False):
        pprint(comm=self.field.cartesiancomm.comm, msg='--- epoch: {}'.format(0))
        if online_obs==True: 
            m = self.magnetization()
            e = self.energy()
            self.mlist.append(m)
            self.elist.append(e)
            self.nlist.append(0)
        for n in range(steps): 
            acc = self.global_update()
            pprint(comm=self.field.cartesiancomm.comm, msg='--- epoch: {}'.format(n+1))
            if online_obs==True:
                mean_acc = self.field.Average(value=acc, dtype='float32')
                m = self.magnetization()
                e = self.energy()
                self.mlist.append(m)
                self.elist.append(e)
                self.nlist.append(n)
                self.acclist.append(float(mean_acc))

    def plot_mc(self): 
        if len(self.mlist)+len(self.elist)+len(self.nlist)>0:
            pplot(comm=self.field.cartesiancomm.comm, x=self.nlist, y=self.mlist, title='Magnetization size={} beta={}'.format(self.field.grid*self.field.cartesiancomm.mpigrid,self.beta), file='./mag.png')
            pplot(comm=self.field.cartesiancomm.comm, x=self.nlist, y=self.elist, title='Energy density size={} beta={}'.format(self.field.grid*self.field.cartesiancomm.mpigrid,self.beta), file='./ene.png')
            pplot(comm=self.field.cartesiancomm.comm, x=self.nlist[1:], y=self.acclist, title='Acceptance size={} beta={}'.format(self.field.grid*self.field.cartesiancomm.mpigrid,self.beta), file='./acc.png')
        else:
            pprint(comm=self.field.cartesiancomm.comm, msg='[x] NO PLOTS. "online_obs" must be set to True')