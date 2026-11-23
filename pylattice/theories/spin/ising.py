from pylattice.core.grid import LatticeReal,LatticeVectorReal
import numpy as np 
from copy import copy
from pylattice.core.parallel import CartesianComm, pprint, pplot
from pylattice.core.cli import Parser, Logger
from rich.progress import Progress, BarColumn, MofNCompleteColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

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
        # H = -sum_<ij> s_i s_j, energy per site. local_E sums s_i * (neighbours) over the sites,
        # which counts every bond twice
        self.local_E = self.local_energy(input_field=self.field)
        return -0.5*self.local_E.average()

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
        # s_i * (sum of the 2d neighbours of i), with the neighbours taken from self.nn_field:
        # self.nn_field must be up to date with the configuration of the neighbours
        local_E = IsingField(grid=input_field.grid, cartesiancomm=input_field.cartesiancomm)
        local_E.fill_value(n=0)
        for mu in range(len(input_field.grid)): 
            local_E = local_E + (self.nn_field.peek_index(mu) + self.nn_field.peek_index(mu+len(input_field.grid))) * input_field
        return local_E
    
    def global_update(self):
        # energy change of flipping each spin, H = -sum_<ij> s_i s_j:  dE_i = 2 s_i sum_nn s_j
        #Even Update
        delta_energy = self.local_energy(self.field)*2
        delta_energy_E, _ = delta_energy.peek_EO_lattices()
        field_E, _ = self.field.peek_EO_lattices()
        acc_E = self.metropolis_test(deltaE = delta_energy_E)
        field_E.value[:]=np.where(acc_E,field_E.value*-1,field_E.value)
        self.field.poke_EO_lattices(E_lattice=field_E)

        #Odd Update (its neighbours are the even spins just updated)
        self.nn_field = self.get_nn_field()
        delta_energy = self.local_energy(self.field)*2
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
        r = np.minimum(1, np.exp(-self.beta*deltaE.value))
        acc_matrix = rng <= r 
        return acc_matrix 
        
    def run_mc(self,steps,online_obs=False,progress=True):
        """steps sweeps (each updates every spin once). online_obs: record magnetization, energy
        per site and acceptance after every sweep. progress: progress bar, drawn by rank 0 only."""
        bar = Progress(TextColumn('[progress.description]{task.description}'), BarColumn(), MofNCompleteColumn(),
                       TimeElapsedColumn(), TimeRemainingColumn(),
                       disable=not (progress and self.field.cartesiancomm.comm.Get_rank() == 0))
        with bar:
            task = bar.add_task('MONTE CARLO', total=steps)
            if online_obs==True: 
                m = self.magnetization()
                e = self.energy()
                self.mlist.append(m)
                self.elist.append(e)
                self.nlist.append(0)
            for n in range(steps): 
                acc = self.global_update()
                if online_obs==True:
                    mean_acc = self.field.Average(value=acc, dtype='float32')
                    m = self.magnetization()
                    e = self.energy()
                    self.mlist.append(m)
                    self.elist.append(e)
                    self.nlist.append(n+1)
                    self.acclist.append(float(mean_acc))
                bar.advance(task)

    def plot_mc(self): 
        if len(self.mlist)+len(self.elist)+len(self.nlist)>0:
            pplot(comm=self.field.cartesiancomm.comm, x=self.nlist, y=self.mlist, title='Magnetization size={} beta={}'.format(self.field.grid*self.field.cartesiancomm.mpigrid,self.beta), file='./mag.png')
            pplot(comm=self.field.cartesiancomm.comm, x=self.nlist, y=self.elist, title='Energy density size={} beta={}'.format(self.field.grid*self.field.cartesiancomm.mpigrid,self.beta), file='./ene.png')
            pplot(comm=self.field.cartesiancomm.comm, x=self.nlist[1:], y=self.acclist, title='Acceptance size={} beta={}'.format(self.field.grid*self.field.cartesiancomm.mpigrid,self.beta), file='./acc.png')
        else:
            pprint(comm=self.field.cartesiancomm.comm, msg='[x] NO PLOTS. "online_obs" must be set to True')


def main(argv=None):
    """Command line driver, e.g.
    mpiexec -np 4 python -m pylattice.theories.spin.ising --grid=20x20 --mpigrid=2x2 --beta=1.5 --steps=2000
    """
    PP = Parser(argv)
    CC = CartesianComm(mpigrid=PP.mpigrid)
    LL = Logger(debug=PP.debug, logfile=PP.logfile)
    local_seed = CC.parallel_RNG(seed=PP.seed)

    field = IsingField(grid=PP.grid, cartesiancomm=CC, initialization='random', seed=local_seed)
    model = IsingModel(field=field, logger=LL.log, beta=PP.beta)

    model.run_mc(steps=PP.steps, online_obs=True)
    model.plot_mc()


if __name__ == '__main__':
    main()
