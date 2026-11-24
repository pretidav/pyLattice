from pylattice.core.grid import LatticeReal,LatticeVectorReal
import numpy as np 
from copy import copy
from pylattice.core.parallel import CartesianComm, pprint
from pylattice.core.cli import Parser, Logger, log_banner
from pylattice.core import montecarlo
from pylattice.core.groups.discrete import Z2
from pylattice.core.callbacks import Checkpointer, RunLog, MeasurementFile

class IsingField(LatticeReal):
    """A spin on each site with values in the group Z2 (+1, -1; pylattice.core.groups.discrete)."""
    def __init__(self,grid,cartesiancomm, initialization='cold', seed = 0): 
        super().__init__(grid,cartesiancomm)
        self.seed = seed 
        self.initialization = initialization
        self.init()  

    def init(self): 
        if self.initialization=='hot': 
            self.value[:]=Z2.identity(self.value.shape)
        if self.initialization=='cold': 
            self.value[:]=Z2.generator
        if self.initialization=='random': 
            # seed: an integer, or a numpy Generator (CartesianComm.rng(seed): one independent stream
            # per rank, which the model can go on using). The same integer on every rank would give
            # every rank the same block.
            rng = np.random.default_rng(self.seed)
            self.value[:]=Z2.random(rng,self.value.shape)

    def load(self, filename, log=None, rng=None, label='restart'):
        """Replaces the spins by those of a checkpoint (see pylattice.core.hdf5io.load_into). The file must
        hold a real field of this global lattice and only spins, +1 or -1. Returns the step it was saved
        at, if recorded. With rng (the Generator of this rank) its state is restored from the file too, when
        the file has it and the layout of the ranks is the same: the run then continues exactly. Do it before building the IsingModel, which reads the field when it is made."""
        from pylattice.core.hdf5io import load_into
        log = log if log is not None else (lambda line: None)
        old = self.value.copy()
        step = load_into(self, filename, log, label)
        bad = self.cartesiancomm.comm.allreduce(int(np.count_nonzero(~Z2.contains(self.value))))
        if bad:
            self.value[...] = old
            raise ValueError("cannot restart from '{}': {} values are not +1 or -1, this is not an Ising "
                             "configuration".format(filename, bad))
        log(label + ": all the spins are +1 or -1; magnetization {:.6f}".format(self.average()))
        if rng is not None:
            from pylattice.core.hdf5io import restore_rng
            restore_rng(rng, filename, self.cartesiancomm.comm, self.cartesiancomm.mpigrid, log)
        return step


class IsingModel():
    def __init__(self,field,logger,beta,rng=None):
        """rng: the numpy Generator of this rank for the updates (default: the stream of this rank
        for seed 0, see CartesianComm.rng); run_mc and sweep use it unless given another."""
        self.beta = beta
        self.log = logger 
        self.field = field  
        self.rng = rng if rng is not None else field.cartesiancomm.rng(0)
        self.history = montecarlo.History()
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
    
    def global_update(self,rng=None):
        rng = self.rng if rng is None else rng
        # energy change of flipping each spin, H = -sum_<ij> s_i s_j:  dE_i = 2 s_i sum_nn s_j
        #Even Update
        delta_energy = self.local_energy(self.field)*2
        delta_energy_E, _ = delta_energy.peek_EO_lattices()
        field_E, _ = self.field.peek_EO_lattices()
        acc_E = self.metropolis_test(deltaE = delta_energy_E, rng = rng)
        field_E.value[:]=np.where(acc_E,Z2.multiply(field_E.value,Z2.generator),field_E.value)   # a flip: times the non-trivial element
        self.field.poke_EO_lattices(E_lattice=field_E)

        #Odd Update (its neighbours are the even spins just updated)
        self.nn_field = self.get_nn_field()
        delta_energy = self.local_energy(self.field)*2
        _,delta_energy_O = delta_energy.peek_EO_lattices()
        _, field_O = self.field.peek_EO_lattices()
        acc_O = self.metropolis_test(deltaE = delta_energy_O, rng = rng)
        field_O.value[:]=np.where(acc_O,Z2.multiply(field_O.value,Z2.generator),field_O.value)
        self.field.poke_EO_lattices(O_lattice=field_O)
        
        self.nn_field = self.get_nn_field() 
        self.local_E = self.local_energy(self.field)

        return (np.sum(acc_E)+np.sum(acc_O))

    def metropolis_test(self,deltaE,rng=None):
        # the action is beta * H: the change of the action of a flip is beta * dE
        rng = self.rng if rng is None else rng
        return montecarlo.metropolis_accept(self.beta*deltaE.value, rng)
        
    @property
    def comm(self):
        return self.field.cartesiancomm.comm

    def sweep(self,rng=None):
        """one sweep: every spin is tried once. Returns (accepted, attempted) on this rank"""
        return self.global_update(rng), self.field.value.size

    def observables(self):
        """magnetization and energy per site (global values)"""
        return {'m': float(self.magnetization()), 'e': float(self.energy())}

    def run_mc(self,steps,online_obs=False,progress=True,thermalization=0,measure_every=1,callbacks=(),rng=None):
        """steps sweeps (each tries every spin once), after thermalization sweeps that are not
        measured. online_obs: record the magnetization and the energy per site (mlist, elist) at
        sweeps nlist, and the acceptance (acclist), every measure_every sweeps; the same data is in
        self.history. progress: progress bar, drawn by rank 0 only. callbacks: see
        pylattice.core.callbacks. rng: the generator of this rank (default self.rng).
        See pylattice.core.montecarlo."""
        self.history = montecarlo.run(self, steps, thermalization=thermalization, measure_every=measure_every,
                                      progress=progress, measure=online_obs, callbacks=callbacks, rng=rng)
        if online_obs:
            self.nlist.extend(self.history.sweeps)
            self.mlist.extend(self.history.values['m'])
            self.elist.extend(self.history.values['e'])
            self.acclist.extend(self.history.acceptance)


def main(argv=None):
    """Command line driver, e.g.
    mpiexec -np 4 python -m pylattice.theories.spin.ising --grid=20x20 --mpigrid=2x2 --beta=1.5 --steps=2000
    """
    PP = Parser(argv)
    CC = CartesianComm(mpigrid=PP.mpigrid)
    LL = Logger(debug=PP.debug, logfile=PP.logfile)
    log_banner(LL.log, PP, CC.comm)               # logo, system and run information
    rng = CC.rng(PP.seed)                         # an independent stream on every rank

    field = IsingField(grid=PP.grid, cartesiancomm=CC, initialization='random', seed=rng)
    if PP.start_from:
        step = field.load(PP.start_from, log=LL.log.info, rng=rng)
        LL.log.info("restart: running {} new sweeps from the configuration saved at step {}; "
                    "the sweep counter starts afresh".format(PP.steps, step))
    model = IsingModel(field=field, logger=LL.log, beta=PP.beta, rng=rng)

    callbacks = [MeasurementFile('meas.json')]          # every measurement, as soon as it is made
    if PP.checkpoint_every:
        callbacks.append(Checkpointer(PP.checkpoint_every, directory='checkpoints', prefix='ising', final=True))
    if PP.runlog_every:
        callbacks.append(RunLog(PP.runlog_every, filename='run.log'))
    model.run_mc(steps=PP.steps, online_obs=True, callbacks=callbacks)
    LL.log.info('done: measurements in meas.json (print and plot them with: python -m pylattice.utils.plot_meas meas.json)')


if __name__ == '__main__':
    main()
