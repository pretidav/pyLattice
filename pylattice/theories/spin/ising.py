from pylattice.core.grid import LatticeReal
import numpy as np 
from pylattice.core.parallel import CartesianComm, pprint
from pylattice.core.cli import Parser, Logger
from pylattice.core.driver import run_simulation
from pylattice.core import montecarlo
from pylattice.core.groups.discrete import Z2

class IsingField(LatticeReal):
    """A spin on each site with values in the group Z2 (the integers +1, -1, stored as int8: one byte per spin;
    pylattice.core.groups.discrete).
    initialization: 'cold' every spin +1 (the identity of Z2, an ordered start), 'hot' every spin random, +1 or -1 with
    equal probability. seed (for 'hot'): an integer, or a numpy Generator (CartesianComm.rng(seed): one independent
    stream per rank, which the model can go on using). The same integer on every rank would give every rank the
    same block of spins."""
    def __init__(self,grid,cartesiancomm, initialization='cold', seed = 0): 
        super().__init__(grid,cartesiancomm)
        self.value = self.value.astype(Z2.elements.dtype)                   # integer spins, not floating point
        if initialization not in ('cold', 'hot'):
            raise ValueError("initialization must be 'cold' or 'hot', not {!r}".format(initialization))
        self.seed = seed 
        self.initialization = initialization
        self.init()  

    def init(self): 
        if self.initialization=='cold': 
            self.value[:]=Z2.identity(self.value.shape)
        if self.initialization=='hot': 
            self.value[:]=Z2.random(np.random.default_rng(self.seed),self.value.shape)

    def load(self, filename, log=None, rng=None, label='restart'):
        """Replaces the spins by those of a checkpoint (see pylattice.core.hdf5io.load_into). The file must
        hold a real field of this global lattice and only spins, +1 or -1. Returns the step it was saved
        at, if recorded. With rng (the Generator of this rank) its state is restored from the file too, when
        the file has it and the layout of the ranks is the same: the run then continues exactly. Do it before building the IsingModel, which reads the field when it is made."""
        from pylattice.core.hdf5io import load_into
        log = log if log is not None else (lambda line: None)
        old = self.value.copy()
        step = load_into(self, filename, log, label)
        not_ising = LatticeReal(grid=self.grid, cartesiancomm=self.cartesiancomm)
        not_ising.value = (~Z2.contains(self.value)).astype(np.float64)
        bad = int(not_ising.reducesum())                                  # how many sites hold something else, on all ranks
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

    def magnetization(self): 
        return self.field.average()

    def energy(self):
        # H = -sum_<ij> s_i s_j, energy per site. local_energy sums s_i * (neighbours) over the sites,
        # which counts every bond twice
        return -0.5*self.local_energy(self.field).average()

    def neighbour_sum(self, field):
        """the sum of the 2d nearest neighbours of every site (a LatticeReal), across the rank boundaries"""
        total = None
        for mu in range(len(field.grid)):
            both = field.shift_forward(mu) + field.shift_backward(mu)
            total = both if total is None else total + both
        return total

    def local_energy(self, field):
        """s_i * (sum of the 2d neighbours of i) on every site"""
        return field * self.neighbour_sum(field)

    def global_update(self, rng=None):
        """One sweep: the even sites, then the odd ones (their neighbours are the even spins just updated).
        A spin is flipped with the Metropolis test of the energy change dE_i = 2 s_i sum_nn s_j,
        H = -sum_<ij> s_i s_j. Returns the number of flips on this rank."""
        rng = self.rng if rng is None else rng
        accepted = 0
        for parity, half in enumerate(('E_lattice', 'O_lattice')):
            delta_energy = (self.local_energy(self.field) * 2).peek_EO_lattices()[parity]
            spins = self.field.peek_EO_lattices()[parity]
            flip = self.metropolis_test(deltaE=delta_energy, rng=rng)
            spins.value[:] = np.where(flip, Z2.multiply(spins.value, Z2.generator), spins.value)   # a flip: times the non-trivial element
            self.field.poke_EO_lattices(**{half: spins})
            accepted += np.sum(flip)
        return accepted

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
    LL = Logger(debug=PP.debug, logfile=PP.logfile is not None, filename=PP.logfile or 'run.log')
    LL.banner(PP, CC.comm)                        # logo, system and run information, in boxes
    rng = CC.rng(PP.seed)                         # an independent stream on every rank

    field = IsingField(grid=PP.grid, cartesiancomm=CC, initialization='hot', seed=rng)
    if PP.start_from:
        step = field.load(PP.start_from, log=LL.log.info, rng=rng)
        LL.log.info("restart: running {} new sweeps from the configuration saved at step {}; "
                    "the sweep counter starts afresh".format(PP.steps, step))
    model = IsingModel(field=field, logger=LL.log, beta=PP.beta, rng=rng)

    run_simulation(model, PP, LL, prefix='ising')     # sweeps, measurement file, checkpoints, run log, summary


if __name__ == '__main__':
    main()
