"""SU(2) lattice gauge theory with the Wilson action, sampled with Metropolis or heat bath sweeps, optionally mixed with
overrelaxation.

    S = beta sum_{x, mu<nu} (1 - (1/2) Re tr P_{mu nu}(x)),      weight exp(-S)

The field is a pylattice.theories.gauge.sun.SU2Field (one SU(2) matrix on each link); its plaquettes, staples, loops and
action are computed there with the lattice operations, so nothing here depends on how the lattice is split over the
ranks. Run it from the command line, e.g.

    mpiexec -np 4 python -m pylattice.theories.gauge.su2 --grid=8x8x8x8 --mpigrid=2x2x1x1 --beta=2.3 --steps=500 \\
        --thermalization=100 --algorithm=heatbath --overrelax=4

which writes the same files as the Ising driver (meas.json with the observables and the acceptance; checkpoints;
run.log) and can restart from a checkpoint (--start-from).

The updates. The links of one direction mu on the sites of one parity (checkerboard) are updated at the same time:
two such links never belong to the same plaquette, so the staple S of each of them (the sum of the products of the
other links of the plaquettes that contain it) does not contain a link that is being updated, and all of them can be
treated independently. The weight of a link U is exp((beta/2) Re tr(U S)). Three updates keep it:

    metropolis      a new link X U, X a random SU(2) element at a distance `eps` from the identity (a symmetric
                    proposal), accepted with probability min(1, exp(-dS)), dS = -(beta/2) Re tr((U' - U) S);
                    `hits` proposals per link and visit with the same staple. The acceptance is measured.
    heatbath        U drawn exactly from its weight (Creutz / Kennedy-Pendleton, pylattice.core.groups.su2):
                    always accepted, no `eps`, and decorrelates much faster
    overrelaxation  S = k W with W in SU(2): U' = W^+ U^+ W^+. It leaves the weight exactly as it is (the action does
                    not change) and moves the link far, so it is always accepted. It is not ergodic alone: a sweep
                    does one Metropolis or heat bath update followed by `overrelax` overrelaxation sweeps.

The acceptance of a sweep is that of the Metropolis proposals (1 for the heat bath); overrelaxation is not counted.
The local lattice must have even extents, for the checkerboard to be the same on all the ranks (as for the even/odd
update of the Ising model).

Observables, measured during the run: plaq_re and plaq_im (the average plaquette (1/2) tr P, whose imaginary part is zero
up to rounding), action (the Wilson action per site) and the acceptance. The expensive ones (spatial and temporal
plaquettes, Polyakov loop, Wilson loops, Creutz ratios) are measured offline on the stored configurations, with
pylattice.theories.gauge.su2_meas.
"""
import math

import numpy as np

from pylattice.core import montecarlo
from pylattice.core.cli import Logger, Parser
from pylattice.core.driver import run_simulation
from pylattice.core.grid import LatticeComplexMatrix, LatticeReal
from pylattice.core.groups import su2 as group
from pylattice.core.parallel import CartesianComm
from pylattice.theories.gauge.sun import SU2Field


class SU2Model:
    """Wilson action SU(2) gauge theory on an SU2Field, with the Metropolis sweep above.

    beta       inverse coupling, 4/g^2 (>= 0)
    algorithm  'metropolis' or 'heatbath': the update made once per sweep
    overrelax  overrelaxation sweeps after it, in every sweep
    eps        (metropolis) size of the proposal: the distance of X from the identity, in (0, 1]. Small: most proposals
               are accepted but the links move little; large: the opposite. Choose it for an acceptance of 0.4 to 0.6
    hits       (metropolis) proposals per link and per visit
    rng        the numpy Generator of this rank for the updates (default: its stream for seed 0, see CartesianComm.rng)
    """

    def __init__(self, field, logger, beta, eps=0.4, hits=1, rng=None, algorithm='metropolis', overrelax=0):
        if algorithm not in ('metropolis', 'heatbath'):
            raise ValueError("algorithm must be 'metropolis' or 'heatbath', not {!r}".format(algorithm))
        if int(overrelax) != overrelax or overrelax < 0:
            raise ValueError("overrelax must be an integer of at least 0, got {}".format(overrelax))
        if not (math.isfinite(beta) and beta >= 0):
            raise ValueError("beta must be finite and >= 0, got {}".format(beta))
        if not 0 < eps <= 1:
            raise ValueError("eps must be in (0, 1], got {}".format(eps))
        if int(hits) != hits or hits < 1:
            raise ValueError("hits must be an integer of at least 1, got {}".format(hits))
        if field.Nd < 2:
            raise ValueError("a gauge field needs at least 2 dimensions, got {}".format(field.Nd))
        try:
            self.even, self.odd = field.get_EO_idx()          # the checkerboard of the sites
        except ValueError as error:
            raise ValueError("the Metropolis sweep updates the sites in a checkerboard: {}".format(error))
        self.field, self.log, self.beta, self.eps, self.hits = field, logger, float(beta), float(eps), int(hits)
        self.algorithm, self.overrelax = algorithm, int(overrelax)
        self.rng = rng if rng is not None else field.cartesiancomm.rng(0)
        self.sites = int(np.prod(field.grid))                  # on this rank
        self.global_sites = self.sites * int(np.prod(field.cartesiancomm.mpigrid))

    @property
    def comm(self):
        return self.field.cartesiancomm.comm

    def proposal(self, link, rng):
        """X U for every link of the lattice `link` (a LatticeComplexMatrix), X at a distance eps from the identity"""
        X = LatticeComplexMatrix(grid=link.grid, cartesiancomm=link.cartesiancomm, N=2)
        X.value = group.to_matrix(group.near_identity_quaternion(self.eps, rng, size=self.sites))
        return X * link

    def link_action_change(self, link, proposed, staple):
        """The change of the action when every link of `link` is replaced by `proposed` on its own, the others
        left as they are: -(beta/2) Re tr((U' - U) S) (a LatticeReal). All the arguments are LatticeComplexMatrix."""
        return ((proposed - link) * staple).trace().re() * (-self.beta / 2)

    def metropolis_update(self, mu, parity, rng):
        """The Metropolis update of the links of direction mu on the sites of `parity` (a boolean array); returns the
        number accepted on this rank"""
        staple = self.field.Staple(mu)                      # of every link; the links of this parity are updated
        link = self.field.peek_index(mu)
        accepted = 0
        for _ in range(self.hits):
            proposed = self.proposal(link, rng)
            change = self.link_action_change(link, proposed, staple)
            accept = montecarlo.metropolis_accept(change.value, rng) & parity
            link.value = np.where(accept[:, None, None], proposed.value, link.value)
            accepted += int(accept.sum())
        self.field.poke_index(mu, link)
        return accepted

    def heatbath_update(self, mu, parity, rng):
        """The heat bath update of the links of direction mu on the sites of `parity`: each is drawn from its weight
        exp((beta/2) Re tr(U S)), whatever it was; returns the number updated on this rank"""
        link = self.field.peek_index(mu)
        new = group.heatbath_link(self.field.Staple(mu).value, self.beta, rng)
        link.value = np.where(parity[:, None, None], new, link.value)
        self.field.poke_index(mu, link)
        return int(parity.sum())

    def overrelaxation_update(self, mu, parity):
        """The overrelaxation of the links of direction mu on the sites of `parity`: U -> W^+ U^+ W^+ with S = k W. Re tr(U S)
        does not change, so neither does the action; applying it twice gives the same links back"""
        _, W = group.su2_decompose(self.field.Staple(mu).value)
        Wdag = np.conj(np.swapaxes(W, -1, -2))
        link = self.field.peek_index(mu)
        new = Wdag @ np.conj(np.swapaxes(link.value, -1, -2)) @ Wdag
        link.value = np.where(parity[:, None, None], new, link.value)
        self.field.poke_index(mu, link)

    def sweep(self, rng=None):
        """One sweep: every link is updated once with the Metropolis or heat bath update, then `overrelax` sweeps of
        overrelaxation. Returns (accepted, attempted) on this rank, those of the first update only"""
        rng = self.rng if rng is None else rng
        update = self.metropolis_update if self.algorithm == 'metropolis' else self.heatbath_update
        accepted = 0
        for mu in range(self.field.Nd):
            for parity in (self.even, self.odd):
                accepted += update(mu, parity, rng)
        for _ in range(self.overrelax):
            for mu in range(self.field.Nd):
                for parity in (self.even, self.odd):
                    self.overrelaxation_update(mu, parity)
        self.field.unitarize()                              # no drift from SU(2) after many products
        hits = self.hits if self.algorithm == 'metropolis' else 1
        return accepted, self.field.Nd * self.sites * hits

    def observables(self):
        """The cheap observables of the configuration, global values (a collective call): see plaquette_observables.
        The others (loops) are measured offline, on the stored configurations: pylattice.theories.gauge.su2_meas."""
        return plaquette_observables(self.field, self.beta)


def plaquette_observables(field, beta):
    """The observables of the plaquette of the SU2Field `field` (a collective call):
        plaq_re, plaq_im  the average plaquette (1/2) tr P over all the sites and planes; the imaginary part is zero up
                          to rounding
        action            the Wilson action per site, summed over the planes: Nd (Nd-1)/2 beta (1 - plaq_re)"""
    plaquette = field.Plaq()
    sites = int(np.prod(field.grid) * np.prod(field.cartesiancomm.mpigrid))
    return {'plaq_re': float(plaquette.real), 'plaq_im': float(plaquette.imag),
            'action': float(field.Action(beta)) / sites}


def load(field, filename, log=None, rng=None, label='restart', tol=1e-6):
    """Replaces the links of `field` by those of a checkpoint (see pylattice.core.hdf5io.load_into): the file must hold
    a field of this lattice and only SU(2) matrices. Returns the step it was saved at, if recorded. With rng its
    state is restored from the file too, when the layout of the ranks is the same: the run then continues exactly."""
    from pylattice.core.hdf5io import load_into, restore_rng        # h5py is needed only to read a checkpoint
    log = log if log is not None else (lambda line: None)
    old = field.value.copy()
    step = load_into(field, filename, log, label)
    unitary = np.abs(field.value @ np.conj(np.swapaxes(field.value, -1, -2)) - np.eye(2)).max(axis=(-1, -2))
    special = np.abs(np.linalg.det(field.value) - 1)
    not_su2 = LatticeReal(grid=field.grid, cartesiancomm=field.cartesiancomm)
    not_su2.value = ((unitary > tol) | (special > tol)).any(axis=-1).astype(np.float64)
    bad = int(not_su2.reducesum())                              # the sites with a link that is not in SU(2), on all ranks
    if bad:
        field.value[...] = old
        raise ValueError("cannot restart from '{}': {} sites have links that are not SU(2) matrices, this is not an "
                         "SU(2) configuration".format(filename, bad))
    log(label + ": all the links are SU(2) matrices")
    if rng is not None:
        restore_rng(rng, filename, field.cartesiancomm.comm, field.cartesiancomm.mpigrid, log)
    return step


class SU2Parser(Parser):
    """The command line of the common options (see Parser), with a 4-dimensional lattice and beta = 2.3 as defaults,
    and --eps, --hits, --init."""

    def add_options(self, parser):
        parser.set_defaults(grid='4x4x4x4', mpigrid='1x1x1x2', beta=2.3)
        parser.add_argument('--algorithm', choices=('metropolis', 'heatbath'), default='metropolis',
                            help='the update of every sweep: Metropolis (its acceptance is measured), or the exact heat bath')
        parser.add_argument('--overrelax', type=_overrelax, default=0, metavar='N',
                            help='overrelaxation sweeps after the update of every sweep (they leave the action '
                                 'unchanged and decorrelate the links; 3-5 is usual with the heat bath)')
        parser.add_argument('--eps', type=_eps, default=0.4,
                            help='(metropolis) size of the proposal, in (0, 1]: choose it for an acceptance of 0.4-0.6')
        parser.add_argument('--hits', type=_hits, default=1, help='(metropolis) proposals per link and visit')
        parser.add_argument('--init', choices=('cold', 'hot'), default='cold',
                            help='the starting configuration (unless --start-from): cold, every link the identity; '
                                 'hot, every link random')

    def __init__(self, argv=None):
        super().__init__(argv)
        self.eps, self.hits, self.init = self.args['eps'], self.args['hits'], self.args['init']
        self.algorithm, self.overrelax = self.args['algorithm'], self.args['overrelax']

    def extra_info(self):
        rows = [('algorithm', self.algorithm), ('overrelaxation sweeps', self.overrelax)]
        if self.algorithm == 'metropolis':
            rows += [('proposal eps', self.eps), ('hits per link', self.hits)]
        return rows + [('init', self.init)]


def _eps(text):
    import argparse
    try:
        eps = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError("'{}' is not a number".format(text))
    if not 0 < eps <= 1:
        raise argparse.ArgumentTypeError("eps must be in (0, 1], got {}".format(text))
    return eps


def _overrelax(text):
    import argparse
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("'{}' is not an integer".format(text))
    if n < 0:
        raise argparse.ArgumentTypeError("overrelax cannot be negative, got {}".format(n))
    return n


def _hits(text):
    import argparse
    try:
        hits = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("'{}' is not an integer".format(text))
    if hits < 1:
        raise argparse.ArgumentTypeError("hits must be at least 1, got {}".format(hits))
    return hits


def main(argv=None):
    """Command line driver, e.g.
    mpiexec -np 4 python -m pylattice.theories.gauge.su2 --grid=8x8x8x8 --mpigrid=2x2x1x1 --beta=2.3 --steps=500
    """
    PP = SU2Parser(argv)
    CC = CartesianComm(mpigrid=PP.mpigrid)
    LL = Logger(debug=PP.debug, logfile=PP.logfile is not None, filename=PP.logfile or 'run.log')
    LL.banner(PP, CC.comm)                        # logo, system and run information, in boxes
    rng = CC.rng(PP.seed)                         # an independent stream on every rank

    field = SU2Field(grid=PP.grid, cartesiancomm=CC, Nd=len(PP.grid), rng=rng)
    if PP.start_from:
        step = load(field, PP.start_from, log=LL.log.info, rng=rng)
        LL.log.info("restart: running {} new sweeps from the configuration saved at step {}; "
                    "the sweep counter starts afresh".format(PP.steps, step))
    elif PP.init == 'hot':
        field.random_config(rng)
    model = SU2Model(field=field, logger=LL.log, beta=PP.beta, eps=PP.eps, hits=PP.hits, rng=rng,
                     algorithm=PP.algorithm, overrelax=PP.overrelax)

    run_simulation(model, PP, LL, prefix='su2')       # sweeps, measurement file, checkpoints, run log, summary


if __name__ == '__main__':
    main()
