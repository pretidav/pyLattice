"""SU(N) lattice gauge theory with the Wilson action, sampled with the Cabibbo-Marinari heat bath, Metropolis or overrelaxation
sweeps. N = 2 is the SU(2) theory: the heat bath is then the exact one and the Cabibbo-Marinari subgroup is the whole group.

    S = beta sum_{x, mu<nu} (1 - (1/N) Re tr P_{mu nu}(x)),      weight exp(-S)

The field is a pylattice.theories.gauge.field.SUNField (an SU2Field for N = 2): one SU(N) matrix on each link; its plaquettes, staples,
loops and action are computed there with the lattice operations, so nothing here depends on how the lattice is split over the
ranks. Run it from the command line, e.g.

    mpiexec -np 4 python -m pylattice.theories.gauge.sun --nc=3 --grid=8x8x8x8 --mpigrid=2x2x1x1 --beta=5.7 --steps=500 \\
        --thermalization=100 --algorithm=heatbath --overrelax=4

which writes the same files as the Ising driver, named su3_grid8x8x8x8_mpi2x2x1x1_b5.7: _meas.json with the observables and the
acceptance, _<sweep>.hdf5 checkpoints (in ./checkpoints), .log
and can restart from a checkpoint (--start-from).

The updates. The links of one direction mu on the sites of one parity (checkerboard) are updated at the same time: two such links
never belong to the same plaquette, so the staple S of each of them (the sum of the products of the other links of the plaquettes
that contain it) does not contain a link that is being updated, and all of them can be treated independently. The weight of a link U
is exp((beta/N) Re tr(U S)). Three updates keep it:

    heatbath        Cabibbo-Marinari: the SU(2) heat bath (pylattice.core.groups.su2) of each of the N (N-1) / 2 SU(2) subgroups
                    (i, j) in turn, on the 2x2 block (i, j) of U S (pylattice.core.groups.su.cabibbo_marinari_link). Each step
                    draws exactly from the conditional weight of its subgroup, so the weight is kept, and together the subgroups
                    reach every element of SU(N). Always accepted, no `eps`. For N = 2 it is the exact heat bath of the link
    metropolis      a new link X U, X a random SU(N) element at a distance `eps` from the identity (a symmetric proposal), accepted
                    with probability min(1, exp(-dS)), dS = -(beta/N) Re tr((U' - U) S); `hits` proposals per link and visit with
                    the same staple. The acceptance is measured
    overrelaxation  in each SU(2) subgroup (i, j) in turn, U -> X U with X = (W^+)^2, kW the SU(2) part of the block (i, j) of U S
                    (pylattice.core.groups.su.cabibbo_marinari_overrelax): the action does not change, so it is always accepted, and
                    it moves the links far. It is not ergodic alone: a sweep does one heat bath or Metropolis update followed by
                    `overrelax` overrelaxation sweeps.

The acceptance of a sweep is that of the Metropolis proposals (1 for the heat bath); overrelaxation is not counted. The local lattice
must have even extents, for the checkerboard to be the same on all the ranks (as for the even/odd update of the Ising model).

Observables, measured during the run: plaq_re and plaq_im (the average plaquette (1/N) tr P, whose imaginary part is zero up to
rounding), action (the Wilson action per site) and the acceptance. The expensive ones (spatial and temporal plaquettes, Polyakov loop,
Wilson loops, Creutz ratios) are measured offline on the stored configurations, with pylattice.theories.gauge.sun_meas.
"""
import argparse
import math

import numpy as np

from pylattice.core import montecarlo
from pylattice.core.cli import Logger, Parser
from pylattice.core.driver import run_simulation
from pylattice.core.grid import LatticeComplexMatrix
from pylattice.core.groups import su, su2 as group
from pylattice.core.parallel import CartesianComm
from pylattice.theories.gauge.field import gauge_field, load_links


class SUNModel:
    """Wilson action SU(N) gauge theory on a SUNField, with the sweeps above.

    beta       inverse coupling, 2N/g^2 (>= 0)
    algorithm  'metropolis' or 'heatbath': the update made once per sweep
    overrelax  overrelaxation sweeps after it, in every sweep
    eps        (metropolis) size of the proposal: the distance of X from the identity, in (0, 1]. Small: most proposals are
               accepted but the links move little; large: the opposite. Choose it for an acceptance of about 0.4 to 0.6
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
            raise ValueError("the sweeps update the sites in a checkerboard: {}".format(error))
        self.field, self.log, self.beta, self.eps, self.hits = field, logger, float(beta), float(eps), int(hits)
        self.algorithm, self.overrelax = algorithm, int(overrelax)
        self.N = field.N
        self.rng = rng if rng is not None else field.cartesiancomm.rng(0)
        self.sites = int(np.prod(field.grid))                  # on this rank
        self.global_sites = self.sites * int(np.prod(field.cartesiancomm.mpigrid))

    @property
    def comm(self):
        return self.field.cartesiancomm.comm

    def proposal(self, link, rng):
        """X U for every link of the lattice `link` (a LatticeComplexMatrix), X at a distance eps from the identity"""
        X = LatticeComplexMatrix(grid=link.grid, cartesiancomm=link.cartesiancomm, N=self.N)
        X.value = self.field.near_identity(self.eps, rng, size=self.sites)
        return X * link

    def link_action_change(self, link, proposed, staple):
        """The change of the action when every link of `link` is replaced by `proposed` on its own, the others
        left as they are: -(beta/N) Re tr((U' - U) S) (a LatticeReal). All the arguments are LatticeComplexMatrix."""
        return ((proposed - link) * staple).trace().re() * (-self.beta / self.N)

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
        """The heat bath update of the links of direction mu on the sites of `parity`: the exact heat bath for N = 2, the
        Cabibbo-Marinari one (all the SU(2) subgroups in turn) otherwise; each link keeps the weight exp((beta/N) Re tr(U S)).
        Returns the number updated on this rank"""
        link = self.field.peek_index(mu)
        staple = self.field.Staple(mu).value
        if self.N == 2:
            new = group.heatbath_link(staple, self.beta, rng)                # independent of the old link
        else:
            new = su.cabibbo_marinari_link(link.value, staple, self.beta / self.N, rng)
        link.value = np.where(parity[:, None, None], new, link.value)
        self.field.poke_index(mu, link)
        return int(parity.sum())

    def overrelaxation_update(self, mu, parity):
        """The overrelaxation of the links of direction mu on the sites of `parity`, in each SU(2) subgroup in turn: Re tr(U S)
        does not change, so neither does the action. For N = 2, U -> W^+ U^+ W^+ with S = k W, which is its own inverse"""
        link = self.field.peek_index(mu)
        staple = self.field.Staple(mu).value
        if self.N == 2:
            _, W = group.su2_decompose(staple)
            Wdag = np.conj(np.swapaxes(W, -1, -2))
            new = Wdag @ np.conj(np.swapaxes(link.value, -1, -2)) @ Wdag
        else:
            new = su.cabibbo_marinari_overrelax_link(link.value, staple)
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
        self.field.unitarize()                              # no drift from SU(N) after many products
        hits = self.hits if self.algorithm == 'metropolis' else 1
        return accepted, self.field.Nd * self.sites * hits

    def observables(self):
        """The cheap observables of the configuration, global values (a collective call): see plaquette_observables.
        The others (loops) are measured offline, on the stored configurations: pylattice.theories.gauge.sun_meas."""
        return plaquette_observables(self.field, self.beta)


def plaquette_observables(field, beta):
    """The observables of the plaquette of the gauge field `field` (a collective call):
        plaq_re, plaq_im  the average plaquette (1/N) tr P over all the sites and planes; the imaginary part is zero up to
                          rounding
        action            the Wilson action per site, summed over the planes: Nd (Nd-1)/2 beta (1 - plaq_re)"""
    plaquette = field.Plaq()
    sites = int(np.prod(field.grid) * np.prod(field.cartesiancomm.mpigrid))
    return {'plaq_re': float(plaquette.real), 'plaq_im': float(plaquette.imag),
            'action': float(field.Action(beta)) / sites}


# ----------------------------------------------------------------------------- the command line
def _eps(text):
    try:
        eps = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError("'{}' is not a number".format(text))
    if not 0 < eps <= 1:
        raise argparse.ArgumentTypeError("eps must be in (0, 1], got {}".format(text))
    return eps


def _overrelax(text):
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("'{}' is not an integer".format(text))
    if n < 0:
        raise argparse.ArgumentTypeError("overrelax cannot be negative, got {}".format(n))
    return n


def _hits(text):
    try:
        hits = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("'{}' is not an integer".format(text))
    if hits < 1:
        raise argparse.ArgumentTypeError("hits must be at least 1, got {}".format(hits))
    return hits


def _colors(text):
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("'{}' is not an integer".format(text))
    if n < 2:
        raise argparse.ArgumentTypeError("the number of colors must be at least 2, got {}".format(n))
    return n


DEFAULT_BETA = {2: 2.3, 3: 5.7}            # near the deconfinement transition of the 4-dimensional lattice; larger N: give --beta


class SUNParser(Parser):
    """The command line of the common options (see Parser), with a 4-dimensional lattice as default, the number of colors --nc
    (default 3) and --algorithm, --overrelax, --eps, --hits, --init. The default of --beta depends on N (DEFAULT_BETA: 2.3 for
    SU(2), 5.7 for SU(3)); for larger N it must be given."""

    def add_options(self, parser):
        self._argparser = parser
        parser.set_defaults(grid='4x4x4x4', mpigrid='1x1x1x2', beta=None)
        for action in parser._actions:
            if action.dest == 'beta':
                action.help = 'inverse coupling (default: 2.3 for SU(2), 5.7 for SU(3); required for N >= 4)'
        parser.add_argument('--nc', type=_colors, default=3, metavar='N',
                            help='the number of colors: the gauge group is SU(N)')
        parser.add_argument('--algorithm', choices=('metropolis', 'heatbath'), default='heatbath',
                            help='the update of every sweep: the Cabibbo-Marinari heat bath (exact, always accepted; for '
                                 'SU(2) the exact heat bath), or Metropolis (its acceptance is measured)')
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
        self.N = self.args['nc']
        self.theory = 'su{}'.format(self.N)                 # the files are named su<N>_grid<grid>_mpi<mpigrid>_b<beta>...
        if self.beta is None:
            if self.N not in DEFAULT_BETA:
                self._argparser.error("--beta is required for SU({}): there is no default".format(self.N))
            self.beta = self.args['beta'] = DEFAULT_BETA[self.N]
        self.eps, self.hits, self.init = self.args['eps'], self.args['hits'], self.args['init']
        self.algorithm, self.overrelax = self.args['algorithm'], self.args['overrelax']

    def extra_info(self):
        rows = [('gauge group', 'SU({})'.format(self.N)), ('algorithm', self.algorithm),
                ('overrelaxation sweeps', self.overrelax)]
        if self.algorithm == 'metropolis':
            rows += [('proposal eps', self.eps), ('hits per link', self.hits)]
        return rows + [('init', self.init)]


def main(argv=None):
    """Command line driver, e.g.
    mpiexec -np 4 python -m pylattice.theories.gauge.sun --nc=3 --grid=8x8x8x8 --mpigrid=2x2x1x1 --beta=5.7 --steps=500
    The run: the parser of the options, the logger with its banner, the field (cold, hot or loaded from a checkpoint), the
    model and run_simulation; the files are named su<N>_grid<grid>_mpi<mpigrid>_b<beta> followed by _meas.json, .log or _<sweep>.hdf5
    """
    PP = SUNParser(argv)
    CC = CartesianComm(mpigrid=PP.mpigrid)
    LL = Logger(debug=PP.debug, logfile=PP.log_filename() is not None, filename=PP.log_filename() or 'run.log')
    LL.banner(PP, CC.comm)                        # logo, system and run information, in boxes
    rng = CC.rng(PP.seed)                         # an independent stream on every rank

    field = gauge_field(PP.grid, CC, Nd=len(PP.grid), N=PP.N, rng=rng)
    if PP.start_from:
        step = load_links(field, PP.start_from, log=LL.log.info, rng=rng)
        LL.log.info("restart: running {} new sweeps from the configuration saved at step {}; "
                    "the sweep counter starts afresh".format(PP.steps, step))
    elif PP.init == 'hot':
        field.random_config(rng)
    model = SUNModel(field=field, logger=LL.log, beta=PP.beta, eps=PP.eps, hits=PP.hits, rng=rng,
                     algorithm=PP.algorithm, overrelax=PP.overrelax)

    run_simulation(model, PP, LL)                     # sweeps, measurement file, checkpoints, run log, summary


if __name__ == '__main__':
    main()
