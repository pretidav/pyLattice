"""SO(N) lattice gauge theory with the Wilson action, sampled with the Cabibbo-Marinari heat bath, Metropolis or overrelaxation
sweeps. The same theory as pylattice.theories.gauge.sun with the real orthogonal matrices of determinant 1 as links:

    S = beta sum_{x, mu<nu} (1 - (1/N) tr P_{mu nu}(x)),      weight exp(-S),      beta = 2N / (a g^2)

(the plaquette action and the coupling of the SO(N) literature, e.g. Bursa-Lau-Teper, Athenodorou-Lau-Teper). The field is a
pylattice.theories.gauge.field.SONField; the staples, plaquettes, loops and action are those of every gauge group
(pylattice.core.groups.gauge_observables), the sweeps are those of SUNModel (checkerboard, the same weight exp((beta/N) tr(U S)) of a
link) with the group-specific steps of pylattice.core.groups.so, run from the command line

    mpiexec -np 4 python -m pylattice.theories.gauge.son --nc=3 --grid=8x8x8x8 --mpigrid=2x2x1x1 --beta=6.0 --steps=500 \\
        --thermalization=100 --algorithm=heatbath --overrelax=4

which writes the files so3_grid8x8x8x8_mpi2x2x1x1_b6 followed by _meas.json, _<sweep>.hdf5 (in ./checkpoints) and .log.

The updates (see SUNModel for the checkerboard and the staple S of a link)
    heatbath        Cabibbo-Marinari: SO(N) is updated one rotation plane (i, j) at a time, N (N-1) / 2 SO(2) subgroups that generate
                    SO(N). In a plane the weight of U -> R(theta) U is exp(c a cos(theta - theta0)), c = beta/N, with (a, theta0)
                    of the block (i, j) of U S (pylattice.core.groups.so.so2_block): a von Mises distribution, drawn exactly. Always
                    accepted. For N = 2 (SO(2) = U(1)) there is a single plane and it is the exact heat bath of the link
    metropolis      a new link X U, X at a distance `eps` from the identity, accepted with probability min(1, exp(-dS))
    overrelaxation  in each plane in turn theta -> 2 theta0 - theta (pylattice.core.groups.so.cabibbo_marinari_overrelax): the
                    action does not change, always accepted, not ergodic alone: a sweep does one heat bath or Metropolis update
                    followed by `overrelax` overrelaxation sweeps.

Observables during the run: plaq_re, plaq_im (identically 0 for real links), action and the acceptance; the loops are measured
offline with pylattice.theories.gauge.son_meas. There is no default beta except for SO(2): the physical range depends on N and on the
dimension (SO(3) is the adjoint-representation SU(2) theory, which has a bulk transition at weak coupling in 4 dimensions); give --beta.
"""
from pylattice.core.groups import so
from pylattice.theories.gauge.sun import SUNModel, SUNParser, main as run

DEFAULT_BETA = {2: 1.0}                    # SO(2) = U(1) in 4 dimensions has its transition at beta about 1.0; larger N: give --beta


class SONModel(SUNModel):
    """Wilson action SO(N) gauge theory on a SONField: the sweeps of SUNModel with the steps of the rotation planes (the arguments
    are those of SUNModel). The weight of a link is exp((beta/N) tr(U S))."""

    def heatbath_links(self, links, staple, rng):
        return so.cabibbo_marinari_link(links, staple, self.beta / self.N, rng)

    def overrelaxed_links(self, links, staple):
        return so.cabibbo_marinari_overrelax_link(links, staple)


class SONParser(SUNParser):
    """The command line of SUNParser for the gauge group SO(N): --nc is the N of SO(N) (default 3), the files are named so<N>_..."""
    prefix, label, default_beta = 'so', 'SO', DEFAULT_BETA
    beta_help = 'inverse coupling 2N/g^2 (default: 1.0 for SO(2); required for N >= 3)'


def main(argv=None):
    """Command line driver, e.g.
    mpiexec -np 4 python -m pylattice.theories.gauge.son --nc=3 --grid=8x8x8x8 --mpigrid=2x2x1x1 --beta=6.0 --steps=500
    Files: so<N>_grid<grid>_mpi<mpigrid>_b<beta> followed by _meas.json, .log or _<sweep>.hdf5"""
    run(argv, parser_class=SONParser, model_class=SONModel, group_name='so')


if __name__ == '__main__':
    main()
