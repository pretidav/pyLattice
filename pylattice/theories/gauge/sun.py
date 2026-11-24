"""SU(2) gauge field: one SU(2) matrix on each link of the lattice, with its updates.
The group and algebra operations it uses are in pylattice.core.groups.su2 (the quaternionic
representation of Creutz, the heat bath of a link, the embedding in SU(N)).

All the updates act on the whole local lattice at once, and draw from a numpy Generator: the one given to
the field (default: the independent stream of this rank, CartesianComm.rng(0)) or the one passed to the
method. Staple, Plaq and Action are not implemented yet, so there is no Monte Carlo sweep with an
acceptance step or a heat bath yet.
"""
import numpy as np

from pylattice.core.grid import LatticeVectorComplexMatrix
from pylattice.core.groups import su2 as sun


class SU2Field(LatticeVectorComplexMatrix):
    """value has shape (sites, Nd, 2, 2): the link from each site in each of the Nd directions
    (double precision, unlike the other lattices)."""

    def __init__(self, grid, cartesiancomm, Nd, rng=None):
        super().__init__(grid=grid, cartesiancomm=cartesiancomm, Nd=Nd, N=2)
        self.Nd = Nd
        self.cartesiancomm = cartesiancomm
        self.N = 2
        self.adj_dim = self.N * self.N - 1
        self.rng = rng if rng is not None else cartesiancomm.rng(0)
        self.value = self.trivial_config()

    def trivial_config(self):
        """every link the identity: the cold start (returns the array)"""
        out = np.zeros(self.value.shape, dtype=complex)
        out[...] = np.eye(self.N)
        return out

    def random_config(self, rng=None):
        """every link an independent Haar-uniform element of SU(2): the hot start"""
        rng = rng if rng is not None else self.rng
        self.value = sun.to_matrix(sun.random_quaternion(rng, size=self.value.shape[:2]))

    def unitarize(self):
        """replace every link by the nearest element of SU(2), to remove the rounding errors of many
        products. A link that is in SU(2) is unchanged."""
        self.value = sun.project_su2(self.value)

    def tepid_lie_algebra_vector(self, eps, rng=None):
        """a quaternion (a0, a) at a distance eps from the identity, in a uniformly random direction"""
        return sun.near_identity_quaternion(eps, rng if rng is not None else self.rng)

    def metropolis_update(self, eps=0.001, rng=None):
        """Multiplies every link on the left by an independent element at a distance eps from the identity.
        This is the proposal of a Metropolis step; there is no acceptance test yet (it needs the action,
        see Action), so every proposal is kept: a random walk of the links, which stays in SU(2)."""
        rng = rng if rng is not None else self.rng
        X = sun.to_matrix(sun.near_identity_quaternion(eps, rng, size=self.value.shape[:2]))
        self.value = X @ self.value

    def Staple(self):
        raise NotImplementedError("the staples need the Wilson action, which is not written yet")

    def Plaq(self):
        raise NotImplementedError("the plaquette needs the staples, which are not written yet")

    def Action(self):
        raise NotImplementedError("the Wilson action is not written yet")
