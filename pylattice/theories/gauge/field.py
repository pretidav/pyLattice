"""Gauge fields: one group element on each link of the lattice. SU(N): SUNField and its SU(2) special case; SO(N): SONField.

SUNField is the field for any N: the links, a cold and a hot start, the nearest-SU(N) projection, gauge transformations,
and the staples, plaquettes, Wilson loops, Polyakov loops and Wilson action, computed with the lattice operations (shifts,
products, traces, reductions), so they are the same however the lattice is split over the ranks. SU2Field is the case N = 2,
with the quaternionic (Creutz) representation of pylattice.core.groups.su2 for the operations that have a faster form there.
SONField is the same for the real orthogonal links of SO(N). The observables and the gauge transformation are written once, in
pylattice.core.groups.gauge_observables. The group and algebra operations are in pylattice.core.groups.su (SU(N), with the SU(2) blocks of Cabibbo-Marinari),
pylattice.core.groups.so (SO(N), with the SO(2) rotations) and
pylattice.core.groups.su2.

All the updates act on the whole local lattice at once, and draw from a numpy Generator: the one given to the field (default:
the independent stream of this rank, CartesianComm.rng(0)) or the one passed to the method. The sweeps that use the field are in
pylattice.theories.gauge.sun.
"""
import numpy as np

from pylattice.core.grid import LatticeReal, LatticeVectorComplexMatrix, LatticeVectorRealMatrix
from pylattice.core.groups import so, su, su2
from pylattice.core.groups.gauge_observables import GaugeLinksMixin


class SUNField(GaugeLinksMixin, LatticeVectorComplexMatrix):
    """value has shape (sites, Nd, N, N): the link from each site in each of the Nd directions, an SU(N) matrix (double
    precision, unlike the other lattices). Nd is the number of dimensions of the lattice. For N = 2 see SU2Field."""

    def __init__(self, grid, cartesiancomm, Nd, N, rng=None):
        if int(N) != N or N < 2:
            raise ValueError("N must be an integer of at least 2, got {}".format(N))
        super().__init__(grid=grid, cartesiancomm=cartesiancomm, Nd=Nd, N=int(N))
        self.Nd = Nd
        self.cartesiancomm = cartesiancomm
        self.N = int(N)
        self.adj_dim = self.N * self.N - 1
        self.rng = rng if rng is not None else cartesiancomm.rng(0)
        self.value = self.trivial_config()

    def trivial_config(self):
        """every link the identity: the cold start (returns the array)"""
        out = np.zeros(self.value.shape, dtype=complex)
        out[...] = np.eye(self.N)
        return out

    def random_config(self, rng=None):
        """every link an independent Haar-uniform element of SU(N): the hot start"""
        rng = rng if rng is not None else self.rng
        self.value = su.random_su(self.N, rng, size=self.value.shape[:2])

    def unitarize(self):
        """replace every link by the nearest element of SU(N), to remove the rounding errors of many products. A link
        that is in SU(N) is unchanged."""
        self.value = su.project_su(self.value)

    def near_identity(self, eps, rng=None, size=None):
        """independent SU(N) elements at a distance eps from the identity (see su.near_identity_su): one for every link, an
        array of shape (sites, Nd, N, N), or with size (e.g. the number of sites, for one direction) of shape size + (N, N)"""
        return su.near_identity_su(self.N, eps, rng if rng is not None else self.rng,
                                   size=self.value.shape[:2] if size is None else size)

    def metropolis_update(self, eps=0.001, rng=None):
        """Multiplies every link on the left by an independent element at a distance eps from the identity. This is the
        proposal of a Metropolis step, without the acceptance test (see pylattice.theories.gauge.sun for the sweeps): every
        proposal is kept, a random walk of the links, which stays in SU(N)."""
        self.value = self.near_identity(eps, rng) @ self.value

    def random_gauge_transformation(self, rng=None):
        """an independent Haar-uniform SU(N) element on every site, shape (sites, N, N)"""
        rng = rng if rng is not None else self.rng
        return su.random_su(self.N, rng, size=self.value.shape[:1])


class SU2Field(SUNField):
    """The SU(2) gauge field: value has shape (sites, Nd, 2, 2). It is the SUNField for N = 2 with the quaternionic form of
    pylattice.core.groups.su2 for the hot start, the projection, the proposals and the gauge transformations (faster, and
    exactly in SU(2)); every other operation is that of SUNField."""

    def __init__(self, grid, cartesiancomm, Nd, rng=None):
        super().__init__(grid=grid, cartesiancomm=cartesiancomm, Nd=Nd, N=2, rng=rng)

    def random_config(self, rng=None):
        """every link an independent Haar-uniform element of SU(2): the hot start"""
        rng = rng if rng is not None else self.rng
        self.value = su2.to_matrix(su2.random_quaternion(rng, size=self.value.shape[:2]))

    def unitarize(self):
        """replace every link by the nearest element of SU(2), to remove the rounding errors of many
        products. A link that is in SU(2) is unchanged."""
        self.value = su2.project_su2(self.value)

    def tepid_lie_algebra_vector(self, eps, rng=None):
        """a quaternion (a0, a) at a distance eps from the identity, in a uniformly random direction"""
        return su2.near_identity_quaternion(eps, rng if rng is not None else self.rng)

    def near_identity(self, eps, rng=None, size=None):
        return su2.to_matrix(su2.near_identity_quaternion(eps, rng if rng is not None else self.rng,
                                                          size=self.value.shape[:2] if size is None else size))

    def metropolis_update(self, eps=0.001, rng=None):
        """Multiplies every link on the left by an independent element at a distance eps from the identity.
        This is the proposal of a Metropolis step, without the acceptance test (see pylattice.theories.gauge.sun for
        the sweeps): every proposal is kept, a random walk of the links, which stays in SU(2)."""
        rng = rng if rng is not None else self.rng
        self.value = self.near_identity(eps, rng) @ self.value

    def random_gauge_transformation(self, rng=None):
        """an independent Haar-uniform SU(2) element on every site, shape (sites, 2, 2)"""
        rng = rng if rng is not None else self.rng
        return su2.to_matrix(su2.random_quaternion(rng, size=self.value.shape[:1]))


class SONField(GaugeLinksMixin, LatticeVectorRealMatrix):
    """value has shape (sites, Nd, N, N): the link from each site in each of the Nd directions, a real orthogonal matrix with
    determinant 1 (double precision). The observables are those of GaugeLinksMixin, the same as for SU(N) (the traces are real, so
    the imaginary part of Plaq and of every loop is exactly 0); the group operations are those of pylattice.core.groups.so."""

    def __init__(self, grid, cartesiancomm, Nd, N, rng=None):
        if int(N) != N or N < 2:
            raise ValueError("N must be an integer of at least 2, got {}".format(N))
        super().__init__(grid=grid, cartesiancomm=cartesiancomm, Nd=Nd, N=int(N))
        self.Nd = Nd
        self.cartesiancomm = cartesiancomm
        self.N = int(N)
        self.adj_dim = self.N * (self.N - 1) // 2
        self.rng = rng if rng is not None else cartesiancomm.rng(0)
        self.value = self.trivial_config()

    def trivial_config(self):
        """every link the identity: the cold start (returns the array)"""
        out = np.zeros(self.value.shape)
        out[...] = np.eye(self.N)
        return out

    def random_config(self, rng=None):
        """every link an independent Haar-uniform element of SO(N): the hot start"""
        rng = rng if rng is not None else self.rng
        self.value = so.random_so(self.N, rng, size=self.value.shape[:2])

    def unitarize(self):
        """replace every link by the nearest element of SO(N), to remove the rounding errors of many products (the name is
        that of SUNField, so a theory can call it for either group). A link that is in SO(N) is unchanged."""
        self.value = so.project_so(self.value)

    def near_identity(self, eps, rng=None, size=None):
        """independent SO(N) elements at a distance eps from the identity (see so.near_identity_so): one for every link, an array
        of shape (sites, Nd, N, N), or with size of shape size + (N, N)"""
        return so.near_identity_so(self.N, eps, rng if rng is not None else self.rng,
                                   size=self.value.shape[:2] if size is None else size)

    def metropolis_update(self, eps=0.001, rng=None):
        """Multiplies every link on the left by an independent element at a distance eps from the identity: the proposal of a
        Metropolis step, without the acceptance test. Stays in SO(N)."""
        self.value = self.near_identity(eps, rng) @ self.value

    def random_gauge_transformation(self, rng=None):
        """an independent Haar-uniform SO(N) element on every site, shape (sites, N, N)"""
        rng = rng if rng is not None else self.rng
        return so.random_so(self.N, rng, size=self.value.shape[:1])


def gauge_field(grid, cartesiancomm, Nd, N, rng=None, group='su'):
    """the field of links of the gauge group: group 'su' gives an SU2Field for N = 2 (quaternionic fast paths) and a SUNField
    otherwise, group 'so' a SONField"""
    if group == 'so':
        return SONField(grid=grid, cartesiancomm=cartesiancomm, Nd=Nd, N=N, rng=rng)
    if group != 'su':
        raise ValueError("the gauge group must be 'su' or 'so', got {!r}".format(group))
    if N == 2:
        return SU2Field(grid=grid, cartesiancomm=cartesiancomm, Nd=Nd, rng=rng)
    return SUNField(grid=grid, cartesiancomm=cartesiancomm, Nd=Nd, N=N, rng=rng)


def load_links(field, filename, log=None, rng=None, label='restart', tol=1e-6):
    """Replaces the links of `field` by those of a checkpoint (see pylattice.core.hdf5io.load_into): the file must hold a field
    of this lattice and only SU(N) matrices (SO(N) matrices for a SONField). Returns the step it was saved at, if recorded. With rng its state is restored
    from the file too, when the layout of the ranks is the same: the run then continues exactly. A collective call."""
    from pylattice.core.hdf5io import load_into, restore_rng        # h5py is needed only to read a checkpoint
    log = log if log is not None else (lambda line: None)
    name = '{}({})'.format('SO' if isinstance(field, SONField) else 'SU', field.N)
    old = field.value.copy()
    step = load_into(field, filename, log, label)
    unitary = np.abs(field.value @ np.conj(np.swapaxes(field.value, -1, -2)) - np.eye(field.N)).max(axis=(-1, -2))
    special = np.abs(np.linalg.det(field.value) - 1)
    wrong = LatticeReal(grid=field.grid, cartesiancomm=field.cartesiancomm)
    wrong.value = ((unitary > tol) | (special > tol)).any(axis=-1).astype(np.float64)
    bad = int(wrong.reducesum())                                   # the sites with a link that is not in the group, on all ranks
    if bad:
        field.value[...] = old
        raise ValueError("cannot restart from '{}': {} sites have links that are not {} matrices, this is not an {} "
                         "configuration".format(filename, bad, name, name))
    log(label + ": all the links are {} matrices".format(name))
    if rng is not None:
        restore_rng(rng, filename, field.cartesiancomm.comm, field.cartesiancomm.mpigrid, log)
    return step
