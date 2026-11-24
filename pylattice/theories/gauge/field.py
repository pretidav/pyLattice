"""SU(N) gauge fields: one SU(N) matrix on each link of the lattice, and the SU(2) special case.

SUNField is the field for any N: the links, a cold and a hot start, the nearest-SU(N) projection, gauge transformations,
and the staples, plaquettes, Wilson loops, Polyakov loops and Wilson action, computed with the lattice operations (shifts,
products, traces, reductions), so they are the same however the lattice is split over the ranks. SU2Field is the case N = 2,
with the quaternionic (Creutz) representation of pylattice.core.groups.su2 for the operations that have a faster form there.
The group and algebra operations are in pylattice.core.groups.su (SU(N), with the SU(2) blocks of Cabibbo-Marinari) and
pylattice.core.groups.su2.

All the updates act on the whole local lattice at once, and draw from a numpy Generator: the one given to the field (default:
the independent stream of this rank, CartesianComm.rng(0)) or the one passed to the method. The sweeps that use the field are in
pylattice.theories.gauge.sun.
"""
import numpy as np

from pylattice.core.grid import LatticeComplexMatrix, LatticeReal, LatticeVectorComplexMatrix
from pylattice.core.groups import su, su2


class SUNField(LatticeVectorComplexMatrix):
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

    def gauge_transform(self, g=None, rng=None):
        """The field after the gauge transformation g (an SU(N) matrix on every site, a LatticeComplexMatrix or an
        array of shape (sites, N, N)):   U'_mu(x) = g(x) U_mu(x) g(x + mu)^+
        Returns a new field of the same type and leaves this one unchanged. With g=None a random transformation is drawn
        (from rng). Every gauge-invariant quantity (Plaq, Action, the loops) is the same on both fields."""
        if g is None:
            g = self.random_gauge_transformation(rng)
        if not isinstance(g, LatticeComplexMatrix):
            g = np.asarray(g, dtype=complex)
            if g.shape != (self.value.shape[0], self.N, self.N):
                raise ValueError("g must have shape {}, not {}".format((self.value.shape[0], self.N, self.N), g.shape))
            G = LatticeComplexMatrix(grid=self.grid, cartesiancomm=self.cartesiancomm, N=self.N)
            G.value = g
            g = G
        out = self.copy()                                      # a field of the same type; every link is replaced below
        for mu in range(self.Nd):
            out.poke_index(mu, g * self.peek_index(mu) * g.shift_forward(mu).adj())
        return out

    def _check_directions(self, *directions):
        if self.Nd < 2:
            raise ValueError("plaquettes and staples need at least 2 dimensions")
        for mu in directions:
            if not 0 <= mu < self.Nd:
                raise ValueError("direction {} is not in 0..{}".format(mu, self.Nd - 1))

    def Plaquette_field(self, mu, nu):
        """P_{mu nu}(x) = U_mu(x) U_nu(x+mu) U_mu(x+nu)^+ U_nu(x)^+ on every site (a LatticeComplexMatrix)"""
        self._check_directions(mu, nu)
        if mu == nu:
            raise ValueError("a plaquette needs two different directions")
        Umu, Unu = self.peek_index(mu), self.peek_index(nu)
        return Umu * Unu.shift_forward(mu) * Umu.shift_forward(nu).adj() * Unu.adj()

    def Staple(self, mu):
        """The sum of the 2 (Nd - 1) staples of the links in direction mu (a LatticeComplexMatrix). For nu != mu:
            U_nu(x+mu) U_mu(x+nu)^+ U_nu(x)^+   +   U_nu(x+mu-nu)^+ U_mu(x-nu)^+ U_nu(x-nu)
        so that Re tr(U_mu(x) Staple_mu(x)) is the sum of the plaquettes that contain the link."""
        self._check_directions(mu)
        Umu, staple = self.peek_index(mu), None
        for nu in (nu for nu in range(self.Nd) if nu != mu):
            Unu = self.peek_index(nu)
            Unu_mu = Unu.shift_forward(mu)
            upper = Unu_mu * Umu.shift_forward(nu).adj() * Unu.adj()
            lower = Unu_mu.shift_backward(nu).adj() * Umu.shift_backward(nu).adj() * Unu.shift_backward(nu)
            staple = upper + lower if staple is None else staple + upper + lower
        return staple

    def _extent(self, mu):
        """number of sites of the whole lattice along mu"""
        return int(self.grid[mu] * self.cartesiancomm.mpigrid[mu])

    def line(self, mu, length):
        """The product of `length` links along mu from every site, U_mu(x) U_mu(x+mu) ... U_mu(x+(length-1)mu) (a
        LatticeComplexMatrix)"""
        self._check_directions(mu)
        if length < 1:
            raise ValueError("the length of a line must be at least 1, got {}".format(length))
        moved = out = self.peek_index(mu)
        for _ in range(length - 1):
            moved = moved.shift_forward(mu)
            out = out * moved
        return out

    @staticmethod
    def _shifted(lattice, mu, steps):
        for _ in range(steps):
            lattice = lattice.shift_forward(mu)
        return lattice

    def loop_field(self, mu, nu, R, T):
        """The Wilson loop of R sites along mu and T along nu, starting at every site:
            line(mu, R)(x) line(nu, T)(x + R mu) line(mu, R)(x + T nu)^+ line(nu, T)(x)^+      (R = T = 1: a plaquette)
        a LatticeComplexMatrix. R and T cannot exceed the extent of the whole lattice along mu and nu."""
        self._check_directions(mu, nu)
        if mu == nu:
            raise ValueError("a loop needs two different directions")
        if not (1 <= R <= self._extent(mu) and 1 <= T <= self._extent(nu)):
            raise ValueError("a {}x{} loop does not fit in the lattice ({} along {}, {} along {})".format(
                R, T, self._extent(mu), mu, self._extent(nu), nu))
        a, b = self.line(mu, R), self.line(nu, T)
        return a * self._shifted(b, mu, R) * self._shifted(a, nu, T).adj() * b.adj()

    def _trace_sum(self, pairs, R=1, T=1):
        """sum over all the sites of the whole lattice and the given (mu, nu) of tr of the R x T loop, and the number
        of terms"""
        fields = [self.Plaquette_field(mu, nu) if R == T == 1 else self.loop_field(mu, nu, R, T) for mu, nu in pairs]
        total = sum((f.trace() for f in fields[1:]), fields[0].trace())
        sites = int(np.prod(self.grid) * np.prod(self.cartesiancomm.mpigrid))
        return complex(total.reducesum()), sites * len(pairs)

    def _ordered_pairs(self):
        return [(mu, nu) for mu in range(self.Nd) for nu in range(self.Nd) if mu != nu]

    def Plaq(self, pairs=None):
        """The plaquette (1/N) tr P averaged over all the sites of the whole lattice and over `pairs`, the ordered pairs
        (mu, nu) of directions to use (default: all the ordered pairs mu != nu): a complex number (1 for a cold
        field). P_{nu mu} = P_{mu nu}^+, so with both orders the imaginary part vanishes up to rounding: it is a check
        of the links and the shifts. With pairs one can separate, e.g., the spatial plaquettes from the temporal ones."""
        self._check_directions()
        total, count = self._trace_sum(self._ordered_pairs() if pairs is None else list(pairs))
        return total / (self.N * count)

    def Wilson(self, R, T):
        """The Wilson loop (1/N) tr of the R x T loops, averaged over all the sites of the whole lattice and all the
        ordered pairs mu != nu (a complex number). The reverse of a loop has the conjugate trace, so the imaginary part is
        zero up to rounding for R = T and for N = 2 (real traces); for N >= 3 and R != T it is not: the Wilson loop is its
        real part, the average of Re tr."""
        self._check_directions()
        total, count = self._trace_sum(self._ordered_pairs(), R, T)
        return total / (self.N * count)

    def Polyakov(self, mu):
        """The Polyakov loop (1/N) tr of the product of the links along the whole extent of direction mu (periodic),
        averaged over the sites: the average of its value on the lattice of the other directions (a complex number)."""
        loop = self.line(mu, self._extent(mu))
        return complex(loop.trace().reducesum()) / (self.N * int(np.prod(self.grid) * np.prod(self.cartesiancomm.mpigrid)))

    def Action(self, beta):
        """the Wilson action S = beta sum_{x, mu<nu} (1 - (1/N) Re tr P_{mu nu}(x)) of the whole lattice
        (the same value on every rank; 0 for a cold field)"""
        self._check_directions()
        total, count = self._trace_sum([(mu, nu) for mu in range(self.Nd) for nu in range(mu + 1, self.Nd)])
        return beta * (count - total.real / self.N)


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


def gauge_field(grid, cartesiancomm, Nd, N, rng=None):
    """the field of SU(N) links: an SU2Field for N = 2 (quaternionic fast paths), a SUNField otherwise"""
    if N == 2:
        return SU2Field(grid=grid, cartesiancomm=cartesiancomm, Nd=Nd, rng=rng)
    return SUNField(grid=grid, cartesiancomm=cartesiancomm, Nd=Nd, N=N, rng=rng)


def load_links(field, filename, log=None, rng=None, label='restart', tol=1e-6):
    """Replaces the links of `field` by those of a checkpoint (see pylattice.core.hdf5io.load_into): the file must hold a field
    of this lattice and only SU(N) matrices. Returns the step it was saved at, if recorded. With rng its state is restored
    from the file too, when the layout of the ranks is the same: the run then continues exactly. A collective call."""
    from pylattice.core.hdf5io import load_into, restore_rng        # h5py is needed only to read a checkpoint
    log = log if log is not None else (lambda line: None)
    name = 'SU({})'.format(field.N)
    old = field.value.copy()
    step = load_into(field, filename, log, label)
    unitary = np.abs(field.value @ np.conj(np.swapaxes(field.value, -1, -2)) - np.eye(field.N)).max(axis=(-1, -2))
    special = np.abs(np.linalg.det(field.value) - 1)
    wrong = LatticeReal(grid=field.grid, cartesiancomm=field.cartesiancomm)
    wrong.value = ((unitary > tol) | (special > tol)).any(axis=-1).astype(np.float64)
    bad = int(wrong.reducesum())                                   # the sites with a link that is not in SU(N), on all ranks
    if bad:
        field.value[...] = old
        raise ValueError("cannot restart from '{}': {} sites have links that are not {} matrices, this is not an {} "
                         "configuration".format(filename, bad, name, name))
    log(label + ": all the links are {} matrices".format(name))
    if rng is not None:
        restore_rng(rng, filename, field.cartesiancomm.comm, field.cartesiancomm.mpigrid, log)
    return step
