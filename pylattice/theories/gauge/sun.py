"""SU(2) gauge field: one SU(2) matrix on each link of the lattice, with its updates.
The group and algebra operations it uses are in pylattice.core.groups.su2 (the quaternionic
representation of Creutz, the heat bath of a link, the embedding in SU(N)).

All the updates act on the whole local lattice at once, and draw from a numpy Generator: the one given to
the field (default: the independent stream of this rank, CartesianComm.rng(0)) or the one passed to the
method. Staples, plaquettes, Wilson loops, Polyakov loops and the Wilson action are computed with the lattice
operations (shifts, products, traces, reductions), so they are the same however the lattice is split over the ranks;
the sweeps that use them are in pylattice.theories.gauge.su2.
"""
import numpy as np

from pylattice.core.grid import LatticeComplexMatrix, LatticeVectorComplexMatrix
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

    def random_gauge_transformation(self, rng=None):
        """an independent Haar-uniform SU(2) element on every site, shape (sites, 2, 2)"""
        rng = rng if rng is not None else self.rng
        return sun.to_matrix(sun.random_quaternion(rng, size=self.value.shape[:1]))

    def gauge_transform(self, g=None, rng=None):
        """The field after the gauge transformation g (an SU(2) matrix on every site, a LatticeComplexMatrix or an
        array of shape (sites, 2, 2)):   U'_mu(x) = g(x) U_mu(x) g(x + mu)^+
        Returns a new SU2Field and leaves this one unchanged. With g=None a random transformation is drawn
        (from rng). Every gauge-invariant quantity (Plaq, Action) is the same on both fields."""
        if g is None:
            g = self.random_gauge_transformation(rng)
        if not isinstance(g, LatticeComplexMatrix):
            g = np.asarray(g, dtype=complex)
            if g.shape != (self.value.shape[0], self.N, self.N):
                raise ValueError("g must have shape {}, not {}".format((self.value.shape[0], self.N, self.N), g.shape))
            G = LatticeComplexMatrix(grid=self.grid, cartesiancomm=self.cartesiancomm, N=self.N)
            G.value = g
            g = G
        out = SU2Field(grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd, rng=self.rng)
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
        ordered pairs mu != nu (so R x T and T x R together): a complex number whose imaginary part is rounding."""
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
