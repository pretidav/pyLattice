"""The observables and the gauge transformation of a gauge field, for any gauge group: staples, plaquettes, Wilson and Polyakov loops,
the Wilson action. GaugeLinksMixin is written only with the operations of the lattice classes (pylattice.core.grid: shifts,
products, adjoints, traces, reductions), so it is the same for SU(N) (complex links) and SO(N) (real links) and the same however the
lattice is split over the ranks. A gauge field class puts it in front of its lattice class, as in
    class SUNField(GaugeLinksMixin, LatticeVectorComplexMatrix)
and only has to provide what is specific of the group: the links (cold and hot start, projection, proposals) and the random
gauge transformation. See pylattice.theories.gauge.field.

The lattice class must give: value of shape (sites, Nd, N, N), Nd, N, grid, cartesiancomm, peek_index(mu) (the lattice of the
links in direction mu), poke_index, copy, and for the single-direction matrix lattices shift_forward/shift_backward, adj, trace, +, *.
"""
import numpy as np


class GaugeLinksMixin:

    def _link_lattice(self, value):
        """a single-direction matrix lattice of the type of the links (see peek_index), holding value, shape (sites, N, N)"""
        out = self.peek_index(0)
        out.value = value
        return out

    def gauge_transform(self, g=None, rng=None):
        """The field after the gauge transformation g (a group element on every site, a lattice of matrices or an array of shape
        (sites, N, N)):   U'_mu(x) = g(x) U_mu(x) g(x + mu)^+
        Returns a new field of the same type and leaves this one unchanged. With g=None a random transformation is drawn
        (from rng, see random_gauge_transformation). Every gauge-invariant quantity (Plaq, Action, the loops) is the same on both
        fields."""
        if g is None:
            g = self.random_gauge_transformation(rng)
        if not hasattr(g, 'shift_forward'):
            g = np.asarray(g, dtype=self.value.dtype)
            if g.shape != (self.value.shape[0], self.N, self.N):
                raise ValueError("g must have shape {}, not {}".format((self.value.shape[0], self.N, self.N), g.shape))
            g = self._link_lattice(g)
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
        """P_{mu nu}(x) = U_mu(x) U_nu(x+mu) U_mu(x+nu)^+ U_nu(x)^+ on every site (a lattice of matrices)"""
        self._check_directions(mu, nu)
        if mu == nu:
            raise ValueError("a plaquette needs two different directions")
        Umu, Unu = self.peek_index(mu), self.peek_index(nu)
        return Umu * Unu.shift_forward(mu) * Umu.shift_forward(nu).adj() * Unu.adj()

    def Staple(self, mu):
        """The sum of the 2 (Nd - 1) staples of the links in direction mu (a lattice of matrices). For nu != mu:
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
        """The product of `length` links along mu from every site, U_mu(x) U_mu(x+mu) ... U_mu(x+(length-1)mu) (a lattice of
        matrices)"""
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
        a lattice of matrices. R and T cannot exceed the extent of the whole lattice along mu and nu."""
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
        of the links and the shifts (for real links it is exactly 0). With pairs one can separate, e.g., the spatial
        plaquettes from the temporal ones."""
        self._check_directions()
        total, count = self._trace_sum(self._ordered_pairs() if pairs is None else list(pairs))
        return total / (self.N * count)

    def Wilson(self, R, T):
        """The Wilson loop (1/N) tr of the R x T loops, averaged over all the sites of the whole lattice and all the
        ordered pairs mu != nu (a complex number). The reverse of a loop has the conjugate trace, so the imaginary part is
        zero up to rounding for R = T and for real traces (SU(2), SO(N)); for SU(N), N >= 3 and R != T it is not: the Wilson
        loop is its real part, the average of Re tr."""
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
