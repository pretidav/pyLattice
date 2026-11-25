"""SO(N): its algebra and group operations, and the Cabibbo-Marinari building blocks. Pure math on real N x N matrices: no
lattice and no MPI here. The counterpart of pylattice.core.groups.su for the real orthogonal matrices with determinant +1
(the connected component of O(N)).

Conventions
    generators      T_a = i A_ij / sqrt(2), one for each pair i < j (in lexicographic order), A_ij = E_ij - E_ji the real
                    antisymmetric matrix of the rotation in the plane (i, j): hermitian, purely imaginary, with
                    tr(T_a T_b) = delta_ab, [T_a, T_b] = i f_abc T_c. There are N (N - 1) / 2 of them.
    algebra vector  w (N (N - 1) / 2 real numbers): the group element is exp(i w.T) = exp(-w.A / sqrt(2)), a real matrix

Matrices have shape (..., N, N): a batch of links at once; algebra vectors (..., N (N - 1) / 2). Everything is double precision
and real (float64). Functions that draw take a numpy Generator (pylattice.core.rng); with None a fresh unseeded one is used.

    generators, structure_constants       the algebra
    exp_algebra, log_algebra              exp(i w.T) and its inverse
    random_so, near_identity_so, project_so   Haar-random elements, elements near the identity, the nearest element
    SO(N), SO3                            the group behind the common interface of groups.base / lie

Cabibbo-Marinari. SO(N) has no exact heat bath, but the links can be updated one rotation plane at a time: the elements
R_ij(theta), the rotations by theta in the plane (i, j), embedded in the identity, are N (N - 1) / 2 SO(2) subgroups that
generate SO(N). For a link weight exp(c tr(U V)) (V the sum of the staples, so that tr(U V) is the sum of the plaquettes
that contain the link) the weight of R_ij(theta) U depends on theta only through
        tr(R_ij(theta) M) = const + a cos(theta - theta0),     M = U V,
        a = hypot(M_ii + M_jj, M_ij - M_ji),   theta0 = atan2(M_ij - M_ji, M_ii + M_jj)
(so2_block), a von Mises distribution that is sampled exactly (von_mises): this is the heat bath of one subgroup
(cabibbo_marinari_step), and a sweep over the planes (cabibbo_marinari_link) is the pseudo-heat-bath. Overrelaxation reflects
theta around theta0 (cabibbo_marinari_overrelax): the weight and the action do not change. SO(2) is abelian and has a single
plane, so for N = 2 the step is the exact heat bath of the whole group; no special code is needed. This module needs no
action: a step takes the staple sum V, which the gauge theory computes.
"""
import math
from functools import lru_cache

import numpy as np

from pylattice.core.groups.lie import LieGroup


def _rng(rng):
    return rng if rng is not None else np.random.default_rng()


def _check_n(N):
    if int(N) != N or N < 2:
        raise ValueError("N must be an integer of at least 2, got {}".format(N))
    return int(N)


def _shape(size):
    return () if size is None else ((size,) if np.isscalar(size) else tuple(size))


def _n_of_dim(dim):
    """the N of SO(N) whose algebra has dimension dim"""
    N = int(round((1 + math.sqrt(1 + 8 * dim)) / 2))
    if N * (N - 1) // 2 != dim or N < 2:
        raise ValueError("{} is not the dimension of an SO(N) algebra (N (N - 1) / 2)".format(dim))
    return N


def planes(N):
    """the rotation planes (i, j), i < j, in the order of the generators"""
    return [(i, j) for i in range(N) for j in range(i + 1, N)]


# ---------------------------------------------------------------------------------------- algebra
@lru_cache(maxsize=None)
def _generators(N):
    T = []
    for i, j in planes(N):
        A = np.zeros((N, N), dtype=complex)
        A[i, j], A[j, i] = 1j / math.sqrt(2), -1j / math.sqrt(2)
        T.append(A)
    T = np.array(T)
    T.setflags(write=False)
    return T


def generators(N):
    """the N (N - 1) / 2 generators T_a as an array of shape (N (N - 1) / 2, N, N) (a copy; hermitian, imaginary)"""
    return _generators(_check_n(N)).copy()


@lru_cache(maxsize=None)
def _structure(N):
    T = _generators(N)
    comm = np.einsum('aij,bjk->abik', T, T) - np.einsum('bij,ajk->abik', T, T)
    f = -1j * np.einsum('abik,cki->abc', comm, T)
    assert np.abs(f.imag).max(initial=0.0) < 1e-12
    f = f.real.copy()
    f.setflags(write=False)
    return f


def structure_constants(N):
    """f_abc of [T_a, T_b] = i f_abc T_c, as an array of shape (N (N - 1) / 2,) * 3: real, totally antisymmetric (a copy)"""
    return _structure(_check_n(N)).copy()


def exp_algebra(w):
    """exp(i w.T) for real vectors w of shape (..., N (N - 1) / 2): the real orthogonal matrices (..., N, N) with determinant 1
    (the exponential of the hermitian matrix w.T through its eigendecomposition; the imaginary part is rounding)."""
    w = np.asarray(w, dtype=float)
    N = _n_of_dim(w.shape[-1])
    H = np.einsum('...a,aij->...ij', w, _generators(N))
    lam, V = np.linalg.eigh(H)
    U = (V * np.exp(1j * lam)[..., None, :]) @ np.conj(np.swapaxes(V, -1, -2))
    return np.ascontiguousarray(U.real)


def log_algebra(U):
    """The vectors w (..., N (N - 1) / 2) of SO(N) matrices U (..., N, N) with exp(i w.T) = U (the inverse of exp_algebra): the
    principal logarithm, the angles of the rotations in (-pi, pi]. Unique unless an eigenvalue is -1 (a rotation by pi). A
    matrix that is not in SO(N) is not checked."""
    from pylattice.core.groups import su             # the eigenphases of a unitary matrix, shared with SU(N)
    U = np.asarray(U, dtype=float)
    N = U.shape[-1]
    w_su = su.log_algebra(U.astype(complex))
    H = np.einsum('...a,aij->...ij', w_su, su._generators(N))
    return np.einsum('aij,...ji->...a', _generators(N), H).real


# ------------------------------------------------------------------------- group elements, arrays
def random_so(N, rng=None, size=None):
    """Haar-distributed (uniform) SO(N) matrices, shape size + (N, N): the Q of the QR decomposition of a Gaussian matrix with
    the signs of R made positive (uniform on O(N)), the first column reversed where the determinant is -1."""
    N = _check_n(N)
    shape = _shape(size)
    Q, R = np.linalg.qr(_rng(rng).standard_normal(shape + (N, N)))
    Q = Q * np.sign(np.diagonal(R, axis1=-2, axis2=-1))[..., None, :]
    Q[..., :, 0] *= np.sign(np.linalg.det(Q))[..., None]
    return Q


def near_identity_so(N, eps, rng=None, size=None):
    """SO(N) elements exp(i w.T) in a uniformly random direction n of the algebra, w = 2 arcsin(eps) n (the same scale as
    pylattice.core.groups.su.near_identity_su; 0 <= eps <= 1). The set is closed under inversion (n -> -n), so a Metropolis
    step U -> X U with such an X is symmetric."""
    N = _check_n(N)
    if not 0 <= eps <= 1:
        raise ValueError("eps must be between 0 and 1, got {}".format(eps))
    n = _rng(rng).standard_normal(_shape(size) + (N * (N - 1) // 2,))
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return exp_algebra(2 * math.asin(eps) * n)


def project_so(M):
    """The SO(N) matrix nearest to the N x N matrices M (..., N, N): the orthogonal matrix of the polar decomposition, with
    the direction of the smallest singular value reversed where the determinant is -1. An element of SO(N) is returned
    unchanged, to rounding. ValueError for a singular matrix (no unique nearest element)."""
    M = np.asarray(M, dtype=float)
    W, s, Vh = np.linalg.svd(M)
    if np.any(s.min(axis=-1) <= 1e-12 * np.maximum(s.max(axis=-1), 1e-300)):
        raise ValueError("a singular matrix has no unique nearest element of SO({})".format(M.shape[-1]))
    W[..., :, -1] *= np.sign(np.linalg.det(W @ Vh))[..., None]
    return W @ Vh


# ------------------------------------------------------------------ the group behind the common interface
class SOGroup(LieGroup):
    """SO(N) as a LieGroup (pylattice.core.groups.lie): the elements are real N x N matrices, an array of them has shape
    size + (N, N), the algebra vectors size + (N (N - 1) / 2,). Make it with SO(N)."""

    def __init__(self, N):
        self.N = _check_n(N)
        self.name = 'SO({})'.format(self.N)
        self.dim = self.N * (self.N - 1) // 2
        self.shape = (self.N, self.N)

    def identity(self, size=None):
        return np.broadcast_to(np.eye(self.N), _shape(size) + self.shape).copy()

    def random(self, rng=None, size=None):
        return random_so(self.N, rng, size)

    def multiply(self, a, b):
        return np.asarray(a) @ np.asarray(b)

    def inverse(self, a):
        return np.swapaxes(np.asarray(a), -1, -2).copy()

    def project(self, x):
        return project_so(x)

    def contains(self, x, tol=1e-9):
        x = np.asarray(x)
        return (np.abs(x @ np.swapaxes(x, -1, -2) - np.eye(self.N)).max(axis=(-2, -1)) <= tol) & (np.abs(np.linalg.det(x) - 1) <= tol)

    def generators(self):
        return generators(self.N)

    def structure_constants(self):
        return structure_constants(self.N)

    def exp(self, w):
        return exp_algebra(w)

    def log(self, g):
        return log_algebra(g)

    def near_identity(self, eps, rng=None, size=None):
        return near_identity_so(self.N, eps, rng, size)

    def subgroups(self):
        """the SO(2) subgroups (i, j), i < j, that a Cabibbo-Marinari sweep goes through"""
        return planes(self.N)


@lru_cache(maxsize=None)
def SO(N):
    """the group SO(N) (one object for each N)"""
    return SOGroup(N)


SO3 = SO(3)


# ------------------------------------------------------------------ Cabibbo-Marinari
def _check_plane(i, j, N):
    if not (0 <= i < N and 0 <= j < N and i != j):
        raise ValueError("a plane needs two different indices in 0..{}, got {}, {}".format(N - 1, i, j))


def embed_so2(theta, i, j, N):
    """The N x N matrix that is the rotation by theta (an array, shape (...)) in the plane (i, j) and the identity elsewhere:
    [[cos, -sin], [sin, cos]] on the rows and columns i and j. Returns shape (..., N, N)."""
    theta = np.asarray(theta, dtype=float)
    _check_plane(i, j, N)
    out = np.zeros(theta.shape + (N, N))
    out[..., np.arange(N), np.arange(N)] = 1.0
    c, s = np.cos(theta), np.sin(theta)
    out[..., i, i], out[..., i, j], out[..., j, i], out[..., j, j] = c, -s, s, c
    return out


def so2_block(M, i, j):
    """(a, theta0) of the plane (i, j) of the N x N matrices M (..., N, N): tr(R_ij(theta) M) = const + a cos(theta - theta0),
    a >= 0, theta0 in (-pi, pi]. Only the rotation part of the 2x2 block matters (its reflection part drops out of the trace);
    a Cabibbo-Marinari step uses the block of M = U V."""
    M = np.asarray(M, dtype=float)
    _check_plane(i, j, M.shape[-1])
    s, c = M[..., i, j] - M[..., j, i], M[..., i, i] + M[..., j, j]
    return np.hypot(c, s), np.arctan2(s, c)


def von_mises(kappa, rng=None):
    """Exact samples of the angles theta in (-pi, pi] with density proportional to exp(kappa cos theta), one for each value of
    kappa >= 0 (an array, any shape). For kappa < 1 the proposal is uniform, accepted with probability exp(kappa (cos theta - 1));
    for kappa >= 1 the algorithm of Best and Fisher (1979), a wrapped-Cauchy proposal. Both are rejection methods: the
    samples are exact, and the loop runs until every value is accepted."""
    rng = _rng(rng)
    kappa = np.asarray(kappa, dtype=float)
    if np.any(kappa < 0) or not np.all(np.isfinite(kappa)):
        raise ValueError("kappa must be finite and not negative")
    out = np.empty(kappa.shape)
    todo = np.ones(kappa.shape, dtype=bool)
    tau = 1 + np.sqrt(1 + 4 * kappa ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        rho = (tau - np.sqrt(2 * tau)) / (2 * kappa)
        r = (1 + rho ** 2) / (2 * rho)
    small = kappa < 1
    while todo.any():
        n = int(todo.sum())
        k, sm = kappa[todo], small[todo]
        u1, u2, u3 = rng.random(n), rng.random(n), rng.random(n)
        # kappa < 1: uniform angle, accept with exp(kappa (cos - 1))
        theta_s = np.pi * (2 * u1 - 1)
        ok_s = u2 < np.exp(k * (np.cos(theta_s) - 1))
        # kappa >= 1: Best-Fisher
        rr = r[todo]
        with np.errstate(invalid='ignore', divide='ignore'):
            z = np.cos(np.pi * u1)
            f = (1 + rr * z) / (rr + z)
            cc = k * (rr - f)
            ok_b = (cc * (2 - cc) - u2 > 0) | (np.log(cc / u2) + 1 - cc >= 0)
        theta_b = np.where(u3 > 0.5, 1.0, -1.0) * np.arccos(np.clip(f, -1.0, 1.0))
        ok = np.where(sm, ok_s, ok_b)
        idx = np.flatnonzero(todo)
        out.flat[idx[ok]] = np.where(sm, theta_s, theta_b)[ok]
        todo.flat[idx[ok]] = False
    return out


def cabibbo_marinari_step(U, V, c, i, j, rng=None):
    """One heat-bath step of the SO(2) subgroup (i, j) of SO(N) for the link weight exp(c tr(U V)), with V the sum of the
    staples of the link (so a plaquette is tr(U staple)): U -> R_ij(theta) U with theta drawn from its exact conditional
    distribution, exp(c a cos(theta - theta0)) with (a, theta0) = so2_block(U V, i, j). U, V: (..., N, N); c >= 0. Returns the
    new U, still in SO(N)."""
    U = np.asarray(U, dtype=float)
    a, theta0 = so2_block(U @ np.asarray(V, dtype=float), i, j)
    theta = theta0 + von_mises(c * a, rng)
    return embed_so2(theta, i, j, U.shape[-1]) @ U


def cabibbo_marinari_link(U, V, c, rng=None, subgroups=None):
    """The heat-bath steps of all the SO(2) subgroups, one after the other, on the links U with the same staple sums V: what one
    visit of a link is in a Cabibbo-Marinari sweep. subgroups: the planes (i, j) to go through (default: all, see
    SOGroup.subgroups). Returns the new U, still in SO(N)."""
    N = np.asarray(U).shape[-1]
    for i, j in (SO(N).subgroups() if subgroups is None else subgroups):
        U = cabibbo_marinari_step(U, V, c, i, j, rng)
    return U


def cabibbo_marinari_overrelax(U, V, i, j):
    """The overrelaxation step of the SO(2) subgroup (i, j) for the link weight exp(c tr(U V)): U -> R_ij(2 theta0) U, with
    theta0 of so2_block(U V, i, j), i.e. theta -> 2 theta0 - theta. tr(U V) keeps its value, so the weight and the action do not
    change (a microcanonical move); the step is its own inverse. It is always accepted, and not ergodic alone: mix it with heat
    bath steps. U, V: (..., N, N). Returns the new U, still in SO(N)."""
    U = np.asarray(U, dtype=float)
    _, theta0 = so2_block(U @ np.asarray(V, dtype=float), i, j)
    return embed_so2(2 * theta0, i, j, U.shape[-1]) @ U


def cabibbo_marinari_overrelax_link(U, V, subgroups=None):
    """The overrelaxation steps of all the SO(2) subgroups, one after the other, on the links U with the same staple sums V (see
    cabibbo_marinari_overrelax). Returns the new U."""
    N = np.asarray(U).shape[-1]
    for i, j in (SO(N).subgroups() if subgroups is None else subgroups):
        U = cabibbo_marinari_overrelax(U, V, i, j)
    return U
