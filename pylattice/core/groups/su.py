"""SU(N): its algebra and group operations, and the Cabibbo-Marinari building blocks. Pure math on N x N matrices: no lattice and
no MPI here. The same things pylattice.core.groups.su2 has for SU(2), for any N, except the heat bath of a link (Creutz,
Kennedy-Pendleton), which exists only for SU(2): an SU(N) link is updated with the SU(2) heat bath on its 2x2 blocks
(Cabibbo-Marinari, below).

Conventions (the same as su2, which is the case N = 2)
    generators      the generalized Gell-Mann matrices T_a, a = 0 .. N^2 - 2, hermitian and traceless with
                    tr(T_a T_b) = delta_ab / 2, [T_a, T_b] = i f_abc T_c, {T_a, T_b} = delta_ab / N + d_abc T_c.
                    N = 2: sigma_a / 2 (Pauli), N = 3: lambda_a / 2 (Gell-Mann), in the usual order: for each column j of the
                    matrix, the symmetric and the antisymmetric generator of every pair (i, j), i < j, then one diagonal
    algebra vector  w (N^2 - 1 real numbers): the group element is exp(i w.T)

Matrices have shape (..., N, N): a batch of links at once; algebra vectors (..., N^2 - 1). N is the size of the matrices
(inferred from the arrays where it can be). Everything is double precision (complex128). Functions that draw take a numpy
Generator (pylattice.core.rng); with None a fresh unseeded one is used.

    generators, structure_constants, d_symbols   the algebra
    exp_algebra, log_algebra                      exp(i w.T) and its inverse
    random_su, near_identity_su, project_su       Haar-random elements, elements near the identity, the nearest element
    SU(N), SU3                                    the group behind the common interface of groups.base / lie
    get_generator, random_lie_algebra_vector, ProjToAlgebra, ExpProjToGroup, LogToAlgebra, unitarize
                                                  the same with pylattice.core.tensors objects, as in su2

Cabibbo-Marinari: the 2x2 blocks of an N x N matrix are picked (su2_block), updated with the SU(2) heat bath of
pylattice.core.groups.su2, and put back (embed_su2). Every SU(N) update is a sequence of these steps over the subgroups (i, j),
which together generate SU(N) (cabibbo_marinari_link does all of them for a link). N = 2 is SU(2) itself, with the single
subgroup (0, 1). This module needs no action: a step takes the sum of the staples V of the links, which the gauge theory
computes.
"""
import math
from functools import lru_cache

import numpy as np

from pylattice.core.groups.lie import LieGroup
from pylattice.core.groups.su2 import heatbath_link, su2_decompose
from pylattice.core.tensors import ComplexMatrix, VectorComplex, VectorReal


def _rng(rng):
    return rng if rng is not None else np.random.default_rng()


def _check_n(N):
    if int(N) != N or N < 2:
        raise ValueError("N must be an integer of at least 2, got {}".format(N))
    return int(N)


def _n_of_dim(dim):
    """the N of SU(N) whose algebra has dimension dim"""
    N = int(round(math.sqrt(dim + 1)))
    if N * N - 1 != dim or N < 2:
        raise ValueError("{} is not the dimension of an SU(N) algebra (N^2 - 1)".format(dim))
    return N


# ---------------------------------------------------------------------------------------- algebra
@lru_cache(maxsize=None)
def _generators(N):
    T = []
    for j in range(1, N):
        for i in range(j):
            S, A = np.zeros((N, N), dtype=complex), np.zeros((N, N), dtype=complex)
            S[i, j] = S[j, i] = 0.5
            A[i, j], A[j, i] = -0.5j, 0.5j
            T += [S, A]
        d = np.zeros(N)
        d[:j], d[j] = 1.0, -j
        T.append(np.diag(d / math.sqrt(2 * j * (j + 1))).astype(complex))
    T = np.array(T)
    T.setflags(write=False)
    return T


def generators(N):
    """the N^2 - 1 generators T_a as an array of shape (N^2 - 1, N, N) (a copy)"""
    return _generators(_check_n(N)).copy()


def get_generator(N, k):
    """the generator T_k, k = 0 .. N^2 - 2, as a ComplexMatrix (a new one each time)"""
    T = _generators(_check_n(N))
    if isinstance(k, bool) or int(k) != k or not 0 <= k < len(T):
        raise IndexError("SU({}) has the generators 0 .. {}, not {!r}".format(N, len(T) - 1, k))
    return ComplexMatrix(value=T[int(k)].copy())


@lru_cache(maxsize=None)
def _structure(N):
    T = _generators(N)
    comm = np.einsum('aij,bjk->abik', T, T) - np.einsum('bij,ajk->abik', T, T)
    anti = np.einsum('aij,bjk->abik', T, T) + np.einsum('bij,ajk->abik', T, T)
    f = (-2j * np.einsum('abik,cki->abc', comm, T))
    d = 2 * np.einsum('abik,cki->abc', anti, T)
    assert np.abs(f.imag).max() < 1e-12 and np.abs(d.imag).max() < 1e-12
    f, d = f.real.copy(), d.real.copy()
    f.setflags(write=False)
    d.setflags(write=False)
    return f, d


def structure_constants(N):
    """f_abc of [T_a, T_b] = i f_abc T_c, as an array of shape (N^2 - 1,) * 3: real, totally antisymmetric (a copy). For
    N = 2 it is the Levi-Civita symbol (pylattice.core.groups.su2.structure_constants)"""
    return _structure(_check_n(N))[0].copy()


def d_symbols(N):
    """d_abc of {T_a, T_b} = delta_ab / N + d_abc T_c, as an array of shape (N^2 - 1,) * 3: real, totally symmetric, and
    zero for N = 2 (a copy)"""
    return _structure(_check_n(N))[1].copy()


def random_lie_algebra_vector(N, rng=None):
    """a vector of the algebra with each component uniform in [-0.5, 0.5): a VectorReal. This is not a uniform element
    of the group (see random_su for that): it is a point of a cube."""
    return VectorReal(value=_rng(rng).random(_check_n(N) ** 2 - 1) - 0.5)


def ProjToAlgebra(input):
    """The components c_a = -2i tr(T_a M) of an N x N matrix M along the generators, a VectorComplex: exact on the algebra
    (M = i sum_a w_a T_a gives c = w), only the first-order part on a group element (for the inverse of the exponential use
    LogToAlgebra)."""
    T = _generators(input.value.shape[-1])
    return VectorComplex(value=-2j * np.einsum('aij,ji->a', T, input.value))


def ExpProjToGroup(input):
    """exp(i w.T) for the vector w of the algebra (N^2 - 1 real components, N inferred), a ComplexMatrix, in SU(N) to
    machine precision"""
    w = np.asarray(input.value)
    if np.iscomplexobj(w):
        raise ValueError("the algebra vector must be real, a complex one has an exponential in SL(N, C), not in SU(N)")
    return ComplexMatrix(value=exp_algebra(w))


def LogToAlgebra(input):
    """The vector w, a VectorReal, of an SU(N) element U = exp(i w.T), the inverse of ExpProjToGroup (see log_algebra)"""
    return VectorReal(value=log_algebra(input.value))


def unitarize(input):
    """The element of SU(N) nearest to an N x N matrix, as a ComplexMatrix (see project_su): removes the rounding
    errors that pile up in a product of group elements; an element of SU(N) is returned unchanged, to rounding."""
    return ComplexMatrix(value=project_su(input.value))


def exp_algebra(w):
    """exp(i w.T) for real vectors w of shape (..., N^2 - 1): the matrices (..., N, N), in SU(N) to machine precision
    (the exponential of the hermitian matrix w.T through its eigendecomposition)."""
    w = np.asarray(w, dtype=float)
    N = _n_of_dim(w.shape[-1])
    H = np.einsum('...a,aij->...ij', w, _generators(N))
    lam, V = np.linalg.eigh(H)
    return (V * np.exp(1j * lam)[..., None, :]) @ np.conj(np.swapaxes(V, -1, -2))


def log_algebra(U):
    """The vectors w (..., N^2 - 1) of SU(N) matrices U (..., N, N) with exp(i w.T) = U (the inverse of exp_algebra). The
    eigenphases of U are taken in (-pi, pi] and, since they must add up to zero (w.T is traceless), the surplus of
    2 pi m is taken off the m largest (or added to the smallest): for an element near the identity this is the small w.
    The principal logarithm is unique unless the phases are at the cut; a matrix that is not in SU(N) is not checked.
    For N = 2 it is the same as pylattice.core.groups.su2.log_algebra, except at U = -1."""
    U = np.asarray(U, dtype=complex)
    N = U.shape[-1]
    T = _generators(N)
    Ud = np.conj(np.swapaxes(U, -1, -2))
    A, B = (U + Ud) / 2, (U - Ud) / 2j                  # hermitian and commuting: U = A + iB
    for c in (math.sqrt(2.0), 1.2345678, 0.7071067 + 0.31):       # a generic combination has the eigenvectors of U
        _, V = np.linalg.eigh(A + c * B)
        diag = np.einsum('...ji,...jk,...ki->...i', np.conj(V), U, V)
        off = np.einsum('...ji,...jk,...kl->...il', np.conj(V), U, V) * (1 - np.eye(N))
        if np.abs(off).max(initial=0.0) < 1e-9:
            break
    else:
        raise ValueError("could not diagonalize the matrices together: are they unitary?")
    theta = np.angle(diag)
    m = np.rint(theta.sum(axis=-1) / (2 * np.pi)).astype(int)[..., None]
    ranks = np.argsort(np.argsort(theta, axis=-1), axis=-1)
    theta = theta - 2 * np.pi * ((m > 0) & (ranks >= N - m)) + 2 * np.pi * ((m < 0) & (ranks < -m))
    H = (V * theta[..., None, :]) @ np.conj(np.swapaxes(V, -1, -2))
    return 2 * np.einsum('aij,...ji->...a', T, H).real


# ------------------------------------------------------------------------- group elements, arrays
def random_su(N, rng=None, size=None):
    """Haar-distributed (uniform) SU(N) matrices, shape size + (N, N): the Q of the QR decomposition of a complex Gaussian
    matrix with the phases of R made positive (uniform on U(N)), divided by an N-th root of its determinant."""
    N = _check_n(N)
    rng = _rng(rng)
    shape = () if size is None else ((size,) if np.isscalar(size) else tuple(size))
    Z = (rng.standard_normal(shape + (N, N)) + 1j * rng.standard_normal(shape + (N, N))) / math.sqrt(2)
    Q, R = np.linalg.qr(Z)
    d = np.diagonal(R, axis1=-2, axis2=-1)
    Q = Q * (d / np.abs(d))[..., None, :]
    return Q / (np.linalg.det(Q) ** (1.0 / N))[..., None, None]


def near_identity_su(N, eps, rng=None, size=None):
    """SU(N) elements exp(i w.T) at the distance eps from the identity, 0 <= eps <= 1, in a uniformly random direction n of
    the algebra: w = 2 arcsin(eps) n, so that for N = 2 it is the quaternion (sqrt(1 - eps^2), eps n) of
    pylattice.core.groups.su2.near_identity_quaternion. The set is closed under inversion (n -> -n), so a Metropolis step
    U -> X U with such an X is symmetric."""
    N = _check_n(N)
    if not 0 <= eps <= 1:
        raise ValueError("eps must be between 0 and 1, got {}".format(eps))
    shape = () if size is None else ((size,) if np.isscalar(size) else tuple(size))
    n = _rng(rng).standard_normal(shape + (N * N - 1,))
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return exp_algebra(2 * math.asin(eps) * n)


def project_su(M):
    """The SU(N) matrix nearest to the N x N matrices M (..., N, N): the unitary matrix of the polar decomposition
    (smallest Frobenius distance among the unitary ones), divided by an N-th root of its determinant to make it special.
    That is the exact nearest element for a matrix close to SU(N), the case of rounding errors; an element of SU(N) is
    returned unchanged, to rounding. ValueError for a matrix with no nearest unitary one (a singular matrix)."""
    M = np.asarray(M, dtype=complex)
    N = M.shape[-1]
    W, s, Vh = np.linalg.svd(M)
    if np.any(s.min(axis=-1) <= 1e-12 * np.maximum(s.max(axis=-1), 1e-300)):
        raise ValueError("a singular matrix has no unique nearest element of SU({})".format(N))
    U = W @ Vh
    return U / (np.linalg.det(U) ** (1.0 / N))[..., None, None]


# ------------------------------------------------------------------ the group behind the common interface
class SUGroup(LieGroup):
    """SU(N) as a LieGroup (pylattice.core.groups.lie): the elements are complex N x N matrices, an array of them has shape
    size + (N, N), the algebra vectors size + (N^2 - 1,). The functions above are the same operations spelled out; this class
    gives them the names every group has. Make it with SU(N); for N = 2 it is the group of pylattice.core.groups.su2.SU2
    (the same generators and conventions)."""

    def __init__(self, N):
        self.N = _check_n(N)
        self.name = 'SU({})'.format(self.N)
        self.dim = self.N ** 2 - 1
        self.shape = (self.N, self.N)

    def identity(self, size=None):
        shape = () if size is None else ((size,) if np.isscalar(size) else tuple(size))
        return np.broadcast_to(np.eye(self.N, dtype=complex), shape + self.shape).copy()

    def random(self, rng=None, size=None):
        return random_su(self.N, rng, size)

    def multiply(self, a, b):
        return np.asarray(a) @ np.asarray(b)

    def inverse(self, a):
        return np.conj(np.swapaxes(np.asarray(a), -1, -2))

    def project(self, x):
        return project_su(x)

    def contains(self, x, tol=1e-9):
        x = np.asarray(x)
        return (np.abs(x @ self.inverse(x) - np.eye(self.N)).max(axis=(-2, -1)) <= tol) & (np.abs(np.linalg.det(x) - 1) <= tol)

    def generators(self):
        return generators(self.N)

    def structure_constants(self):
        return structure_constants(self.N)

    def exp(self, w):
        return exp_algebra(w)

    def log(self, g):
        return log_algebra(g)

    def near_identity(self, eps, rng=None, size=None):
        return near_identity_su(self.N, eps, rng, size)

    def subgroups(self):
        """the SU(2) subgroups (i, j), i < j, that a Cabibbo-Marinari sweep goes through"""
        return [(i, j) for i in range(self.N) for j in range(i + 1, self.N)]


@lru_cache(maxsize=None)
def SU(N):
    """the group SU(N) (one object for each N)"""
    return SUGroup(N)


SU3 = SU(3)


# ------------------------------------------------------------------ Cabibbo-Marinari
def embed_su2(U, i, j, N):
    """The N x N matrix that is U (2x2, shape (..., 2, 2)) on the rows and columns i and j and the identity
    elsewhere: an element of the SU(2) subgroup (i, j) of SU(N). Returns shape (..., N, N)."""
    U = np.asarray(U)
    _check_block(i, j, N)
    out = np.zeros(U.shape[:-2] + (N, N), dtype=complex)
    out[..., np.arange(N), np.arange(N)] = 1.0
    out[..., i, i], out[..., i, j] = U[..., 0, 0], U[..., 0, 1]
    out[..., j, i], out[..., j, j] = U[..., 1, 0], U[..., 1, 1]
    return out


def su2_block(M, i, j):
    """The 2x2 block of rows and columns i, j of the N x N matrices M, written as k W with k >= 0 and W in
    SU(2) (the SU(2) part of the block, see to_quaternion; returns (k, W) as su2_decompose does). It is
    what a Cabibbo-Marinari step needs: the update of the subgroup (i, j) of a link U with staple sum V
    uses the block (i, j) of U V."""
    M = np.asarray(M)
    _check_block(i, j, M.shape[-1])
    return su2_decompose(M[..., [i, j], :][..., [i, j]])


def _check_block(i, j, N):
    if not (0 <= i < N and 0 <= j < N and i != j):
        raise ValueError("a block needs two different indices in 0..{}, got {}, {}".format(N - 1, i, j))


def cabibbo_marinari_step(U, V, c, i, j, rng=None):
    """One heat-bath step of the SU(2) subgroup (i, j) of SU(N) for the link weight exp(c Re tr(U V)), with V
    the sum of the staples of the link (so a plaquette is tr(U staple)): U -> X U with X in the subgroup,
    drawn from its exact conditional distribution. Only the block (i, j) of U V matters, since tr(X M) sees
    only that block. U, V: (..., N, N); c >= 0. Returns the new U, still in SU(N).

    A sweep over the links is a loop over the subgroups, e.g. (0, 1), (1, 2), (0, 2) for SU(3)."""
    U = np.asarray(U)
    k, W = su2_block(U @ np.asarray(V), i, j)
    X = heatbath_link(k[..., None, None] * W, 2 * c, rng)       # weight exp((2c/2) Re tr(X kW)) = exp(c Re tr(X M))
    return embed_su2(X, i, j, U.shape[-1]) @ U


def cabibbo_marinari_link(U, V, c, rng=None, subgroups=None):
    """The heat-bath steps of all the SU(2) subgroups, one after the other, on the links U with the same staple sums V: what
    one visit of a link is in a Cabibbo-Marinari sweep. subgroups: the pairs (i, j) to go through (default: all the pairs
    i < j, see SUGroup.subgroups). The staple does not depend on the link, so it is the same for all the steps. Returns the
    new U, still in SU(N)."""
    N = np.asarray(U).shape[-1]
    for i, j in (SU(N).subgroups() if subgroups is None else subgroups):
        U = cabibbo_marinari_step(U, V, c, i, j, rng)
    return U


def cabibbo_marinari_overrelax(U, V, i, j):
    """The overrelaxation step of the SU(2) subgroup (i, j) of SU(N) for the link weight exp(c Re tr(U V)), V the sum of the
    staples of the link: U -> X U with X = (W^+)^2 embedded in the subgroup, where k W is the SU(2) part of the block (i, j)
    of U V. Then tr(U V) keeps its real part, so the weight and the action do not change (a microcanonical move), while the
    link moves as far as it can; the step is its own inverse (W goes to W^+). It is always accepted, and not ergodic alone:
    mix it with heat bath steps. U, V: (..., N, N). Returns the new U, still in SU(N). For N = 2 it is U -> W^+ U^+ W^+
    with S = k W the staple sum, the overrelaxation of the SU(2) theory (N = 2 of pylattice.theories.gauge.sun)."""
    U = np.asarray(U)
    _, W = su2_block(U @ np.asarray(V), i, j)
    Wd = np.conj(np.swapaxes(W, -1, -2))
    return embed_su2(Wd @ Wd, i, j, U.shape[-1]) @ U


def cabibbo_marinari_overrelax_link(U, V, subgroups=None):
    """The overrelaxation steps of all the SU(2) subgroups, one after the other, on the links U with the same staple sums
    V (see cabibbo_marinari_overrelax). subgroups: the pairs (i, j) to go through (default: all the pairs i < j). The weight
    and the action do not change. Returns the new U."""
    N = np.asarray(U).shape[-1]
    for i, j in (SU(N).subgroups() if subgroups is None else subgroups):
        U = cabibbo_marinari_overrelax(U, V, i, j)
    return U
