"""SU(2): the algebra, the exponential map, the projection onto the group, and the quaternionic
representation of Creutz. Pure math on 2x2 matrices: no lattice and no MPI here. The gauge theory built on
these is in pylattice.theories.gauge.sun.

Why SU(2) first: every SU(N) update (Cabibbo-Marinari) is a sequence of SU(2) updates on the 2x2
subblocks embedded in SU(N): heatbath_link updates one block, and pylattice.core.groups.su picks and puts
back the blocks. SU2 (the last class of this file) is the same group behind the common interface of
pylattice.core.groups.base / lie, so a theory can be written for any group.

Conventions
    generators      T_a = sigma_a / 2, [T_a, T_b] = i eps_abc T_c, tr(T_a T_b) = delta_ab / 2
    algebra vector  w (3 real numbers): the group element is exp(i w.T)
    quaternion      q = (a0, a1, a2, a3), the element U = a0 + i a.sigma = [[a0 + i a3,  a2 + i a1],
                                                                          [-a2 + i a1, a0 - i a3]]
                    U is in SU(2) exactly when |q| = 1; det U = |q|^2, tr U / 2 = a0, U^dagger = (a0, -a),
                    exp(i w.T) = (cos(|w|/2), sin(|w|/2) w/|w|)

Two layers. The functions with tensors (get_generator, random_lie_algebra_vector, ProjToAlgebra,
ExpProjToGroup, LogToAlgebra, unitarize) take and return pylattice.core.tensors objects. The quaternionic
ones (to_quaternion, to_matrix, quat_mul, ..., heatbath_link) work on numpy arrays of any batch shape:
a quaternion is the last axis (4,), a matrix the last two axes (2, 2); this is what the lattice code uses,
a whole lattice of links at once. Everything is double precision (complex128).

Random numbers: every function that draws takes a numpy Generator (rng); with None a fresh unseeded one
is used. Give one (pylattice.core.rng) for a reproducible run.
"""
import numpy as np
from scipy.linalg import expm

from pylattice.core.groups.lie import LieGroup
from pylattice.core.tensors import ComplexMatrix, VectorReal, VectorComplex

_SIGMA = np.array([[[0, 1], [1, 0]],
                   [[0, -1j], [1j, 0]],
                   [[1, 0], [0, -1]]], dtype=complex)
_SIGMA.setflags(write=False)

ADJOINT_DIM = 3


def _rng(rng):
    return rng if rng is not None else np.random.default_rng()


# ---------------------------------------------------------------------------------------- algebra
def generators():
    """the three generators T_a = sigma_a / 2 as an array of shape (3, 2, 2) (a copy)"""
    return _SIGMA / 2


def get_generator(k):
    """the generator T_k, k = 0, 1, 2, as a ComplexMatrix (a new one each time)"""
    if k not in (0, 1, 2) or isinstance(k, bool):
        raise IndexError("SU(2) has the generators 0, 1, 2, not {!r}".format(k))
    return ComplexMatrix(value=_SIGMA[int(k)] / 2)


def structure_constants(i, j, k):
    """f_ijk of [T_i, T_j] = i f_ijk T_k: the Levi-Civita symbol, +1 for the cyclic (0,1,2), -1 for
    the anticyclic ones, 0 when two indices are equal"""
    for x in (i, j, k):
        if x not in (0, 1, 2):
            raise IndexError("SU(2) has the indices 0, 1, 2, not {!r}".format(x))
    return int((i - j) * (j - k) * (k - i) // 2)


def random_lie_algebra_vector(rng=None):
    """a vector of the algebra with each component uniform in [-0.5, 0.5): a VectorReal. This is not a
    uniform element of the group (see random_quaternion for that): it is a point of a cube."""
    return VectorReal(value=_rng(rng).random(ADJOINT_DIM) - 0.5)


def ProjToAlgebra(input):
    """The components c_a = -2i tr(T_a M) of a 2x2 matrix M along the generators, a VectorComplex.

    It is exact on the algebra: M = i sum_a w_a T_a gives c = w. On a group element exp(i w.T) it is
    only the first-order part, 2 sin(|w|/2) w/|w|, and it drops the part along the identity; for the
    exact inverse of the exponential use LogToAlgebra."""
    return VectorComplex(value=-2j * np.einsum('aij,ji->a', generators(), input.value))


def ExpProjToGroup(input):
    """exp(i w.T) for the vector w of the algebra, a ComplexMatrix. For a real w it is an element of SU(2)
    to machine precision (closed form cos(|w|/2) + i sin(|w|/2) w.sigma/|w|); a complex w gives a
    complex combination of the generators, whose exponential is in SL(2, C), not in SU(2)."""
    w = np.asarray(input.value)
    if w.shape != (ADJOINT_DIM,):
        raise ValueError("expected a vector of {} components, got shape {}".format(ADJOINT_DIM, w.shape))
    if np.iscomplexobj(w):
        return ComplexMatrix(value=expm(1j * np.einsum('a,aij->ij', w, generators())))
    return ComplexMatrix(value=exp_algebra(w))


def LogToAlgebra(input):
    """The vector w, a VectorReal, of an SU(2) element U = exp(i w.T), the inverse of ExpProjToGroup.
    Of the equivalent vectors the one with |w| in [0, 2 pi] is returned (the same U comes back from the
    exponential). For U = -1 the axis is not defined: w = (0, 0, 2 pi)."""
    return VectorReal(value=log_algebra(input.value))


def exp_algebra(w):
    """exp(i w.T) for real vectors w of shape (..., 3): the matrices (..., 2, 2), in SU(2) to machine
    precision (closed form cos(|w|/2) + i sin(|w|/2) w.sigma/|w|, finite at w = 0)."""
    w = np.asarray(w, dtype=float)
    theta = np.linalg.norm(w, axis=-1, keepdims=True)
    half_sinc = 0.5 * np.sinc(theta / (2 * np.pi))                         # sin(theta/2) / theta
    return to_matrix(np.concatenate([np.cos(theta / 2), half_sinc * w], axis=-1))


def log_algebra(U):
    """The vectors w (..., 3) of SU(2) matrices U (..., 2, 2) with exp(i w.T) = U, |w| in [0, 2 pi];
    (0, 0, 2 pi) for U = -1, whose axis is not defined."""
    q = to_quaternion(U)
    a0, a = q[..., 0], q[..., 1:]
    norm = np.linalg.norm(a, axis=-1)
    small = norm < 1e-12
    safe = np.where(small, 1.0, norm)
    w = (2 * np.arctan2(norm, a0) / safe)[..., None] * a
    w = np.where(small[..., None] & (a0 > 0)[..., None], 2 * a / np.where(a0 > 0, a0, 1.0)[..., None], w)
    minus_one = np.broadcast_to(np.array([0.0, 0.0, 2 * np.pi]), w.shape)
    return np.where(small[..., None] & (a0 <= 0)[..., None], minus_one, w)


def unitarize(input):
    """The element of SU(2) nearest to a 2x2 matrix (the one that maximises Re tr(V^dagger M), i.e. the
    smallest Frobenius distance), as a ComplexMatrix. Use it to remove the rounding errors that pile up
    in a product of group elements: an element of SU(2) is returned unchanged, to rounding. See project_su2.

    ValueError if the matrix has no SU(2) part at all (e.g. diag(1, -1) or the zero matrix)."""
    return ComplexMatrix(value=project_su2(input.value))


# -------------------------------------------------------------------- quaternions (Creutz), arrays
def to_quaternion(M):
    """The quaternion (a0, a1, a2, a3) of the matrices M, shape (..., 2, 2) -> (..., 4).

    For an SU(2) element, or a real multiple of one, or a sum of them, this is the whole matrix. A general
    complex 2x2 matrix has 8 real numbers, the quaternion keeps the 4 of them that lie along
    1, i sigma_a: the rest is orthogonal to SU(2) (in tr(V^dagger M)) and is dropped."""
    M = np.asarray(M)
    if M.shape[-2:] != (2, 2):
        raise ValueError("expected 2x2 matrices, got shape {}".format(M.shape))
    return np.stack([(M[..., 0, 0] + M[..., 1, 1]).real / 2,
                     (M[..., 0, 1] + M[..., 1, 0]).imag / 2,
                     (M[..., 0, 1] - M[..., 1, 0]).real / 2,
                     (M[..., 0, 0] - M[..., 1, 1]).imag / 2], axis=-1)


def to_matrix(q):
    """The matrices a0 + i a.sigma of the quaternions q, shape (..., 4) -> (..., 2, 2), complex128"""
    q = np.asarray(q, dtype=float)
    if q.shape[-1] != 4:
        raise ValueError("expected quaternions of 4 components, got shape {}".format(q.shape))
    a0, a1, a2, a3 = (q[..., i] for i in range(4))
    out = np.empty(q.shape[:-1] + (2, 2), dtype=complex)
    out[..., 0, 0] = a0 + 1j * a3
    out[..., 0, 1] = a2 + 1j * a1
    out[..., 1, 0] = -a2 + 1j * a1
    out[..., 1, 1] = a0 - 1j * a3
    return out


def quat_mul(p, q):
    """The product p q of quaternions, the same as the product of their matrices:
    (p0 q0 - p.q,  p0 q + q0 p - p x q). It does not commute (the cross product)."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    p0, pv, q0, qv = p[..., :1], p[..., 1:], q[..., :1], q[..., 1:]
    return np.concatenate([p0 * q0 - np.sum(pv * qv, axis=-1, keepdims=True),
                           p0 * qv + q0 * pv - np.cross(pv, qv)], axis=-1)


def quat_conj(q):
    """(a0, -a): the adjoint of the matrix, and the inverse for a unit quaternion"""
    q = np.array(q, dtype=float)
    q[..., 1:] *= -1
    return q


def quat_normalize(q):
    """q / |q|; ValueError if some |q| is 0"""
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("the quaternion 0 has no direction: it is not a multiple of an SU(2) element")
    return q / norm


def project_su2(M):
    """The element of SU(2) nearest to each 2x2 matrix of M, shape (..., 2, 2): the normalised quaternion
    part (see to_quaternion). The nearest one is a unique element, with the right sign: -U stays -U."""
    return to_matrix(quat_normalize(to_quaternion(M)))


def random_quaternion(rng=None, size=None):
    """Unit quaternions uniform on the 3-sphere, i.e. Haar-distributed SU(2) elements. size: a shape
    (None for a single one); the result has shape size + (4,)."""
    shape = () if size is None else ((size,) if np.isscalar(size) else tuple(size))
    return quat_normalize(_rng(rng).standard_normal(shape + (4,)))


def near_identity_quaternion(eps, rng=None, size=None):
    """Unit quaternions (sqrt(1 - eps^2), eps n) with n uniform on the sphere: SU(2) elements at a distance
    eps from the identity, in a uniformly random direction. The set is closed under inversion, so a
    Metropolis step U -> X U with such an X is symmetric. 0 <= eps <= 1."""
    if not 0 <= eps <= 1:
        raise ValueError("eps must be between 0 and 1, got {}".format(eps))
    shape = () if size is None else ((size,) if np.isscalar(size) else tuple(size))
    n = quat_normalize(np.concatenate([np.zeros(shape + (1,)), _rng(rng).standard_normal(shape + (3,))], axis=-1))
    n[..., 0] = np.sqrt(1.0 - eps * eps)
    n[..., 1:] *= eps
    return n


def su2_decompose(V):
    """V = k W with k >= 0 and W in SU(2), for V a sum of SU(2) matrices (the sum of the staples of a link):
    returns (k, W) with k = sqrt(det V), shape (...,) and (..., 2, 2). A sum of SU(2) matrices is a real
    multiple of one, that is why this is possible only in SU(2). k = 0 gives W = 1."""
    q = to_quaternion(V)
    k = np.linalg.norm(q, axis=-1)
    safe = np.where(k > 0, k, 1.0)[..., None]
    unit = np.where(k[..., None] > 0, q / safe, np.array([1.0, 0, 0, 0]))
    return k, to_matrix(unit)


KP_THRESHOLD = 6.0              # alpha above which Kennedy-Pendleton is faster than Creutz (measured)


def _alpha_shape(alpha, size):
    alpha = np.asarray(alpha, dtype=float)
    if np.any(alpha < 0) or not np.all(np.isfinite(alpha)):
        raise ValueError("alpha must be finite and >= 0")
    shape = np.broadcast(alpha, np.empty(() if size is None else size)).shape
    return np.broadcast_to(alpha, shape).reshape(-1), shape


def _rejection(alpha, rng, propose):
    """a0 for each alpha (1-d, all > 0): propose(alpha, rng) -> (a0, accepted), repeated for those rejected"""
    out = np.empty(alpha.shape)
    todo = np.arange(alpha.size)
    while todo.size:
        a0, accepted = propose(alpha[todo], rng)
        out[todo[accepted]] = a0[accepted]
        todo = todo[~accepted]
    return out


def _creutz(a, rng):
    # exponential part exactly (r in [exp(-2a), 1], via expm1/log1p for small a), sqrt(1 - a0^2) by acceptance
    a0 = np.clip(1.0 + np.log1p(rng.random(a.size) * np.expm1(-2.0 * a)) / a, -1.0, 1.0)
    return a0, rng.random(a.size) ** 2 < 1.0 - a0 * a0                     # u < sqrt(1 - a0^2)


def _kennedy_pendleton(a, rng):
    # a0 = 1 - delta, delta = -(ln x1 + cos^2(2 pi x2) ln x3) / a, accepted if x4^2 <= 1 - delta / 2
    n = a.size
    x1, x3 = 1.0 - rng.random(n), 1.0 - rng.random(n)                      # in (0, 1]: no log(0)
    x2, x4 = rng.random(n), rng.random(n)
    delta = -(np.log(x1) + np.cos(2 * np.pi * x2) ** 2 * np.log(x3)) / a
    return 1.0 - delta, x4 * x4 <= 1.0 - delta / 2.0


def _sample_a0(alpha, rng, size, method):
    alpha, shape = _alpha_shape(alpha, size)
    rng = _rng(rng)
    out = np.empty(alpha.shape)
    zero = alpha == 0
    out[zero] = random_quaternion(rng, size=int(zero.sum()))[:, 0]           # no weight: uniform on the sphere
    kp = alpha > KP_THRESHOLD if method == 'auto' else np.full(alpha.shape, method == 'kp')
    for use_kp, propose in ((False, _creutz), (True, _kennedy_pendleton)):
        pick = ~zero & (kp == use_kp)
        if pick.any():
            out[pick] = _rejection(alpha[pick], rng, propose)
    return out.reshape(shape)


def creutz_a0(alpha, rng=None, size=None):
    """a0 distributed as  sqrt(1 - a0^2) exp(alpha a0)  on [-1, 1], by the method of Creutz (1980): the
    exponential part is drawn exactly, a0 = 1 + ln(r)/alpha with r uniform in [exp(-2 alpha), 1], and
    the factor sqrt(1 - a0^2) is an acceptance test. alpha = 0 is the Haar distribution.
    Its acceptance falls like 1/sqrt(alpha): for large alpha use kennedy_pendleton_a0, or sample_a0
    which chooses.

    alpha: a number or an array (broadcast with size, a shape); alpha >= 0."""
    return _sample_a0(alpha, rng, size, 'creutz')


def kennedy_pendleton_a0(alpha, rng=None, size=None):
    """The same distribution as creutz_a0, by the method of Kennedy and Pendleton (1985): a0 = 1 - delta
    with delta from two logarithms and a cosine, accepted with probability sqrt(1 - delta/2). Its
    acceptance goes to 1 as alpha grows (above 95% for alpha > 10), but it is slow for small alpha (about
    28 trials per sample at alpha = 0.1): see sample_a0. alpha = 0 is the Haar distribution."""
    return _sample_a0(alpha, rng, size, 'kp')


def sample_a0(alpha, rng=None, size=None):
    """The same distribution again, with the faster of the two methods for each alpha: Creutz up to
    KP_THRESHOLD and Kennedy-Pendleton above it. This is what heatbath_link uses."""
    return _sample_a0(alpha, rng, size, 'auto')


def heatbath_link(V, beta, rng=None):
    """A new SU(2) link U drawn with probability  exp((beta/2) Re tr(U V))  where V is the sum of the
    staples of the link (the products of the other links of the plaquettes that contain it, so that a
    plaquette is tr(U staple)). Exact, independent of the old link, and fast at any beta (Creutz for small beta k, Kennedy-Pendleton for large). V: (..., 2, 2), one link per matrix.

    With V = k W, W in SU(2): U = X W^dagger, X with a0 from sample_a0(beta k) and the direction of its
    vector part uniform on the sphere. beta >= 0. Returns (..., 2, 2)."""
    if beta < 0:
        raise ValueError("beta must be >= 0, got {}".format(beta))
    rng = _rng(rng)
    k, W = su2_decompose(V)
    a0 = sample_a0(beta * k, rng)
    n = quat_normalize(rng.standard_normal(k.shape + (3,)))
    q = np.concatenate([a0[..., None], np.sqrt(np.maximum(1.0 - a0 * a0, 0.0))[..., None] * n], axis=-1)
    return to_matrix(q) @ np.conj(np.swapaxes(W, -1, -2))


# ------------------------------------------------------------------ the group behind the common interface
class SU2Group(LieGroup):
    """SU(2) as a LieGroup (pylattice.core.groups.lie): the elements are complex 2x2 matrices, an array of them
    has shape size + (2, 2), the algebra vectors have shape size + (3,). The functions above are the same
    operations spelled out; this class only gives them the names every group has, so that a theory can take
    "a group" as an argument. Use the instance SU2."""
    name = 'SU(2)'
    dim = 3
    shape = (2, 2)

    def identity(self, size=None):
        shape = () if size is None else ((size,) if np.isscalar(size) else tuple(size))
        return np.broadcast_to(np.eye(2, dtype=complex), shape + (2, 2)).copy()

    def random(self, rng=None, size=None):
        return to_matrix(random_quaternion(rng, size))

    def multiply(self, a, b):
        return np.asarray(a) @ np.asarray(b)

    def inverse(self, a):
        return np.conj(np.swapaxes(np.asarray(a), -1, -2))

    def project(self, x):
        return project_su2(x)

    def contains(self, x, tol=1e-9):
        x = np.asarray(x)
        ok = (np.abs(x @ self.inverse(x) - np.eye(2)).max(axis=(-2, -1)) <= tol) & \
             (np.abs(np.linalg.det(x) - 1) <= tol)
        return ok

    def generators(self):
        return generators()

    def exp(self, w):
        return exp_algebra(w)

    def log(self, g):
        return log_algebra(g)

    def near_identity(self, eps, rng=None, size=None):
        return to_matrix(near_identity_quaternion(eps, rng, size))


SU2 = SU2Group()
