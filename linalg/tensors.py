"""
Small tensor algebra used by the lattice classes.

Classes (``value`` is always a NumPy array; the last axes hold the structure,
so any leading axes are treated as a batch and every operation broadcasts over
them):

    =====================  ==========  ================================
    class                  value       meaning
    =====================  ==========  ================================
    Real / Complex         (1,)        scalar
    VectorReal/Complex     (Nd,)       vector
    RealMatrix/Complex..   (N, N)      matrix
    VectorRealMatrix/..    (Nd, N, N)  vector of matrices
    =====================  ==========  ================================

``Complex`` is a subclass of ``Real`` and ``ComplexMatrix`` of ``RealMatrix``
(as before), ``Identity`` is a ``RealMatrix``.

Binary operators (S = scalar, V = vector, M = matrix, T = vector of matrices;
a "scalar" is a Real/Complex or a Python/NumPy number). The result is complex
if and only if one of the operands is complex; anything else raises TypeError.

    +  -   S?S, V?V, M?M, T?T, M?T, T?M, and S?X for any X (the scalar is added
           to every entry); M?T broadcasts the matrix over the vector index.
    *      S*X, X*S (scaling), V*V (entrywise), M*M (matrix product),
           M*V (M v), V*M (v^T M), T*T / T*M / M*T (product for every index)
    /      X/S, S/S
    **     S**S, and M**int, T**int (matrix power, negative = inverse)
    -x     any class

Methods: conj, re, im, transpose, adj, copy on every class; trace, det, inv,
norm on matrices; dot, vdot, norm on vectors. ``dot`` is bilinear (no
conjugation), ``vdot`` conjugates the left operand.

Notes: arrays given as ``value=`` are stored as they are (no copy) unless a
dtype change is needed (integers become float64, real data given to a complex
class becomes complex); complex data given to a real class raises TypeError.
Results never share memory with their operands, except ``T[mu]`` which returns
a view of that component.
"""
import numpy as np
from operator import index as _index

_RANK = {'S': 0, 'V': 1, 'M': 2, 'T': 3}
_CLS_R = _CLS_C = None                      # kind -> real / complex class (set at the end)
_SCALAR_TYPES = {int: False, float: False, bool: False, complex: True}


def Kroneker(i, j):
    return 1 if i == j else 0


Kronecker = Kroneker


# ---------------------------------------------------------------- helpers ---
def _make(kind, cplx, value):
    """Instance of the class for (kind, cplx) around ``value``: no validation, no copy."""
    obj = object.__new__((_CLS_C if cplx else _CLS_R)[kind])
    obj.value = value
    return obj


def _coerce(value, cplx):
    """Array with a dtype suited to a real (cplx=False) or complex class."""
    if type(value) is np.ndarray and value.dtype.kind == ('c' if cplx else 'f'):
        return value                                    # fast path: nothing to do
    if isinstance(value, _Tensor):
        value = value.value
    a = np.asarray(value)
    k = a.dtype.kind
    if cplx:
        if k != 'c':
            a = a.astype(np.result_type(a.dtype, np.complex64))
    else:
        if k == 'c':
            raise TypeError("complex data cannot be stored in a real tensor")
        if k != 'f':
            a = a.astype(np.float64)
    return a


def _scalar_cplx(x):
    """None if x is not a plain number, else whether it is complex."""
    c = _SCALAR_TYPES.get(type(x))
    if c is not None:
        return c
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return False
    if isinstance(x, np.complexfloating):
        return True
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return x.dtype.kind == 'c'
    return None


def _operand(x):
    """(kind, value, is_complex) for a tensor or a number, else None."""
    try:
        return x._kind, x.value, x._cplx
    except AttributeError:
        c = _scalar_cplx(x)
        return None if c is None else ('S', x, c)


def _sc(vs, rank):
    """Make a (possibly batched) scalar value broadcast against `rank` axes."""
    if type(vs) is not np.ndarray:                      # Python / NumPy number
        return vs
    s = vs.shape
    if rank == 0 or s == (1,) or s == ():
        return vs
    return vs.reshape(s + (1,) * rank)


def _check_shape(ka, va, vb):
    r = _RANK[ka]
    if va.shape[-r:] != vb.shape[-r:]:
        raise ValueError("shape mismatch: %s vs %s" % (va.shape, vb.shape))


def _check_n(va, vb):
    if va.shape[-1] != vb.shape[-1]:
        raise ValueError("matrix size mismatch: %s vs %s" % (va.shape, vb.shape))


def _T(a):
    return np.swapaxes(a, -1, -2)


def _additive(a, b, sub):
    # operand unpacking is inlined: this is the hot path
    ka = getattr(a, '_kind', None)          # no exception is raised for plain numbers
    if ka is None:
        ca = _scalar_cplx(a)
        if ca is None:
            return NotImplemented
        ka, va = 'S', a
    else:
        va, ca = a.value, a._cplx
    kb = getattr(b, '_kind', None)
    if kb is None:
        cb = _scalar_cplx(b)
        if cb is None:
            return NotImplemented
        kb, vb = 'S', b
    else:
        vb, cb = b.value, b._cplx
    if ka == kb:
        if ka != 'S':
            _check_shape(ka, va, vb)
        k = ka
    elif ka == 'S':
        k, va = kb, _sc(va, _RANK[kb])
    elif kb == 'S':
        k, vb = ka, _sc(vb, _RANK[ka])
    elif ka == 'M' and kb == 'T':
        _check_n(va, vb)
        k, va = 'T', va[..., None, :, :]
    elif ka == 'T' and kb == 'M':
        _check_n(va, vb)
        k, vb = 'T', vb[..., None, :, :]
    else:
        return NotImplemented
    obj = object.__new__((_CLS_C if (ca or cb) else _CLS_R)[k])
    obj.value = va - vb if sub else va + vb
    return obj


def _multiply(a, b):
    ka = getattr(a, '_kind', None)          # no exception is raised for plain numbers
    if ka is None:
        ca = _scalar_cplx(a)
        if ca is None:
            return NotImplemented
        ka, va = 'S', a
    else:
        va, ca = a.value, a._cplx
    kb = getattr(b, '_kind', None)
    if kb is None:
        cb = _scalar_cplx(b)
        if cb is None:
            return NotImplemented
        kb, vb = 'S', b
    else:
        vb, cb = b.value, b._cplx
    if ka == 'M' and kb == 'M':
        if va.shape[-1] != vb.shape[-1]:
            _check_n(va, vb)
        k, res = 'M', (np.dot(va, vb) if (va.ndim == 2 and vb.ndim == 2) else va @ vb)
    elif ka == 'S':
        if kb == 'S':
            k, res = 'S', va * vb
        else:
            k, res = kb, (_sc(va, _RANK[kb]) if type(va) is np.ndarray else va) * vb
    elif kb == 'S':
        k, res = ka, va * (_sc(vb, _RANK[ka]) if type(vb) is np.ndarray else vb)
    elif ka == 'V' and kb == 'V':
        _check_shape('V', va, vb)
        k, res = 'V', va * vb
    elif ka == 'M' and kb == 'V':
        _check_n(va, vb)
        k = 'V'
        res = np.dot(va, vb) if (va.ndim == 2 and vb.ndim == 1) else (va @ vb[..., None])[..., 0]
    elif ka == 'V' and kb == 'M':
        _check_n(va, vb)
        k = 'V'
        res = np.dot(va, vb) if (va.ndim == 1 and vb.ndim == 2) else (va[..., None, :] @ vb)[..., 0, :]
    elif ka == 'T' and kb == 'T':
        _check_shape('T', va, vb)
        k, res = 'T', va @ vb
    elif ka == 'T' and kb == 'M':
        _check_n(va, vb)
        k, res = 'T', va @ vb[..., None, :, :]
    elif ka == 'M' and kb == 'T':
        _check_n(va, vb)
        k, res = 'T', va[..., None, :, :] @ vb
    else:
        return NotImplemented
    obj = object.__new__((_CLS_C if (ca or cb) else _CLS_R)[k])
    obj.value = res
    return obj


def _divide(a, b):
    oa, ob = _operand(a), _operand(b)
    if oa is None or ob is None or ob[0] != 'S':
        return NotImplemented
    ka, va, ca = oa
    _, vb, cb = ob
    return _make(ka, ca or cb, va / _sc(vb, _RANK[ka]))


def _power(a, b):
    oa = _operand(a)
    if oa is None:
        return NotImplemented
    ka, va, ca = oa
    if ka == 'S':
        ob = _operand(b)
        if ob is None or ob[0] != 'S':
            return NotImplemented
        return _make('S', ca or ob[2], va ** ob[1])
    if ka in ('M', 'T') and isinstance(b, (int, np.integer)) and not isinstance(b, bool):
        return _make(ka, ca, np.linalg.matrix_power(va, int(b)))
    return NotImplemented


def _readonly_dim(getter, name):
    """Size attributes are derived from value.shape; assigning the same size is
    accepted, anything else is an error."""
    def setter(self, v):
        if v != getter(self):
            raise AttributeError("%s is derived from value.shape" % name)
    return property(getter, setter)


# ------------------------------------------------------------------- base ---
class _Tensor:
    # make NumPy scalars/arrays defer to our reflected operators
    __array_ufunc__ = None

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if '_cplx' in cls.__dict__:
            cls._ck = 'c' if cls._cplx else 'f'         # expected dtype kind of value

    # arithmetic -------------------------------------------------------
    def __add__(self, other):
        return _additive(self, other, False)

    def __radd__(self, other):
        return _additive(other, self, False)

    def __sub__(self, other):
        return _additive(self, other, True)

    def __rsub__(self, other):
        return _additive(other, self, True)

    def __mul__(self, other):
        return _multiply(self, other)

    def __rmul__(self, other):
        return _multiply(other, self)

    def __truediv__(self, other):
        return _divide(self, other)

    def __rtruediv__(self, other):
        return _divide(other, self)

    def __pow__(self, n, mod=None):
        return NotImplemented if mod is not None else _power(self, n)

    def __rpow__(self, base):
        return _power(base, self)

    def __neg__(self):
        return _make(self._kind, self._cplx, -self.value)

    def __pos__(self):
        return self.copy()

    # element-wise structure ------------------------------------------
    def copy(self):
        return _make(self._kind, self._cplx, self.value.copy())

    def conj(self):
        v = np.conj(self.value) if self._cplx else self.value.copy()
        return _make(self._kind, self._cplx, v)

    def re(self):
        v = np.real(self.value).copy() if self._cplx else self.value.copy()
        return _make(self._kind, False, v)

    def im(self):
        v = np.imag(self.value).copy() if self._cplx else np.zeros_like(self.value)
        return _make(self._kind, False, v)

    def transpose(self):
        return self.copy()

    def adj(self):
        return self.conj()

    def __repr__(self):
        return "%s(%r)" % (type(self).__name__, self.value)


# ----------------------------------------------------------------- scalars ---
class Real(_Tensor):
    _kind, _cplx = 'S', False

    def __init__(self, value=0):
        self.value = np.atleast_1d(_coerce(value, self._cplx))

    def __abs__(self):
        return _make('S', False, np.abs(self.value))


class Complex(Real):
    _cplx = True


# ----------------------------------------------------------------- vectors ---
class _VectorBase(_Tensor):
    _kind = 'V'
    Nd = _readonly_dim(lambda self: self.value.shape[-1], 'Nd')

    def __init__(self, Nd=None, value=None):
        if value is not None:
            a = value if (type(value) is np.ndarray and value.dtype.kind == self._ck) \
                else _coerce(value, self._cplx)
            if a.ndim < 1:
                raise ValueError("value must be at least 1-dimensional")
            if Nd is not None and a.shape[-1] != Nd:
                raise ValueError("Nd=%s does not match value of shape %s" % (Nd, a.shape))
            self.value = a
        elif Nd is not None:
            self.value = np.zeros(Nd, dtype=complex if self._cplx else float)
        else:
            raise ValueError("%s needs Nd or value" % type(self).__name__)

    def __getitem__(self, mu):
        return _make('S', self._cplx, np.array(self.value[..., _index(mu)], ndmin=1))

    def poke_component(self, mu, m):
        o = _operand(m)
        if o is None or o[0] != 'S':
            raise TypeError("a scalar is required, got %s" % type(m).__name__)
        _, val, cplx = o
        if cplx and not self._cplx:
            raise TypeError("cannot store a complex value in a real vector")
        if getattr(val, 'shape', None) == (1,):
            val = val[0]
        self.value[..., _index(mu)] = val

    def dot(self, rhs):
        """Bilinear product sum_i a_i b_i (no conjugation); returns a scalar."""
        return self._dot(rhs, False)

    def vdot(self, rhs):
        """Inner product sum_i conj(a_i) b_i; returns a scalar."""
        return self._dot(rhs, True)

    def _dot(self, rhs, conj):
        if not isinstance(rhs, _VectorBase):
            raise TypeError("dot needs a vector, got %s" % type(rhs).__name__)
        _check_shape('V', self.value, rhs.value)
        a = np.conj(self.value) if (conj and self._cplx) else self.value
        b = rhs.value
        r = np.dot(a, b) if (a.ndim == 1 and b.ndim == 1) else np.einsum('...i,...i->...', a, b)
        return _make('S', self._cplx or rhs._cplx, np.atleast_1d(r))

    def norm(self):
        return _make('S', False, np.atleast_1d(np.linalg.norm(self.value, axis=-1)))


class VectorReal(_VectorBase):
    _cplx = False


class VectorComplex(_VectorBase):
    _cplx = True


# ------------------------------------------------- matrices / vec of matrices ---
class _MatrixLike(_Tensor):
    N = _readonly_dim(lambda self: self.value.shape[-1], 'N')

    def transpose(self):
        return _make(self._kind, self._cplx, _T(self.value).copy())

    def adj(self):
        v = np.conjugate(_T(self.value), order='C') if self._cplx else _T(self.value).copy()
        return _make(self._kind, self._cplx, v)

    def trace(self):
        t = self.value.diagonal(axis1=-2, axis2=-1).sum(axis=-1)
        return _make('S' if self._kind == 'M' else 'V', self._cplx, np.atleast_1d(t))

    def det(self):
        d = np.linalg.det(self.value)
        return _make('S' if self._kind == 'M' else 'V', self._cplx, np.atleast_1d(d))

    def inv(self):
        return _make(self._kind, self._cplx, np.linalg.inv(self.value))

    def norm(self):
        """Frobenius norm (of each matrix)."""
        n = np.linalg.norm(self.value, axis=(-2, -1))
        return _make('S' if self._kind == 'M' else 'V', False, np.atleast_1d(n))


class _MatrixBase(_MatrixLike):
    _kind = 'M'

    def __init__(self, N=None, value=None):
        if value is not None:
            a = value if (type(value) is np.ndarray and value.dtype.kind == self._ck) \
                else _coerce(value, self._cplx)
            if a.ndim < 2 or a.shape[-1] != a.shape[-2]:
                raise ValueError("value must be square, got shape %s" % (a.shape,))
            if N is not None and a.shape[-1] != N:
                raise ValueError("N=%s does not match value of shape %s" % (N, a.shape))
            self.value = a
        elif N is not None:
            self.value = np.zeros((N, N), dtype=complex if self._cplx else float)
        else:
            raise ValueError("%s needs N or value" % type(self).__name__)


class RealMatrix(_MatrixBase):
    _cplx = False


class ComplexMatrix(RealMatrix):
    _cplx = True


class Identity(RealMatrix):
    def __init__(self, N):
        super().__init__(value=np.eye(N))


class _VecMatBase(_MatrixLike):
    _kind = 'T'
    Nd = _readonly_dim(lambda self: self.value.shape[-3], 'Nd')

    def __init__(self, Nd=None, N=None, value=None):
        if value is not None:
            a = value if (type(value) is np.ndarray and value.dtype.kind == self._ck) \
                else _coerce(value, self._cplx)
            if a.ndim < 3 or a.shape[-1] != a.shape[-2]:
                raise ValueError("value must have shape (Nd, N, N), got %s" % (a.shape,))
            if (Nd is not None and a.shape[-3] != Nd) or (N is not None and a.shape[-1] != N):
                raise ValueError("Nd=%s, N=%s do not match value of shape %s" % (Nd, N, a.shape))
            self.value = a
        elif Nd is not None and N is not None:
            self.value = np.zeros((Nd, N, N), dtype=complex if self._cplx else float)
        else:
            raise ValueError("%s needs Nd and N, or value" % type(self).__name__)

    def __getitem__(self, mu):
        """Component mu as a matrix. It is a view: writing to it changes self."""
        return _make('M', self._cplx, self.value[..., _index(mu), :, :])

    def poke_component(self, mu, m):
        if isinstance(m, np.ndarray):
            arr, cplx = m, m.dtype.kind == 'c'
        else:
            o = _operand(m)
            if o is None or o[0] != 'M':
                raise TypeError("a matrix is required, got %s" % type(m).__name__)
            _, arr, cplx = o
        if cplx and not self._cplx:
            raise TypeError("cannot store a complex matrix in a real vector of matrices")
        if arr.shape[-2:] != (self.N, self.N):
            raise ValueError("expected an %dx%d matrix, got %s" % (self.N, self.N, arr.shape))
        self.value[..., _index(mu), :, :] = arr


class VectorRealMatrix(_VecMatBase):
    _cplx = False


class VectorComplexMatrix(_VecMatBase):
    _cplx = True


_CLS_R = {'S': Real, 'V': VectorReal, 'M': RealMatrix, 'T': VectorRealMatrix}
_CLS_C = {'S': Complex, 'V': VectorComplex, 'M': ComplexMatrix, 'T': VectorComplexMatrix}
