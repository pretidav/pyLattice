"""Tests of linalg.tensors.

Part 1 checks hand-computed values. Part 2 checks every operation and every pair of
classes against NumPy references, including the type and dtype of each result.

Run with:  pytest tests/core/test_tensors.py
"""
import itertools

import numpy as np
import pytest

from pylattice.core.tensors import (Real, Complex, VectorReal, VectorComplex, RealMatrix, ComplexMatrix,
                            VectorRealMatrix, VectorComplexMatrix, Identity, Kroneker)

N, ND = 3, 4
rng = np.random.default_rng(1234)

# (kind, is_complex) -> class.  kinds: S scalar, V vector, M matrix, T vector of matrices
CLASSES = {('S', False): Real, ('S', True): Complex,
           ('V', False): VectorReal, ('V', True): VectorComplex,
           ('M', False): RealMatrix, ('M', True): ComplexMatrix,
           ('T', False): VectorRealMatrix, ('T', True): VectorComplexMatrix}
SHAPES = {'S': (1,), 'V': (ND,), 'M': (N, N), 'T': (ND, N, N)}
ALL = list(CLASSES)


def check(res, kind, cplx, expected, tol=1e-10):
    """res must be exactly the class for (kind, cplx), with matching dtype kind, shape and values."""
    assert type(res) is CLASSES[(kind, cplx)], (type(res).__name__, kind, cplx)
    assert (res.value.dtype.kind == 'c') == cplx, res.value.dtype
    np.testing.assert_allclose(res.value, expected, rtol=tol, atol=tol)


# =========================================================================
# Part 1: hand-computed values
# =========================================================================
tol32 = 1e-5          # for the float32 / complex64 data below

a = Real(10)
b = Real(-3)
c = 10
ac = Complex(3 + 5j)
bc = Complex(2 - 8j)

A = RealMatrix(value=np.array([[0, 1], [3, 4]]))
B = RealMatrix(value=np.array([[2, 1], [0, 2]]))
Ac = ComplexMatrix(value=np.array([[0, 1j], [3j, 4]], dtype=complex))
Bc = ComplexMatrix(value=np.array([[2, 1 + 7j], [0, 2 + 9j]], dtype=complex))


def test_real():
    check(a + b, 'S', False, [7.])
    check(a - b, 'S', False, [13.])
    check(a * b, 'S', False, [-30.])
    check(a * c, 'S', False, [100.])
    check(a ** c, 'S', False, [1e10])
    check(a ** b, 'S', False, [0.001])
    check(a * bc, 'S', True, [20 - 80j])


def test_complex():
    check(ac + bc, 'S', True, [5 - 3j])
    check(ac - bc, 'S', True, [1 + 13j])
    check(ac * bc, 'S', True, [46 - 14j])
    check(ac * a, 'S', True, [30 + 50j])
    check(ac.conj(), 'S', True, [3 - 5j])
    check(ac.re(), 'S', False, [3.])
    check(ac.im(), 'S', False, [5.])


def test_realmatrix():
    check(A + B, 'M', False, [[2, 2], [3, 6]])
    check(A - B, 'M', False, [[-2, 0], [3, 2]])
    check(A * B, 'M', False, [[0, 2], [6, 11]])
    check(B.trace(), 'S', False, [4.])
    check(B.det(), 'S', False, [4.])
    check(B.inv(), 'M', False, [[0.5, -0.25], [0, 0.5]])
    check(B.inv() * B, 'M', False, np.eye(2))


def test_complexmatrix():
    check(Ac.im(), 'M', False, [[0, 1], [3, 0]])
    check(Ac.re(), 'M', False, [[0, 0], [0, 4]])
    check(Ac + Bc, 'M', True, [[2, 1 + 8j], [3j, 6 + 9j]])
    check(Ac - Bc, 'M', True, [[-2, -1 - 6j], [3j, 2 - 9j]])
    check(Ac * Bc, 'M', True, [[0, -9 + 2j], [6j, -13 + 39j]])
    check(Ac * B, 'M', True, [[0, 2j], [6j, 8 + 3j]])
    check(A * Bc, 'M', True, [[0, 2 + 9j], [6, 11 + 57j]])
    check(Bc.trace(), 'S', True, [4 + 9j])
    assert Bc.trace().value[0] == 4 + 9j            # SU2.py relies on .trace().value[0]
    check(Bc.det(), 'S', True, [4 + 18j])
    check(Bc.inv() * Bc, 'M', True, np.eye(2))
    check(Bc.adj(), 'M', True, [[2, 0], [1 - 7j, 2 - 9j]])


def test_vectorreal():
    v = VectorReal(value=np.array([2, 3]))
    w = VectorReal(value=np.array([5, 7]))
    M = RealMatrix(value=np.array([[0, 1], [3, 4]]))
    check(v + w, 'V', False, [7, 10])
    check(v - w, 'V', False, [-3, -4])
    check(v * w, 'V', False, [10, 21])
    check(M * v, 'V', False, [3, 18])
    check(v * M, 'V', False, [9, 14])               # v^T M
    check(v.dot(w), 'S', False, [31])
    check(v.vdot(w), 'S', False, [31])
    check(v.norm(), 'S', False, [np.sqrt(13)])


def test_vectorcomplex():
    v = VectorComplex(value=np.array([2, 3j]))
    w = VectorComplex(value=np.array([5j, 7]))
    M = ComplexMatrix(value=np.array([[0, 1j], [3, 4]]))
    check(v + w, 'V', True, [2 + 5j, 7 + 3j])
    check(v - w, 'V', True, [2 - 5j, 3j - 7])
    check(v * w, 'V', True, [10j, 21j])
    check(M * v, 'V', True, [-3, 6 + 12j])
    check(v.dot(w), 'S', True, [31j])               # bilinear: no conjugation
    check(v.vdot(w), 'S', True, [-11j])             # conj(v) . w
    check(v.norm(), 'S', False, [np.sqrt(13)])


def test_vectorrealmatrix():
    A = VectorRealMatrix(value=np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype='float32'))
    B = VectorRealMatrix(value=np.array([[[1, -2], [-3, 4]], [[5, -6], [7, 8]]], dtype='float32'))
    check(A + B, 'T', False, [[[2, 0], [0, 8]], [[10, 0], [14, 16]]], tol32)
    check(A - B, 'T', False, [[[0, 4], [6, 0]], [[0, 12], [0, 0]]], tol32)
    check(A * B, 'T', False, [[[-5, 6], [-9, 10]], [[67, 18], [91, 22]]], tol32)
    check(A.transpose(), 'T', False, [[[1, 3], [2, 4]], [[5, 7], [6, 8]]], tol32)
    check(A.det(), 'V', False, [-2, -2], tol32)
    check(A.trace(), 'V', False, [5, 13], tol32)
    check(A.inv(), 'T', False, [[[-2, 1], [1.5, -0.5]], [[-4, 3], [3.5, -2.5]]], tol32)
    assert A.Nd == 2 and A.N == 2


def test_vectorcomplexmatrix():
    A = VectorComplexMatrix(value=np.array([[[1, 2j], [3j, 4]], [[5j, 6], [7, 8j]]], dtype='complex64'))
    B = VectorComplexMatrix(value=np.array([[[1j, -2], [-3, 4j]], [[5j, -6], [7j, 8]]], dtype='complex64'))
    check(A + B, 'T', True, [[[1 + 1j, -2 + 2j], [-3 + 3j, 4 + 4j]], [[10j, 0], [7 + 7j, 8 + 8j]]], tol32)
    check(A - B, 'T', True, [[[1 - 1j, 2 + 2j], [3 + 3j, 4 - 4j]], [[0, 12], [7 - 7j, -8 + 8j]]], tol32)
    check(A * B, 'T', True, [[[-5j, -10], [-15, 10j]], [[-25 + 42j, 48 - 30j], [-56 + 35j, -42 + 64j]]], tol32)
    check(A.transpose(), 'T', True, [[[1, 3j], [2j, 4]], [[5j, 7], [6, 8j]]], tol32)
    check(A.det(), 'V', True, [10, -82], tol32)
    check(A.trace(), 'V', True, [5, 13j], tol32)
    check(A.inv(), 'T', True, [[[0.4, -0.2j], [-0.3j, 0.1]],
                               [[-0.09756097j, 0.07317073], [0.08536585, -0.06097561j]]], tol32)
    check(A.conj(), 'T', True, [[[1, -2j], [-3j, 4]], [[-5j, 6], [7, -8j]]], tol32)
    check(A.adj(), 'T', True, [[[1, -3j], [-2j, 4]], [[-5j, 7], [6, -8j]]], tol32)
    check(A.re(), 'T', False, [[[1, 0], [0, 4]], [[0, 6], [7, 0]]], tol32)
    check(A.im(), 'T', False, [[[0, 2], [3, 0]], [[5, 0], [0, 8]]], tol32)
    assert A.value.dtype == np.complex64 and (A * B).value.dtype == np.complex64


# =========================================================================
# Part 2: every operation and every pair of classes, against NumPy
# =========================================================================
def data(kind, cplx, batch=(), dtype=None, vlen=ND):
    shape = batch + ((vlen,) if kind == 'V' else SHAPES[kind])
    x = rng.standard_normal(shape)
    if cplx:
        x = x + 1j * rng.standard_normal(shape)
    return x.astype(dtype) if dtype else x


def build(kind, cplx, batch=(), dtype=None, vlen=ND):
    x = data(kind, cplx, batch, dtype, vlen)
    return CLASSES[(kind, cplx)](value=x), x


def ref_add(ka, x, kb, y, sub):
    f = (lambda p, q: p - q) if sub else (lambda p, q: p + q)
    if ka == kb or 'S' in (ka, kb):
        return (kb if ka == 'S' else ka), f(x, y)
    if (ka, kb) == ('M', 'T'):
        return 'T', f(x[None], y)
    if (ka, kb) == ('T', 'M'):
        return 'T', f(x, y[None])
    return None                                     # unsupported pair


def ref_mul(ka, x, kb, y):
    if 'S' in (ka, kb):
        return (kb if ka == 'S' else ka), x * y
    table = {('V', 'V'): 'V', ('M', 'M'): 'M', ('M', 'V'): 'V', ('V', 'M'): 'V',
             ('T', 'T'): 'T', ('T', 'M'): 'T', ('M', 'T'): 'T'}
    if (ka, kb) not in table:
        return None
    return table[(ka, kb)], (x * y if (ka, kb) == ('V', 'V') else x @ y)


# --- binary operators -----------------------------------------------------
@pytest.mark.parametrize('ca,cb', list(itertools.product(ALL, ALL)))
def test_binary_between_all_classes(ca, cb):
    (ka, xa), (kb, xb) = ca, cb
    vlen = N if {ka, kb} == {'V', 'M'} else ND      # M v needs len(v) == N
    X, x = build(ka, xa, vlen=vlen)
    Y, y = build(kb, xb, vlen=vlen)
    for op, exp in [(lambda p, q: p + q, ref_add(ka, x, kb, y, False)),
                    (lambda p, q: p - q, ref_add(ka, x, kb, y, True)),
                    (lambda p, q: p * q, ref_mul(ka, x, kb, y))]:
        if exp is None:
            with pytest.raises(TypeError):
                op(X, Y)
        else:
            check(op(X, Y), exp[0], xa or xb, exp[1])


SCALARS = [2, 2.5, 1 + 2j, np.float32(1.5), np.float64(2.0), np.int64(3), np.complex64(1 - 1j),
           np.complex128(0.5j)]


@pytest.mark.parametrize('cls', ALL)
@pytest.mark.parametrize('s', SCALARS, ids=lambda s: type(s).__name__ + str(s))
def test_scalars_on_both_sides(cls, s):
    kind, cx = cls
    X, x = build(kind, cx)
    cr = cx or isinstance(s, (complex, np.complexfloating))
    check(X + s, kind, cr, x + s)
    check(s + X, kind, cr, s + x)
    check(X - s, kind, cr, x - s)
    check(s - X, kind, cr, s - x)
    check(X * s, kind, cr, x * s)
    check(s * X, kind, cr, s * x)
    check(X / s, kind, cr, x / s)
    if kind == 'S':
        check(s / X, kind, cr, s / x)
    else:
        with pytest.raises(TypeError):
            s / X


@pytest.mark.parametrize('cls', ALL)
def test_division_by_tensor_scalar(cls):
    kind, cx = cls
    X, x = build(kind, cx)
    for cs in (False, True):
        S, s = build('S', cs)
        check(X / S, kind, cx or cs, x / s)


def test_unsupported_operands():
    V, _ = build('V', False)
    M, _ = build('M', False)
    T, _ = build('T', False)
    for bad in (lambda: V + M, lambda: V * T, lambda: T * V, lambda: M / M, lambda: M + 'a',
                lambda: M * None,
                lambda: np.ones((N, N)) * M,         # arrays are not silently absorbed
                lambda: M * np.ones((N, N))):
        with pytest.raises(TypeError):
            bad()


def test_numpy_scalar_on_the_left_is_not_swallowed():
    M, m = build('M', False)
    for s in (np.float32(2), np.float64(2), np.int32(2), np.complex64(2j)):
        r = s * M
        assert isinstance(r, RealMatrix) and not isinstance(r, np.ndarray)
        np.testing.assert_allclose(r.value, s * m)


@pytest.mark.parametrize('kind', ['V', 'M', 'T'])
@pytest.mark.parametrize('ca,cb', list(itertools.product([False, True], repeat=2)))
def test_shape_mismatch_raises(kind, ca, cb):
    X = CLASSES[(kind, ca)](value=data(kind, ca))
    big = {'V': (ND + 1,), 'M': (N + 1, N + 1), 'T': (ND + 1, N, N)}[kind]
    Y = CLASSES[(kind, cb)](value=rng.standard_normal(big))
    for op in (lambda p, q: p + q, lambda p, q: p - q, lambda p, q: p * q):
        with pytest.raises(ValueError):
            op(X, Y)
    if kind == 'T':                                 # same Nd, different N
        Z = CLASSES[(kind, cb)](value=rng.standard_normal((ND, N + 1, N + 1)))
        for op in (lambda p, q: p + q, lambda p, q: p * q):
            with pytest.raises(ValueError):
                op(X, Z)


def test_matrix_with_vector_of_matrices_size_mismatch():
    M2 = RealMatrix(value=np.eye(2))
    T, _ = build('T', False)
    for op in (lambda p, q: p + q, lambda p, q: q - p, lambda p, q: p * q, lambda p, q: q * p):
        with pytest.raises(ValueError):
            op(M2, T)


# --- unary methods ---------------------------------------------------------
@pytest.mark.parametrize('cls', ALL)
def test_structure_methods(cls):
    kind, cx = cls
    X, x = build(kind, cx)
    check(-X, kind, cx, -x)
    check(+X, kind, cx, x)
    check(X.copy(), kind, cx, x)
    check(X.conj(), kind, cx, np.conj(x))
    check(X.re(), kind, False, np.real(x))
    check(X.im(), kind, False, np.imag(x))
    if kind in ('M', 'T'):
        check(X.transpose(), kind, cx, np.swapaxes(x, -1, -2))
        check(X.adj(), kind, cx, np.conj(np.swapaxes(x, -1, -2)))
    else:
        check(X.transpose(), kind, cx, x)
        check(X.adj(), kind, cx, np.conj(x))


@pytest.mark.parametrize('cls', ALL)
def test_results_do_not_alias_operands(cls):
    kind, cx = cls
    X, x = build(kind, cx)
    keep = x.copy()
    for r in (X.copy(), X.conj(), X.re(), X.im(), X.transpose(), X.adj(), -X, X + 0, X * 1):
        assert not np.shares_memory(r.value, X.value)
        r.value[...] = 123
    np.testing.assert_array_equal(X.value, keep)


@pytest.mark.parametrize('cls', [c for c in ALL if c[0] in 'MT'])
def test_matrix_methods(cls):
    kind, cx = cls
    X, x = build(kind, cx)
    rk = 'S' if kind == 'M' else 'V'
    check(X.trace(), rk, cx, np.atleast_1d(np.trace(x, axis1=-2, axis2=-1)))
    check(X.det(), rk, cx, np.atleast_1d(np.linalg.det(x)))
    check(X.inv(), kind, cx, np.linalg.inv(x))
    check(X.norm(), rk, False, np.atleast_1d(np.linalg.norm(x, axis=(-2, -1))))
    assert X.N == N
    if kind == 'T':
        assert X.Nd == ND


@pytest.mark.parametrize('cls', [c for c in ALL if c[0] == 'V'])
def test_vector_methods(cls):
    _, cx = cls
    X, x = build('V', cx)
    for cw in (False, True):
        W, w = build('V', cw)
        check(X.dot(W), 'S', cx or cw, np.atleast_1d(np.sum(x * w)))
        check(X.vdot(W), 'S', cx or cw, np.atleast_1d(np.vdot(x, w)))
    check(X.norm(), 'S', False, np.atleast_1d(np.linalg.norm(x)))
    assert X.Nd == ND
    with pytest.raises(TypeError):
        X.dot(Real(2))
    with pytest.raises(ValueError):
        X.dot(VectorReal(value=np.ones(ND + 1)))


def test_scalar_abs():
    check(abs(Complex(3 + 4j)), 'S', False, [5.])
    check(abs(Real(-2)), 'S', False, [2.])


# --- power -----------------------------------------------------------------
def test_power():
    check(2 ** Real(3), 'S', False, [8.0])
    check(Complex(1j) ** 2, 'S', True, [-1 + 0j])
    check(Real(4) ** Complex(0.5), 'S', True, [2 + 0j])
    for cx in (False, True):
        M, m = build('M', cx)
        T, t = build('T', cx)
        V, _ = build('V', cx)
        for n in (0, 1, 2, 3, -1, -2):
            check(M ** n, 'M', cx, np.linalg.matrix_power(m, n))
            check(T ** n, 'T', cx, np.linalg.matrix_power(t, n))
        for bad in (lambda: M ** 0.5, lambda: M ** Real(2), lambda: V ** 2, lambda: 2 ** M):
            with pytest.raises(TypeError):
                bad()


# --- constructors ----------------------------------------------------------
@pytest.mark.parametrize('cls', ALL)
def test_defaults_are_zero(cls):
    kind, cx = cls
    c = CLASSES[cls]
    x = {'S': lambda: c(), 'V': lambda: c(Nd=ND), 'M': lambda: c(N=N), 'T': lambda: c(Nd=ND, N=N)}[kind]()
    check(x, kind, cx, np.zeros(SHAPES[kind]))


def test_constructor_dtypes():
    assert VectorReal(value=np.array([1, 2])).value.dtype == np.float64
    assert RealMatrix(value=np.eye(2, dtype=np.float32)).value.dtype == np.float32
    assert ComplexMatrix(value=np.eye(2, dtype=np.float32)).value.dtype == np.complex64
    assert ComplexMatrix(value=np.eye(2)).value.dtype == np.complex128
    assert VectorComplex(value=np.array([1, 2])).value.dtype == np.complex128
    assert Real(3).value.dtype == np.float64 and Real(3).value.shape == (1,)
    assert Complex(3).value.dtype == np.complex128
    assert Real(np.float32(3)).value.dtype == np.float32
    for bad in (lambda: Real(1j), lambda: VectorReal(value=np.array([1j])),
                lambda: RealMatrix(value=np.eye(2) * 1j),
                lambda: VectorRealMatrix(value=np.ones((2, 2, 2)) * 1j)):
        with pytest.raises(TypeError):              # complex data in a real class
            bad()


def test_constructor_validation():
    for bad in (lambda: VectorReal(), lambda: RealMatrix(), lambda: VectorRealMatrix(Nd=2),
                lambda: VectorRealMatrix(N=2), lambda: RealMatrix(value=np.ones((2, 3))),
                lambda: RealMatrix(value=np.ones(3)), lambda: VectorRealMatrix(value=np.ones((2, 2))),
                lambda: VectorReal(Nd=3, value=np.ones(4)), lambda: RealMatrix(N=3, value=np.eye(2)),
                lambda: VectorComplexMatrix(Nd=3, N=2, value=np.ones((2, 2, 2)))):
        with pytest.raises(ValueError):
            bad()


def test_value_is_not_copied_when_dtype_fits():
    x = np.eye(3)
    assert RealMatrix(value=x).value is x
    z = np.eye(3, dtype=complex)
    assert ComplexMatrix(value=z).value is z


def test_dims_follow_value_and_are_read_only():
    M = RealMatrix(N=2)
    M.value = np.eye(3)
    assert M.N == 3
    M.N = 3                                         # same size: accepted
    with pytest.raises(AttributeError):
        M.N = 2
    T = VectorComplexMatrix(Nd=2, N=3)
    assert (T.Nd, T.N) == (2, 3)


def test_identity_and_kroneker():
    I = Identity(3)
    assert isinstance(I, RealMatrix) and I.value.dtype == np.float64
    M, m = build('M', True)
    check(I * M, 'M', True, m)
    check(M * I, 'M', True, m)
    check(I + I, 'M', False, 2 * np.eye(3))
    assert Kroneker(1, 1) == 1 and Kroneker(1, 2) == 0


# --- components (get / poke) -----------------------------------------------
def test_vector_getitem_and_poke():
    for cx in (False, True):
        V, v = build('V', cx)
        for mu in range(ND):
            check(V[mu], 'S', cx, np.array([v[mu]]))
        assert len(list(V)) == ND
        V[0].value[0] = 99                          # a scalar copy, not a view
        np.testing.assert_array_equal(V.value, v)
        with pytest.raises(IndexError):
            V[ND]
        V.poke_component(1, Real(5.0))
        V.poke_component(2, 7)
        V.poke_component(3, np.float32(1.5))
        assert list(V.value[1:]) == [5.0, 7.0, 1.5]
        if cx:
            V.poke_component(0, Complex(2j))
            V.poke_component(1, 3j)
            assert V.value[0] == 2j and V.value[1] == 3j
        else:
            for bad in (Complex(2j), 3j, np.complex64(1j)):
                with pytest.raises(TypeError):
                    V.poke_component(0, bad)
        with pytest.raises(TypeError):
            V.poke_component(0, RealMatrix(N=2))
        with pytest.raises(TypeError):
            V[0:2]


def test_vector_of_matrices_getitem_and_poke():
    for cx in (False, True):
        T, t = build('T', cx)
        for mu in range(ND):
            check(T[mu], 'M', cx, t[mu])
        T[0].value[0, 0] = 42                       # documented: a view
        assert T.value[0, 0, 0] == 42
        T.poke_component(1, RealMatrix(value=np.eye(N)))
        np.testing.assert_array_equal(T.value[1], np.eye(N))
        T.poke_component(2, 2 * np.eye(N))
        np.testing.assert_array_equal(T.value[2], 2 * np.eye(N))
        if cx:
            T.poke_component(3, ComplexMatrix(value=1j * np.eye(N)))
            np.testing.assert_array_equal(T.value[3], 1j * np.eye(N))
        else:
            for bad in (ComplexMatrix(value=1j * np.eye(N)), 1j * np.eye(N)):
                with pytest.raises(TypeError):
                    T.poke_component(0, bad)
        with pytest.raises(ValueError):
            T.poke_component(0, np.eye(N + 1))
        with pytest.raises(TypeError):
            T.poke_component(0, Real(1))


# --- precision and leading batch axes ---------------------------------------
@pytest.mark.parametrize('real_dt,cplx_dt', [(np.float32, np.complex64), (np.float64, np.complex128)])
def test_precision_is_preserved(real_dt, cplx_dt):
    for ka, kb in [('T', 'T'), ('M', 'M'), ('M', 'V'), ('T', 'M'), ('V', 'V')]:
        for cx, dt in ((False, real_dt), (True, cplx_dt)):
            vlen = N if {ka, kb} == {'V', 'M'} else ND
            X, _ = build(ka, cx, dtype=dt, vlen=vlen)
            Y, _ = build(kb, cx, dtype=dt, vlen=vlen)
            results = [X * Y] + ([X + Y, X - Y] if ka == kb else [])
            for r in results:
                assert r.value.dtype == dt, (ka, kb, r.value.dtype)
    M, _ = build('M', True, dtype=cplx_dt)
    for r in (M.inv(), M.adj(), M.conj(), M.trace(), M.det(), M * 2.0, M / 2, 2 * M, -M):
        assert r.value.dtype == cplx_dt
    assert M.re().value.dtype == real_dt and M.norm().value.dtype == real_dt


@pytest.mark.parametrize('batch', [(5,), (2, 3)])
def test_leading_batch_axes(batch):
    for cx in (False, True):
        T, t = build('T', cx, batch)
        U, u = build('T', cx, batch)
        M, m = build('M', cx, batch)
        V, v = build('V', cx, batch, vlen=N)
        # a batch of scalars has the batch shape itself (as trace() returns), no trailing 1
        s = rng.standard_normal(batch) + (1j * rng.standard_normal(batch) if cx else 0)
        S = CLASSES[('S', cx)](value=s)
        check(T * U, 'T', cx, t @ u)
        check(T * M, 'T', cx, t @ m[..., None, :, :])
        check(M * T, 'T', cx, m[..., None, :, :] @ t)
        check(T + M, 'T', cx, t + m[..., None, :, :])
        check(M * V, 'V', cx, np.einsum('...ij,...j->...i', m, v))
        check(V * M, 'V', cx, np.einsum('...i,...ij->...j', v, m))
        check(M * M, 'M', cx, m @ m)
        check(T.adj(), 'T', cx, np.conj(np.swapaxes(t, -1, -2)))
        check(T.trace(), 'V', cx, np.trace(t, axis1=-2, axis2=-1))
        check(M.trace(), 'S', cx, np.trace(m, axis1=-2, axis2=-1))
        check(M.det(), 'S', cx, np.linalg.det(m))
        check(T.inv(), 'T', cx, np.linalg.inv(t))
        check(T ** 2, 'T', cx, t @ t)
        check(V.dot(V), 'S', cx, np.sum(v * v, axis=-1))
        # a batched scalar scales every structured entry of its own batch element
        check(M.trace() * T, 'T', cx, np.trace(m, axis1=-2, axis2=-1)[..., None, None, None] * t)
        check(S * M, 'M', cx, s[..., None, None] * m)
        check(M * S, 'M', cx, m * s[..., None, None])
        check(S * T, 'T', cx, s[..., None, None, None] * t)
        check(S * V, 'V', cx, s[..., None] * v)
        check(S + M, 'M', cx, s[..., None, None] + m)
        check(M / S, 'M', cx, m / s[..., None, None])
        assert T.N == N and T.Nd == ND and V.Nd == N
