"""The arithmetic of the lattice classes (pylattice.core.grid) acts on all the sites at once. Each result is
compared with the same operation done site by site with numpy, and the precision of the data is kept.
One process. Run with:  pytest tests/core/test_grid_vectorized.py
"""
import numpy as np
import pytest

from pylattice.core.grid import (LatticeReal, LatticeComplex, LatticeRealMatrix, LatticeComplexMatrix, LatticeVectorReal,
                                 LatticeVectorComplex, LatticeVectorRealMatrix, LatticeVectorComplexMatrix)
from pylattice.core.parallel import CartesianComm
from pylattice.core.tensors import Real, Complex, RealMatrix, ComplexMatrix

GRID, N, ND = [3, 4], 3, 2
SITES = 12
CC = CartesianComm(mpigrid=[1, 1])
rng = np.random.default_rng(0)


def rnd(shape, cplx, dtype=None):
    x = rng.standard_normal(shape)
    out = x + 1j * rng.standard_normal(shape) if cplx else x
    return out.astype(dtype or ('complex64' if cplx else 'float32'))


def lat(cls, value, **kw):
    out = cls(grid=GRID, cartesiancomm=CC, **kw)
    out.value = value
    return out


def cmat(dtype='complex64'):
    return lat(LatticeComplexMatrix, rnd((SITES, N, N), True, dtype), N=N)


def vmat(dtype='complex64'):
    return lat(LatticeVectorComplexMatrix, rnd((SITES, ND, N, N), True, dtype), Nd=ND, N=N)


def close(a, b, tol=1e-4):
    assert a.shape == b.shape and np.abs(a - b).max() < tol


@pytest.mark.parametrize('dtype,cplx,cls', [('float32', False, LatticeRealMatrix), ('complex64', True, LatticeComplexMatrix)])
def test_matrix_lattice_unary_operations_match_a_site_loop(dtype, cplx, cls):
    A = lat(cls, rnd((SITES, N, N), cplx), N=N)
    close(A.transpose().value, np.array([m.T for m in A.value]))
    close(A.trace().value, np.array([np.trace(m) for m in A.value]))
    close(A.det().value, np.array([np.linalg.det(m) for m in A.value]))
    close(A.inv().value, np.array([np.linalg.inv(m) for m in A.value]), 1e-3)
    assert A.transpose().value.dtype == A.value.dtype and A.inv().value.dtype == A.value.dtype
    if cplx:
        close(A.conj().value, np.conj(A.value))
        close(A.adj().value, np.array([m.conj().T for m in A.value]))


def test_matrix_lattice_products():
    A, B = cmat(), cmat()
    close((A * B).value, np.array([a @ b for a, b in zip(A.value, B.value)]))
    t = ComplexMatrix(N=N, value=rnd((N, N), True))
    close((A * t).value, np.array([a @ t.value for a in A.value]))
    close((A * Complex(1 + 2j)).value, A.value * (1 + 2j))
    close((A * 2.5).value, A.value * 2.5)
    close((A + t).value, A.value + t.value)
    close((A - B).value, A.value - B.value)


def test_vector_of_matrices_operations_match_a_site_and_direction_loop():
    U, V, A = vmat(), vmat(), cmat()
    t = ComplexMatrix(N=N, value=rnd((N, N), True))
    ref = lambda f: np.array([[f(s, mu) for mu in range(ND)] for s in range(SITES)])
    close(U.transpose().value, ref(lambda s, mu: U.value[s, mu].T))
    close(U.conj().value, np.conj(U.value))
    close(U.adj().value, ref(lambda s, mu: U.value[s, mu].conj().T))
    close(U.trace().value, ref(lambda s, mu: np.trace(U.value[s, mu])))
    close(U.det().value, ref(lambda s, mu: np.linalg.det(U.value[s, mu])), 1e-3)
    close(U.inv().value, ref(lambda s, mu: np.linalg.inv(U.value[s, mu])), 1e-3)
    close((U * V).value, ref(lambda s, mu: U.value[s, mu] @ V.value[s, mu]))
    close((U * A).value, ref(lambda s, mu: U.value[s, mu] @ A.value[s]))
    close((U * t).value, ref(lambda s, mu: U.value[s, mu] @ t.value))
    close((U * 2).value, U.value * 2)
    close((U + A).value, ref(lambda s, mu: U.value[s, mu] + A.value[s]))
    close((U - A).value, ref(lambda s, mu: U.value[s, mu] - A.value[s]))
    close((U + t).value, ref(lambda s, mu: U.value[s, mu] + t.value))
    close((U + 1.5).value, U.value + 1.5)


def test_vector_lattices():
    a = lat(LatticeVectorReal, rnd((SITES, ND), False), Nd=ND)
    b = lat(LatticeVectorReal, rnd((SITES, ND), False), Nd=ND)
    s = lat(LatticeReal, rnd((SITES,), False))
    close((a * b).value, np.array([np.dot(x, y) for x, y in zip(a.value, b.value)]))
    close((a * s).value, a.value * s.value[:, None])
    close(a.transpose().value, a.value)
    z = lat(LatticeVectorComplex, rnd((SITES, ND), True), Nd=ND)
    c = lat(LatticeComplex, rnd((SITES,), True))
    close((z * a).value, np.array([np.dot(x, y) for x, y in zip(z.value, a.value)]))
    close((z * c).value, z.value * c.value[:, None])
    close(z.conj().value, np.conj(z.value))
    close(z.transpose().value, z.value)


def test_double_precision_is_kept_by_every_operation():
    U, V, A = vmat('complex128'), vmat('complex128'), cmat('complex128')
    for r in (U * V, U * A, U + V, U.adj(), U.conj(), U.trace(), U.inv(), A * A, A.adj(), A.trace(), A.inv(), U.transpose()):
        assert r.value.dtype == np.complex128
    x = rnd((SITES, N, N), True, 'complex128')
    assert np.abs((lat(LatticeComplexMatrix, x, N=N) * lat(LatticeComplexMatrix, x, N=N)).value - x @ x).max() < 1e-13


def test_reductions_keep_the_precision_of_the_data():
    tiny = 1e-12
    for cls, kw, dtype in ((LatticeComplex, {}, 'complex128'), (LatticeReal, {}, 'float64')):
        L = lat(cls, np.full(SITES, 1 + tiny, dtype=dtype))
        assert L.reducesum().dtype == np.dtype(dtype) and abs(L.reducesum() - SITES * (1 + tiny)) < 1e-13
        assert abs(L.average() - (1 + tiny)) < 1e-14
    assert lat(LatticeComplex, np.ones(SITES, dtype='complex64')).reducesum().dtype == np.complex64


def test_copy_and_shifts_do_not_touch_the_original_and_agree_with_moveforward():
    A = cmat()
    before = A.value.copy()
    B = A.copy()
    B.value[:] = 0
    assert np.array_equal(A.value, before)
    for mu in range(2):
        F, K = A.shift_forward(mu), A.shift_backward(mu)
        assert np.array_equal(A.value, before) and F is not A
        M = A.copy(); M.moveforward(mu)
        assert np.array_equal(F.value, M.value)
        assert np.array_equal(F.shift_backward(mu).value, before) and np.array_equal(K.shift_forward(mu).value, before)
        grid = np.arange(SITES).reshape(GRID)                                    # value at x + mu
        assert np.array_equal(F.value, before[np.roll(grid, -1, axis=mu).ravel()])


def test_peek_and_poke_index_of_the_vector_lattices():
    U = vmat('complex128')
    P = U.peek_index(1)
    assert isinstance(P, LatticeComplexMatrix) and P.value.dtype == np.complex128 and np.array_equal(P.value, U.value[:, 1])
    P.value[:] = 0
    assert np.abs(U.value[:, 1]).max() > 0                                         # a copy
    U.poke_index(0, P)
    assert np.abs(U.value[:, 0]).max() == 0
    z = lat(LatticeVectorComplex, rnd((SITES, ND), True), Nd=ND)
    assert isinstance(z.peek_index(0), LatticeComplex)
    r = lat(LatticeVectorRealMatrix, rnd((SITES, ND, N, N), False), Nd=ND, N=N)
    assert isinstance(r.peek_index(0), LatticeRealMatrix)


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_') and not hasattr(f, 'pytestmark'):
            f(); print('[+]', name)
    for dtype, cplx, cls in (('float32', False, LatticeRealMatrix), ('complex64', True, LatticeComplexMatrix)):
        test_matrix_lattice_unary_operations_match_a_site_loop(dtype, cplx, cls); print('[+] unary', cls.__name__)
