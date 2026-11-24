"""Tests of SU(N) built from SU(2) subgroups (pylattice.core.groups.su): the embedding, the 2x2 blocks and the
Cabibbo-Marinari heat-bath step. No MPI. Run with:  pytest tests/core/test_su.py
"""
import numpy as np
import pytest
from scipy.special import iv

from pylattice.core.groups import su, su2 as sun

EPS = 1e-13
I2 = np.eye(2)


def dagger(M):
    return np.conj(np.swapaxes(M, -1, -2))


def assert_su2(U, tol=1e-12):
    U = np.asarray(U)
    assert np.abs(U @ dagger(U) - I2).max() < tol and np.abs(dagger(U) @ U - I2).max() < tol
    assert np.abs(np.linalg.det(U) - 1).max() < tol


def haar(n, seed):
    return sun.to_matrix(sun.random_quaternion(np.random.default_rng(seed), size=n))


# ------------------------------------------------------------------------- SU(2) inside SU(N)
@pytest.mark.parametrize('N,i,j', [(2, 0, 1), (3, 0, 1), (3, 1, 2), (3, 0, 2), (4, 3, 1)])
def test_embedded_su2_is_in_sun_and_acts_on_two_rows_only(N, i, j):
    U = haar(10, 38)
    E = su.embed_su2(U, i, j, N)
    assert E.shape == (10, N, N)
    assert np.abs(E @ dagger(E) - np.eye(N)).max() < 1e-12 and np.abs(np.linalg.det(E) - 1).max() < 1e-12
    others = [x for x in range(N) if x not in (i, j)]
    for x in others:                                                       # the identity outside the block
        assert np.abs(E[:, x, :] - np.eye(N)[x]).max() == 0 and np.abs(E[:, :, x] - np.eye(N)[:, x]).max() == 0
    assert np.array_equal(E[:, [i, j]][:, :, [i, j]], U)


def test_embedding_is_a_group_homomorphism():
    A, B = haar(10, 39), haar(10, 40)
    assert np.abs(su.embed_su2(A, 0, 2, 3) @ su.embed_su2(B, 0, 2, 3) - su.embed_su2(A @ B, 0, 2, 3)).max() < 1e-12
    assert np.abs(su.embed_su2(A, 0, 1, 3) @ su.embed_su2(B, 1, 2, 3) - su.embed_su2(B, 1, 2, 3) @ su.embed_su2(A, 0, 1, 3)).max() > 1e-3   # different subgroups share a row: they do not commute


def test_su2_block_of_an_embedded_element_is_the_element():
    U = haar(10, 41)
    k, W = su.su2_block(su.embed_su2(U, 1, 3, 4), 1, 3)
    assert np.abs(k - 1).max() < EPS and np.abs(W - U).max() < EPS


def test_su2_block_picks_the_right_rows_and_columns():
    rng = np.random.default_rng(42)
    M = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    for i, j in ((0, 1), (0, 2), (1, 2), (2, 0)):
        k, W = su.su2_block(M, i, j)
        k2, W2 = sun.su2_decompose(M[np.ix_([i, j], [i, j])])
        assert k == k2 and np.array_equal(W, W2)
    for bad in ((0, 0), (0, 3), (-1, 1)):
        with pytest.raises(ValueError):
            su.su2_block(M, *bad)
        with pytest.raises(ValueError):
            su.embed_su2(I2, bad[0], bad[1], 3)


def random_sun(n, N, seed):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((n, N, N)) + 1j * rng.standard_normal((n, N, N)))
    d = np.diagonal(r, axis1=1, axis2=2)
    q = q * (d / np.abs(d))[:, None, :]                                     # the phases that make Q Haar-distributed in U(N)
    return q / (np.linalg.det(q) ** (1 / N))[:, None, None]                 # det = 1: a Haar element of SU(N) up to the centre


def test_cabibbo_marinari_step_for_n_equal_2_is_the_su2_heat_bath():
    # the step on the only subgroup of SU(2) draws U with weight exp(c Re tr(U V)): exactly heatbath_link(V, 2c)
    rng = np.random.default_rng(44)
    V = np.broadcast_to(haar(1, 45)[0] * 1.7, (100000, 2, 2))
    U = su.cabibbo_marinari_step(haar(100000, 46), V, 0.9, 0, 1, rng)
    assert_su2(U)
    k = 1.7
    assert abs(np.einsum('nij,nji->n', U, V).real.mean() - 2 * k * iv(2, 2 * 0.9 * k) / iv(1, 2 * 0.9 * k)) < 0.01


@pytest.mark.parametrize('N', [3, 4])
def test_cabibbo_marinari_sweep_stays_in_sun_and_forgets_where_it_started(N):
    rng = np.random.default_rng(47)
    n, c = 4000, 0.4
    V = np.broadcast_to(sum(0.5 * (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) for _ in range(3)), (n, N, N))
    blocks = [(i, j) for i in range(N) for j in range(i + 1, N)]
    means = []
    for U in (np.broadcast_to(np.eye(N, dtype=complex), (n, N, N)).copy(), random_sun(n, N, 48)):
        history = []
        for sweep in range(12):
            for i, j in blocks:
                U = su.cabibbo_marinari_step(U, V, c, i, j, rng)
            history.append(np.einsum('nij,nji->n', U, V).real.mean())
        assert np.abs(U @ dagger(U) - np.eye(N)).max() < 1e-10 and np.abs(np.linalg.det(U) - 1).max() < 1e-10   # still in SU(N)
        means.append(np.mean(history[6:]))
    assert abs(means[0] - means[1]) < 0.08 * abs(means[0]) + 0.05            # the same equilibrium from the cold and the hot start
    haar_value = np.einsum('nij,nji->n', random_sun(n, N, 49), V).real.mean()
    assert abs(haar_value) < 0.2                                            # no weight: tr(U V) averages to 0
    assert min(means) > haar_value + 0.5                                     # the weight pulls tr(U V) up, a long way


def test_cabibbo_marinari_with_zero_coupling_is_haar_on_the_subgroup():
    rng = np.random.default_rng(50)
    n = 100000
    U = np.broadcast_to(np.eye(3, dtype=complex), (n, 3, 3)).copy()
    U = su.cabibbo_marinari_step(U, np.ones((n, 3, 3), dtype=complex), 0.0, 0, 2, rng)
    q = sun.to_quaternion(U[:, [0, 2]][:, :, [0, 2]])
    assert np.abs(q.mean(axis=0)).max() < 0.006 and np.abs(q.T @ q / n - np.eye(4) / 4).max() < 0.006
    assert np.abs(U[:, 1, 1] - 1).max() == 0 and np.abs(U[:, 1, [0, 2]]).max() == 0   # the row of the third index is untouched
