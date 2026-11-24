"""Tests of SU(N) (pylattice.core.groups.su): the generators and the algebra, the exponential and logarithm, Haar-random
elements, the projection, the group object, and the SU(2) subgroups with the Cabibbo-Marinari heat-bath step. No MPI.
Run with:  pytest tests/core/test_su.py
"""
import numpy as np
import pytest
from scipy.linalg import expm
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


# ========================================================================= the algebra of SU(N)
NS = [2, 3, 4, 5]


def commutator(A, B):
    return A @ B - B @ A


def unitary_special(U, tol=1e-12):
    U = np.asarray(U)
    return (np.abs(U @ dagger(U) - np.eye(U.shape[-1])).max() < tol) and (np.abs(np.linalg.det(U) - 1).max() < tol)


@pytest.mark.parametrize('N', NS)
def test_generators_are_hermitian_traceless_orthonormal_and_a_basis(N):
    T = su.generators(N)
    d = N * N - 1
    assert T.shape == (d, N, N) and T.dtype == complex
    assert np.abs(T - dagger(T)).max() < EPS and np.abs(np.trace(T, axis1=-2, axis2=-1)).max() < EPS
    assert np.abs(np.einsum('aij,bji->ab', T, T) - np.eye(d) / 2).max() < EPS          # tr(T_a T_b) = delta_ab / 2
    assert np.linalg.matrix_rank(T.reshape(d, -1)) == d                                  # they span the algebra
    T[0, 0, 0] = 99                                                                      # a copy: the module's are not changed
    assert su.generators(N)[0, 0, 0] != 99


def test_generators_are_the_pauli_and_gell_mann_matrices_in_the_usual_order():
    assert np.array_equal(su.generators(2), sun.generators())                           # sigma_a / 2, as in su2
    s3 = 1 / np.sqrt(3)
    gell_mann = [np.array(m, dtype=complex) for m in (
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]], [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 0, 0], [1, 0, 0]], [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 1, 0]], [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
        [[s3, 0, 0], [0, s3, 0], [0, 0, -2 * s3]])]
    assert np.abs(su.generators(3) - np.array(gell_mann) / 2).max() < EPS


@pytest.mark.parametrize('N', NS)
def test_structure_constants_and_d_symbols_satisfy_the_algebra(N):
    T, f, d = su.generators(N), su.structure_constants(N), su.d_symbols(N)
    D = N * N - 1
    assert f.shape == d.shape == (D, D, D) and f.dtype == d.dtype == float
    assert np.abs(f + np.swapaxes(f, 0, 1)).max() < EPS and np.abs(f + np.swapaxes(f, 1, 2)).max() < EPS    # antisymmetric
    assert np.abs(d - np.swapaxes(d, 0, 1)).max() < EPS and np.abs(d - np.swapaxes(d, 1, 2)).max() < EPS    # symmetric
    comm = commutator(T[:, None], T[None, :])                                                 # [T_a, T_b] = i f_abc T_c
    assert np.abs(comm - 1j * np.einsum('abc,cij->abij', f, T)).max() < EPS
    anti = T[:, None] @ T[None, :] + T[None, :] @ T[:, None]                                  # {T_a, T_b} = delta_ab / N + d_abc T_c
    assert np.abs(anti - (np.eye(D)[:, :, None, None] / N * np.eye(N) + np.einsum('abc,cij->abij', d, T))).max() < EPS
    jacobi = np.einsum('abe,cde->abcd', f, f) + np.einsum('bce,ade->abcd', f, f) + np.einsum('cae,bde->abcd', f, f)
    assert np.abs(jacobi).max() < 1e-12                                                       # the Jacobi identity
    assert np.abs(np.einsum('acd,bcd->ab', f, f) - N * np.eye(D)).max() < 1e-12              # f_acd f_bcd = N delta_ab
    f[0, 1, 2] += 1                                                                           # copies
    assert su.structure_constants(N)[0, 1, 2] != f[0, 1, 2]


def test_known_structure_constants():
    f3, d3 = su.structure_constants(3), su.d_symbols(3)
    assert f3[0, 1, 2] == pytest.approx(1) and f3[0, 3, 6] == pytest.approx(0.5) and f3[3, 4, 7] == pytest.approx(np.sqrt(3) / 2)
    assert d3[0, 0, 7] == pytest.approx(1 / np.sqrt(3)) and d3[7, 7, 7] == pytest.approx(-1 / np.sqrt(3))
    f2 = su.structure_constants(2)
    assert all(f2[i, j, k] == sun.structure_constants(i, j, k) for i in range(3) for j in range(3) for k in range(3))
    assert np.abs(su.d_symbols(2)).max() < EPS                                                # SU(2) has no d symbols


def test_algebra_of_a_bad_n_or_index():
    for bad in (1, 0, -3, 2.5):
        with pytest.raises(ValueError):
            su.generators(bad)
    with pytest.raises(IndexError):
        su.get_generator(3, 8)
    with pytest.raises(IndexError):
        su.get_generator(3, -1)
    with pytest.raises(ValueError, match='dimension'):
        su.exp_algebra(np.zeros(5))                                                          # 5 is not N^2 - 1


# ========================================================================= exponential and logarithm
@pytest.mark.parametrize('N', NS)
def test_exp_algebra_is_in_sun_matches_expm_and_has_the_group_properties(N):
    rng = np.random.default_rng(N)
    D = N * N - 1
    w = rng.standard_normal((20, D))
    U = su.exp_algebra(w)
    assert U.shape == (20, N, N) and unitary_special(U)
    T = su.generators(N)
    for k in range(4):
        assert np.abs(U[k] - expm(1j * np.einsum('a,aij->ij', w[k], T))).max() < 1e-12
    assert np.abs(su.exp_algebra(np.zeros(D)) - np.eye(N)).max() < EPS
    assert np.abs(su.exp_algebra(-w) - dagger(U)).max() < 1e-12                                # exp(-w) = exp(w)^-1
    a = 0.7                                                                                   # one direction: it is a one-parameter group
    e = np.zeros(D)
    e[D // 2] = 1
    assert np.abs(su.exp_algebra(a * e) @ su.exp_algebra(0.4 * e) - su.exp_algebra(1.1 * e)).max() < 1e-12
    assert su.exp_algebra(np.zeros((3, 2, D))).shape == (3, 2, N, N)


def test_exp_algebra_for_n_equal_2_is_the_su2_exponential():
    w = np.random.default_rng(1).standard_normal((30, 3))
    assert np.abs(su.exp_algebra(w) - sun.exp_algebra(w)).max() < 1e-12


@pytest.mark.parametrize('N', NS)
def test_log_algebra_inverts_the_exponential(N):
    D = N * N - 1
    rng = np.random.default_rng(10 + N)
    w = 0.15 * rng.standard_normal((30, D))                                                   # near the identity: the principal one
    assert np.abs(su.log_algebra(su.exp_algebra(w)) - w).max() < 1e-10
    U = su.random_su(N, rng, size=40)                                                         # every element is an exponential
    L = su.log_algebra(U)
    assert L.shape == (40, D) and np.abs(su.exp_algebra(L) - U).max() < 1e-10
    assert np.abs(su.log_algebra(np.eye(N))).max() < EPS
    assert su.log_algebra(U[0]).shape == (D,)                                                  # a single matrix


@pytest.mark.parametrize('N', [3, 4])
def test_log_of_the_center_elements_and_of_degenerate_matrices(N):
    for k in range(N):                                                                         # exp(2 pi i k / N) times the identity
        U = np.exp(2j * np.pi * k / N) * np.eye(N)
        assert np.abs(su.exp_algebra(su.log_algebra(U)) - U).max() < 1e-10
    P = np.diag(np.exp(1j * np.array([0.3, 0.3] + [-0.6 / (N - 2)] * (N - 2))))                 # two equal eigenvalues
    assert np.abs(su.exp_algebra(su.log_algebra(P)) - P).max() < 1e-10


def test_log_algebra_for_n_equal_2_is_the_su2_logarithm():
    U = su.random_su(2, np.random.default_rng(3), size=30)
    assert np.abs(su.log_algebra(U) - sun.log_algebra(U)).max() < 1e-9


# ========================================================================= elements: Haar, near the identity, nearest
@pytest.mark.parametrize('N', NS)
def test_random_su_is_in_sun_reproducible_and_haar_distributed(N):
    U = su.random_su(N, np.random.default_rng(1), size=(40, 25))
    assert U.shape == (40, 25, N, N) and unitary_special(U)
    assert su.random_su(N, np.random.default_rng(1)).shape == (N, N)
    assert np.array_equal(su.random_su(N, np.random.default_rng(2), size=5), su.random_su(N, np.random.default_rng(2), size=5))
    assert not np.array_equal(su.random_su(N, np.random.default_rng(2), size=5), su.random_su(N, np.random.default_rng(3), size=5))
    tr = np.trace(su.random_su(N, np.random.default_rng(4), size=8000), axis1=-2, axis2=-1)
    assert abs(tr.mean()) < 0.06 and abs(np.mean(np.abs(tr) ** 2) - 1) < 0.1                  # <tr U> = 0, <|tr U|^2> = 1 for Haar


@pytest.mark.parametrize('N', [2, 3, 4])
def test_near_identity_su_is_at_the_requested_distance_in_a_symmetric_set(N):
    n = su.near_identity_su(N, 0.3, np.random.default_rng(5), size=2000)
    assert unitary_special(n)
    assert np.abs(np.linalg.norm(su.log_algebra(n), axis=-1) - 2 * np.arcsin(0.3)).max() < 1e-9
    w = su.log_algebra(n)
    assert np.abs(w.mean(axis=0)).max() < 0.05 * 2 * np.arcsin(0.3) * 2                         # no preferred direction
    assert np.abs(su.near_identity_su(N, 0.0, np.random.default_rng(6), size=4) - np.eye(N)).max() < EPS
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            su.near_identity_su(N, bad)
    assert su.near_identity_su(N, 0.2, np.random.default_rng(8), size=(3, 4)).shape == (3, 4, N, N)


def test_near_identity_su_for_n_equal_2_is_the_quaternion_set_of_su2():
    n = su.near_identity_su(2, 0.4, np.random.default_rng(9), size=500)
    q = sun.to_quaternion(n)
    assert np.abs(q[..., 0] - np.sqrt(1 - 0.16)).max() < 1e-12 and np.abs(np.linalg.norm(q[..., 1:], axis=-1) - 0.4).max() < 1e-12


@pytest.mark.parametrize('N', NS)
def test_project_su_fixes_the_group_and_removes_rounding_errors(N):
    rng = np.random.default_rng(20 + N)
    U = su.random_su(N, rng, size=30)
    assert np.abs(su.project_su(U) - U).max() < 1e-12                                          # an element is its own nearest
    noisy = U * (1 + 1e-4) + 1e-4 * (rng.standard_normal(U.shape) + 1j * rng.standard_normal(U.shape))
    P = su.project_su(noisy)
    assert unitary_special(P) and np.abs(P - U).max() < 2e-3
    assert np.abs(su.project_su(P) - P).max() < 1e-13                                          # idempotent
    long_product = U[0] @ U[1]
    for _ in range(200):
        long_product = long_product @ U[2]                                                     # rounding builds up
    assert unitary_special(su.project_su(long_product), 1e-12)
    with pytest.raises(ValueError, match='singular'):
        su.project_su(np.zeros((N, N)))


def test_project_su_for_n_equal_2_agrees_with_the_su2_projection_near_the_group():
    U = su.random_su(2, np.random.default_rng(1), size=50)
    noisy = U + 1e-5 * np.random.default_rng(2).standard_normal(U.shape)
    assert np.abs(su.project_su(noisy) - sun.project_su2(noisy)).max() < 1e-9


# ========================================================================= the group object
def test_the_group_object():
    g3, g4 = su.SU(3), su.SU(4)
    assert su.SU(3) is g3 and su.SU3 is g3 and g3 is not g4 and repr(g3) == 'SU(3)'
    assert (g3.N, g3.dim, g3.shape, g4.dim) == (3, 8, (3, 3), 15)
    assert g3.subgroups() == [(0, 1), (0, 2), (1, 2)] and len(g4.subgroups()) == 6 and su.SU(2).subgroups() == [(0, 1)]
    from pylattice.core.groups.lie import LieGroup
    assert isinstance(g3, LieGroup)
    with pytest.raises(ValueError):
        su.SU(1)
    assert np.array_equal(g3.generators(), su.generators(3)) and np.array_equal(g3.structure_constants(), su.structure_constants(3))
    rng = np.random.default_rng(1)
    x = g3.random(rng, size=10)
    assert g3.contains(x).all() and g3.contains(g3.multiply(x, g3.inverse(x))).all()
    assert np.abs(g3.multiply(x, g3.inverse(x)) - g3.identity(10)).max() < 1e-12
    assert not g3.contains(np.full((3, 3), 0.4)).any() and not g3.contains(2 * x[0])


@pytest.mark.parametrize('N', [2, 3, 4])
def test_the_lie_group_operations_of_sun(N):
    g = su.SU(N)
    w = np.random.default_rng(10).standard_normal((40, g.dim)) * 0.3
    assert g.contains(g.exp(w)).all() and np.abs(g.exp(-w) - g.inverse(g.exp(w))).max() < 1e-12
    assert np.abs(g.exp(g.log(g.exp(w))) - g.exp(w)).max() < 1e-10
    x = g.random(np.random.default_rng(11), size=40)
    assert np.abs(g.exp(g.log(x)) - x).max() < 1e-10
    n = g.near_identity(0.1, np.random.default_rng(12), size=100)
    assert g.contains(n).all() and np.abs(n - g.identity(100)).max() < 0.25


def test_su2_of_this_module_is_the_su2_group_of_su2_py():
    g = su.SU(2)
    assert np.array_equal(g.generators(), sun.SU2.generators())
    w = np.random.default_rng(1).standard_normal((20, 3))
    assert np.abs(g.exp(w) - sun.SU2.exp(w)).max() < 1e-12
    x = sun.SU2.random(np.random.default_rng(2), size=20)
    assert np.abs(g.log(x) - sun.SU2.log(x)).max() < 1e-9 and np.abs(g.project(x) - sun.SU2.project(x)).max() < 1e-12


# ========================================================================= tensors
def test_the_tensor_level_functions():
    from pylattice.core.tensors import ComplexMatrix, VectorReal, VectorComplex
    for N in (2, 3):
        T = su.generators(N)
        for k in range(N * N - 1):
            g = su.get_generator(N, k)
            assert isinstance(g, ComplexMatrix) and np.array_equal(g.value, T[k])
        g = su.get_generator(N, 0)
        g.value[0, 0] = 5                                                                        # a new one each time
        assert su.get_generator(N, 0).value[0, 0] != 5
        w = np.random.default_rng(N).standard_normal(N * N - 1) * 0.2
        M = ComplexMatrix(value=1j * np.einsum('a,aij->ij', w, T))                                # M = i w.T: in the algebra
        c = su.ProjToAlgebra(M)
        assert isinstance(c, VectorComplex) and np.abs(c.value - w).max() < 1e-12
        U = su.ExpProjToGroup(VectorReal(value=w))
        assert isinstance(U, ComplexMatrix) and unitary_special(U.value)
        back = su.LogToAlgebra(U)
        assert isinstance(back, VectorReal) and np.abs(back.value - w).max() < 1e-10
        assert np.abs(su.unitarize(ComplexMatrix(value=U.value * 1.001)).value - U.value).max() < 1e-2
        assert su.random_lie_algebra_vector(N, np.random.default_rng(1)).value.shape == (N * N - 1,)
    assert np.abs(su.ExpProjToGroup(VectorReal(value=np.array([0.1, 0.2, 0.3]))).value
                  - sun.ExpProjToGroup(VectorReal(value=np.array([0.1, 0.2, 0.3]))).value).max() < 1e-12
    with pytest.raises(ValueError):
        su.ExpProjToGroup(VectorComplex(value=np.array([0.1j, 0.2, 0.3])))


# ========================================================================= the link update of Cabibbo-Marinari
def test_cabibbo_marinari_link_goes_through_all_the_subgroups_and_stays_in_sun():
    for N in (2, 3, 4):
        rng = np.random.default_rng(N)
        U, V = su.random_su(N, rng, size=60), 2.0 * su.random_su(N, rng, size=60)
        assert unitary_special(su.cabibbo_marinari_link(U, V, 1.0, np.random.default_rng(1)))
        assert np.array_equal(su.cabibbo_marinari_link(U, V, 1.0, np.random.default_rng(1), subgroups=[]), U)   # none: unchanged
        one = su.cabibbo_marinari_link(U, V, 1.0, np.random.default_rng(1), subgroups=[(0, 1)])
        assert np.array_equal(one, su.cabibbo_marinari_step(U, V, 1.0, 0, 1, np.random.default_rng(1)))
    U, V = su.random_su(2, np.random.default_rng(5), size=30), 2.0 * su.random_su(2, np.random.default_rng(6), size=30)
    assert np.array_equal(su.cabibbo_marinari_link(U, V, 1.0, np.random.default_rng(2)),
                          su.cabibbo_marinari_step(U, V, 1.0, 0, 1, np.random.default_rng(2)))    # N = 2: the SU(2) heat bath
    rng = np.random.default_rng(7)                                                                 # at large coupling the link aligns with V
    V = 3.0 * su.random_su(3, rng, size=200)
    new = su.cabibbo_marinari_link(su.random_su(3, rng, size=200), V, 40.0, rng)
    assert (np.trace(new @ V, axis1=-2, axis2=-1).real / (3 * 3.0)).mean() > 0.9


# ========================================================================= overrelaxation of the subgroups
@pytest.mark.parametrize('N', [2, 3, 4])
def test_overrelaxation_keeps_the_weight_stays_in_sun_and_is_its_own_inverse(N):
    rng = np.random.default_rng(N)
    U, V = su.random_su(N, rng, size=80), 1.5 * su.random_su(N, rng, size=80) + 0.5 * su.random_su(N, rng, size=80)
    for i, j in su.SU(N).subgroups():
        new = su.cabibbo_marinari_overrelax(U, V, i, j)
        assert unitary_special(new)
        assert np.abs(np.trace(new @ V, axis1=-2, axis2=-1).real - np.trace(U @ V, axis1=-2, axis2=-1).real).max() < 1e-12
        assert np.abs(su.cabibbo_marinari_overrelax(new, V, i, j) - U).max() < 1e-12                  # applied twice: back
        assert np.abs(new - U).max() > 0.1                                                           # it moves the links
        rest = [a for a in range(N) if a not in (i, j)]                                              # only rows i, j change
        assert np.abs(new[..., rest, :] - (U[..., rest, :])).max() < 1e-12 if rest else True
    every = su.cabibbo_marinari_overrelax_link(U, V)
    assert unitary_special(every)
    assert np.abs(np.trace(every @ V, axis1=-2, axis2=-1).real - np.trace(U @ V, axis1=-2, axis2=-1).real).max() < 1e-11
    assert np.array_equal(su.cabibbo_marinari_overrelax_link(U, V, subgroups=[]), U)
    one = su.cabibbo_marinari_overrelax_link(U, V, subgroups=[(0, 1)])
    assert np.array_equal(one, su.cabibbo_marinari_overrelax(U, V, 0, 1))


def test_overrelaxation_for_n_equal_2_is_the_reflection_of_the_su2_theory():
    rng = np.random.default_rng(3)
    U, S = su.random_su(2, rng, size=50), 2.0 * su.random_su(2, rng, size=50) + su.random_su(2, rng, size=50)
    _, W = sun.su2_decompose(S)
    want = dagger(W) @ dagger(U) @ dagger(W)                                                         # U -> W^+ U^+ W^+, S = k W
    assert np.abs(su.cabibbo_marinari_overrelax(U, S, 0, 1) - want).max() < 1e-12
    assert np.abs(su.cabibbo_marinari_overrelax(U, np.zeros_like(S), 0, 1) - U).max() < 1e-12     # k = 0: W = 1, X = 1


def test_overrelaxation_of_a_subgroup_checks_its_indices():
    U = su.random_su(3, np.random.default_rng(1), size=2)
    with pytest.raises(ValueError):
        su.cabibbo_marinari_overrelax(U, U, 1, 1)
    with pytest.raises(ValueError):
        su.cabibbo_marinari_overrelax(U, U, 0, 3)
