"""Tests of SO(N) (pylattice.core.groups.so): the generators and the algebra, the exponential and logarithm, Haar-random
elements, the projection, the group object, the von Mises sampler and the SO(2) subgroups with the Cabibbo-Marinari steps.
No MPI, tiny arrays; the few statistical checks are vectorized and use large tolerances (5 standard errors).
Run with:  pytest tests/core/test_so.py
"""
import math

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.special import iv

from pylattice.core.groups import so

NS = [2, 3, 4, 5]


def orthogonal_special(U, tol=1e-12):
    U = np.asarray(U)
    n = U.shape[-1]
    return (np.abs(U @ np.swapaxes(U, -1, -2) - np.eye(n)).max() < tol) and (np.abs(np.linalg.det(U) - 1).max() < tol)


# ------------------------------------------------------------------------------------------------ algebra
@pytest.mark.parametrize('N', NS)
def test_generators_are_hermitian_imaginary_and_orthonormal(N):
    T = so.generators(N)
    d = N * (N - 1) // 2
    assert T.shape == (d, N, N)
    assert np.abs(T - np.conj(np.swapaxes(T, -1, -2))).max() < 1e-15 and np.abs(T.real).max() < 1e-15      # hermitian, imaginary
    assert np.abs(np.einsum('aij,bji->ab', T, T) - np.eye(d)).max() < 1e-14                              # tr(T_a T_b) = delta
    T[0] += 1                                                                                             # a copy
    assert np.abs(so.generators(N)[0] - T[0]).max() > 0.5
    assert [so.planes(N)[0], so.planes(N)[-1]] == [(0, 1), (N - 2, N - 1)] and len(so.planes(N)) == d


@pytest.mark.parametrize('N', NS)
def test_structure_constants_are_real_antisymmetric_and_obey_jacobi(N):
    f, T = so.structure_constants(N), so.generators(N)
    assert np.abs(f + np.swapaxes(f, 0, 1)).max() < 1e-14 and np.abs(f + np.swapaxes(f, 1, 2)).max() < 1e-14
    comm = np.einsum('aij,bjk->abik', T, T) - np.einsum('bij,ajk->abik', T, T)
    assert np.abs(comm - 1j * np.einsum('abc,cik->abik', f, T)).max() < 1e-14                            # [T_a, T_b] = i f_abc T_c
    jac = np.einsum('abe,ecd->abcd', f, f) + np.einsum('bce,ead->abcd', f, f) + np.einsum('cae,ebd->abcd', f, f)
    assert np.abs(jac).max() < 1e-14
    if N == 3:
        assert np.abs(np.abs(f[0, 1, 2]) - 1 / math.sqrt(2)) < 1e-14                                    # so(3): f = eps / sqrt(2)


def test_wrong_sizes_and_dimensions_are_refused():
    for bad in (1, 0, 2.5):
        with pytest.raises(ValueError):
            so.generators(bad)
    with pytest.raises(ValueError):
        so.exp_algebra(np.zeros(2))                                                                      # 2 is not N (N - 1) / 2
    assert so.exp_algebra(np.zeros(1)).shape == (2, 2) and so.exp_algebra(np.zeros((4, 6))).shape == (4, 4, 4)


@pytest.mark.parametrize('N', NS)
def test_exp_is_special_orthogonal_and_the_matrix_exponential(N):
    rng = np.random.default_rng(1)
    d = N * (N - 1) // 2
    w = rng.standard_normal((20, d))
    U = so.exp_algebra(w)
    assert U.dtype == np.float64 and orthogonal_special(U)
    A = np.einsum('...a,aij->...ij', w, (1j * so.generators(N)).real)                                    # i w.T = -sum_a w_a A_a / sqrt(2), real
    assert np.abs(U - np.array([expm(a) for a in A])).max() < 1e-12
    assert np.abs(so.exp_algebra(-w) - np.swapaxes(U, -1, -2)).max() < 1e-13                             # exp(-w) = exp(w)^-1
    assert np.abs(so.exp_algebra(np.zeros(d)) - np.eye(N)).max() < 1e-15


def test_so2_is_the_rotation_by_w_over_root_two_and_so3_the_rotation_with_that_angle():
    w = np.array([0.7])
    th = 0.7 / math.sqrt(2)
    assert np.abs(so.exp_algebra(w) - np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])).max() < 1e-14
    v = np.random.default_rng(2).standard_normal((30, 3))
    R = so.exp_algebra(v)
    assert np.abs(np.trace(R, axis1=-2, axis2=-1) - (1 + 2 * np.cos(np.linalg.norm(v, axis=-1) / math.sqrt(2)))).max() < 1e-12


@pytest.mark.parametrize('N', NS)
def test_log_inverts_exp(N):
    rng = np.random.default_rng(3)
    d = N * (N - 1) // 2
    w = 0.3 * rng.standard_normal((30, d))
    assert np.abs(so.log_algebra(so.exp_algebra(w)) - w).max() < 1e-10
    U = so.random_so(N, rng, size=30)
    assert np.abs(so.exp_algebra(so.log_algebra(U)) - U).max() < 1e-10                                   # every element is an exponential
    assert np.abs(so.log_algebra(np.eye(N))).max() < 1e-12


@pytest.mark.parametrize('N', NS)
def test_near_identity_has_the_distance_eps_and_is_closed_under_inversion(N):
    rng = np.random.default_rng(4)
    X = so.near_identity_so(N, 0.2, rng, size=500)
    assert orthogonal_special(X) and X.shape == (500, N, N)
    assert np.abs(np.linalg.norm(so.log_algebra(X), axis=-1) - 2 * math.asin(0.2)).max() < 1e-9          # |w| = 2 arcsin(eps)
    w = so.log_algebra(X)
    assert (np.abs(w.mean(axis=0)) < 5 * w.std(axis=0) / math.sqrt(500)).all()                            # directions are symmetric
    assert np.abs(so.near_identity_so(N, 0, rng) - np.eye(N)).max() < 1e-15
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            so.near_identity_so(N, bad)
    a = so.near_identity_so(N, 0.3, np.random.default_rng(5), size=4)
    assert np.array_equal(a, so.near_identity_so(N, 0.3, np.random.default_rng(5), size=4))


# ---------------------------------------------------------------------------------------- group elements
@pytest.mark.parametrize('N', NS)
def test_random_so_is_special_orthogonal_and_haar(N):
    rng = np.random.default_rng(6)
    U = so.random_so(N, rng, size=20000)
    assert orthogonal_special(U, 1e-12) and U.dtype == np.float64
    tr = np.trace(U, axis1=-2, axis2=-1)
    n = len(U)
    assert abs(tr.mean()) < 5 * tr.std() / math.sqrt(n)                                                  # no trivial component
    expected = 2.0 if N == 2 else 1.0                                                                    # int chi^2 = number of irreps in V x V
    se = (tr ** 2).std() / math.sqrt(n)
    assert abs((tr ** 2).mean() - expected) < 5 * se
    assert abs((U[:, 0, 0] ** 2).mean() - 1 / N) < 5 * (U[:, 0, 0] ** 2).std() / math.sqrt(n)             # <U_ij^2> = 1/N
    assert np.array_equal(so.random_so(N, np.random.default_rng(7), size=3), so.random_so(N, np.random.default_rng(7), size=3))
    assert so.random_so(N, rng).shape == (N, N) and so.random_so(N, rng, size=(2, 3)).shape == (2, 3, N, N)


@pytest.mark.parametrize('N', NS)
def test_project_is_the_nearest_special_orthogonal_matrix(N):
    rng = np.random.default_rng(8)
    U = so.random_so(N, rng, size=30)
    assert np.abs(so.project_so(U) - U).max() < 1e-12                                                    # an element is unchanged
    P = so.project_so(U + 1e-4 * rng.standard_normal(U.shape))
    assert orthogonal_special(P) and np.abs(P - U).max() < 1e-3
    R = U.copy()
    R[:, :, 0] *= -1                                                                                      # det -1: in O(N), not in SO(N)
    assert np.abs(np.linalg.det(R) + 1).max() < 1e-12 and orthogonal_special(so.project_so(R))
    with pytest.raises(ValueError):
        so.project_so(np.zeros((N, N)))


@pytest.mark.parametrize('N', NS)
def test_the_group_object(N):
    g = so.SO(N)
    assert g is so.SO(N) and repr(g) == 'SO({})'.format(N) and g.dim == N * (N - 1) // 2 and g.shape == (N, N)
    assert so.SO3 is so.SO(3)
    assert g.subgroups() == so.planes(N)
    U = g.random(np.random.default_rng(9), size=5)
    assert g.contains(U).all() and not g.contains(2 * U).any() and not g.contains(U + 0.1).any()
    R = U.copy()
    R[:, :, 0] *= -1
    assert not g.contains(R).any()                                                                         # det = -1 is not SO(N)
    assert np.abs(g.multiply(U, g.inverse(U)) - g.identity(5)).max() < 1e-13 and g.identity().dtype == np.float64
    assert np.abs(g.exp(g.log(U)) - U).max() < 1e-10 and np.abs(g.structure_constants() - so.structure_constants(N)).max() == 0
    assert g.contains(g.near_identity(0.2, np.random.default_rng(10), size=4)).all()


def test_so3_is_the_adjoint_of_su2():
    # the rotation of an SU(2) element in the adjoint representation, R_ab = tr(sigma_a U sigma_b U^+) / 2, is an SO(3) element
    # with the products of SU(2); the algebra of SO(3) built here has the same structure (f = eps / sqrt(2) up to the scale)
    from pylattice.core.groups import su2
    sig = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])

    def ad(U):
        return np.einsum('aij,...jk,bkl,...li->...ab', sig, U, sig, np.conj(np.swapaxes(U, -1, -2))).real / 2

    U = su2.to_matrix(su2.random_quaternion(np.random.default_rng(11), size=20))
    V = su2.to_matrix(su2.random_quaternion(np.random.default_rng(12), size=20))
    assert orthogonal_special(ad(U)) and np.abs(ad(U @ V) - ad(U) @ ad(V)).max() < 1e-12
    assert np.abs(np.trace(ad(U), axis1=-2, axis2=-1) - (np.trace(U, axis1=-2, axis2=-1).real ** 2 - 1)).max() < 1e-12    # chi_adj = chi^2 - 1
    assert so.SO(3).contains(ad(U)).all()


# ---------------------------------------------------------------------------------------- Cabibbo-Marinari
@pytest.mark.parametrize('N', NS)
def test_embed_so2_is_the_rotation_in_the_plane_and_the_identity_elsewhere(N):
    th = np.array([0.3, -1.1, 2.0])
    for i, j in so.planes(N):
        R = so.embed_so2(th, i, j, N)
        assert R.shape == (3, N, N) and orthogonal_special(R)
        assert np.abs(R[:, [i, j]][:, :, [i, j]] - np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]).transpose(2, 0, 1)).max() < 1e-15
        rest = [k for k in range(N) if k not in (i, j)]
        assert np.abs(R[:, rest][:, :, rest] - np.eye(N - 2)).max(initial=0) < 1e-15 and np.abs(R[:, rest][:, :, [i, j]]).max(initial=0) == 0
    assert so.embed_so2(0.5, 0, 1, N).shape == (N, N)
    for bad in ((0, 0), (0, N), (-1, 0)):
        with pytest.raises(ValueError):
            so.embed_so2(0.1, bad[0], bad[1], N)


@pytest.mark.parametrize('N', [2, 3, 4])
def test_the_trace_in_a_plane_is_a_cosine_with_the_block_amplitude_and_phase(N):
    rng = np.random.default_rng(13)
    M = rng.standard_normal((6, N, N))
    th = rng.uniform(-math.pi, math.pi, 6)
    for i, j in so.planes(N):
        a, th0 = so.so2_block(M, i, j)
        assert (a >= 0).all() and (np.abs(th0) <= math.pi).all()
        t = lambda x: np.trace(so.embed_so2(x, i, j, N) @ M, axis1=-2, axis2=-1)
        # tr(R(theta) M) - tr(M) is a (cos(theta - theta0) - cos(theta0)): fixes both the amplitude and the phase
        assert np.abs(t(th) - t(np.zeros(6)) - a * (np.cos(th - th0) - np.cos(th0))).max() < 1e-12
    M[:, 0, 1] += 3.0
    M[:, 1, 0] += 3.0                                                                                      # the reflection part: no effect
    a0, _ = so.so2_block(rng.standard_normal((2, N, N)), 0, 1)
    assert so.so2_block(M, 0, 1)[0].shape == (6,) and a0.shape == (2,)


@pytest.mark.parametrize('kappa', [0.0, 0.3, 0.999, 1.0, 2.0, 10.0, 200.0])
def test_von_mises_has_the_exact_moments(kappa):
    n = 400000
    th = so.von_mises(np.full(n, kappa), np.random.default_rng(14))
    assert th.shape == (n,) and (np.abs(th) <= math.pi).all()
    for m in (1, 2):
        exact = iv(m, kappa) / iv(0, kappa)                                                                # <cos(m theta)>
        c = np.cos(m * th)
        assert abs(c.mean() - exact) < 5 * c.std() / math.sqrt(n) + 1e-12, (kappa, m)
    assert abs(np.sin(th).mean()) < 5 * np.sin(th).std() / math.sqrt(n) + 1e-12                             # symmetric around 0


def test_von_mises_inputs_and_shapes():
    rng = np.random.default_rng(15)
    assert so.von_mises(np.array(2.0), rng).shape == () and so.von_mises(np.zeros((2, 3)), rng).shape == (2, 3)
    assert so.von_mises(np.zeros(0), rng).shape == (0,)
    k = np.array([0.5, 5.0, 50.0])
    assert np.array_equal(so.von_mises(k, np.random.default_rng(16)), so.von_mises(k, np.random.default_rng(16)))
    for bad in (-1.0, np.nan, np.inf):
        with pytest.raises(ValueError):
            so.von_mises(np.array([bad]))
    t = so.von_mises(np.array([0.5, 5.0, 500.0]).repeat(20000), rng).reshape(3, -1)                          # each kappa has its own spread
    assert t[0].std() > t[1].std() > t[2].std()


@pytest.mark.parametrize('N', [2, 3, 4])
def test_heat_bath_step_uses_the_coupling_and_changes_only_the_plane(N, monkeypatch):
    rng = np.random.default_rng(17)
    U, V = so.random_so(N, rng, size=8), rng.standard_normal((8, N, N))
    seen = []
    real = so.von_mises
    monkeypatch.setattr(so, 'von_mises', lambda kappa, rng=None: (seen.append(np.array(kappa)), real(kappa, rng))[1])
    for i, j in so.planes(N):
        seen.clear()
        c = 0.7
        U2 = so.cabibbo_marinari_step(U, V, c, i, j, np.random.default_rng(18))
        a, _ = so.so2_block(U @ V, i, j)
        assert np.abs(seen[0] - c * a).max() < 1e-13                                                         # weight exp(c a cos(theta - theta0))
        assert orthogonal_special(U2)
        rest = [k for k in range(N) if k not in (i, j)]
        assert np.abs(U2[:, rest] - U[:, rest]).max(initial=0) == 0                                                   # other rows untouched


@pytest.mark.parametrize('N', [2, 3, 4])
def test_heat_bath_at_strong_coupling_aligns_the_link_and_never_loses_the_plane_maximum(N):
    rng = np.random.default_rng(19)
    U, V = so.random_so(N, rng, size=10), 2 * so.random_so(N, rng, size=10)
    for i, j in so.planes(N):
        U2 = so.cabibbo_marinari_step(U, V, 1e8, i, j, rng)
        a, th0 = so.so2_block(U2 @ V, i, j)
        assert np.abs(th0).max() < 1e-3                                                                       # the maximum of tr(U V) in the plane
        before = np.trace(U @ V, axis1=-2, axis2=-1)
        assert (np.trace(U2 @ V, axis1=-2, axis2=-1) >= before - 1e-7).all()
    far = so.cabibbo_marinari_link(U, V, 1e8, rng)
    assert orthogonal_special(far) and (np.trace(far @ V, axis1=-2, axis2=-1) >= np.trace(U @ V, axis1=-2, axis2=-1) - 1e-7).all()
    assert so.cabibbo_marinari_link(U, V, 0.0, rng, subgroups=[]).shape == U.shape and np.array_equal(
        so.cabibbo_marinari_link(U, V, 0.3, rng, subgroups=[]), U)                                          # no subgroup, no change


@pytest.mark.parametrize('N', [2, 3, 4])
def test_overrelaxation_keeps_the_weight_and_is_an_involution(N):
    rng = np.random.default_rng(20)
    U, V = so.random_so(N, rng, size=10), rng.standard_normal((10, N, N))
    for i, j in so.planes(N):
        U2 = so.cabibbo_marinari_overrelax(U, V, i, j)
        assert orthogonal_special(U2)
        assert np.abs(np.trace(U2 @ V, axis1=-2, axis2=-1) - np.trace(U @ V, axis1=-2, axis2=-1)).max() < 1e-12     # action unchanged
        assert np.abs(so.cabibbo_marinari_overrelax(U2, V, i, j) - U).max() < 1e-12                                   # its own inverse
        a, th0 = so.so2_block(U @ V, i, j)
        a2, th2 = so.so2_block(U2 @ V, i, j)
        assert np.abs(a2 - a).max() < 1e-12 and np.abs(np.angle(np.exp(1j * (th2 + th0)))).max() < 1e-9           # theta0 -> -theta0
        assert np.abs(U2 - U).max() > 1e-3                                                                    # it does move the link
    L = so.cabibbo_marinari_overrelax_link(U, V)
    assert orthogonal_special(L) and np.abs(np.trace(L @ V, axis1=-2, axis2=-1) - np.trace(U @ V, axis1=-2, axis2=-1)).max() < 1e-11
    assert np.array_equal(so.cabibbo_marinari_overrelax_link(U, V, subgroups=[]), U)


def test_so2_overrelaxation_is_the_reflection_of_the_angle():
    # N = 2: U = R(phi), tr(U V) = a cos(phi + ...); the step maps the angle of U to 2 theta0 + phi
    rng = np.random.default_rng(21)
    U, V = so.random_so(2, rng, size=5), rng.standard_normal((5, 2, 2))
    a, th0 = so.so2_block(U @ V, 0, 1)
    assert np.abs(so.cabibbo_marinari_overrelax(U, V, 0, 1) - so.embed_so2(2 * th0, 0, 1, 2) @ U).max() < 1e-14
