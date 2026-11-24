"""Tests of the SU(2) group and algebra (pylattice.core.groups.su2): every property of the generators,
the exponential and logarithm, the projection onto the group, the quaternionic representation of Creutz
(products, Haar measure, the sum of staples, the heat bath) and the SU2 object of the common group interface.
No MPI. Run with:  pytest tests/core/test_su2.py
"""
import itertools

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.special import iv, ive
from scipy.stats import kstest

from pylattice.core.groups import su2 as sun
from pylattice.core.tensors import ComplexMatrix, VectorComplex, VectorReal

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


T = [sun.get_generator(a).value for a in range(3)]


# ------------------------------------------------------------------------------- generators
def test_generators_are_the_pauli_matrices_over_two():
    sigma = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
    for a in range(3):
        assert np.array_equal(T[a], sigma[a] / 2)
    assert np.array_equal(sun.generators(), np.array(T))
    assert sun.generators().shape == (3, 2, 2)


def test_generators_are_double_precision_and_not_shared():
    assert all(t.dtype == np.complex128 for t in T)
    g = sun.get_generator(0)
    g.value[0, 0] = 99
    assert sun.get_generator(0).value[0, 0] == 0                      # a new copy each time
    sun.generators()[0, 0, 0] = 99
    assert sun.generators()[0, 0, 0] == 0


def test_the_generators_are_hermitian_and_traceless():
    for t in T:
        assert np.array_equal(t, dagger(t)) and abs(np.trace(t)) == 0


def test_the_generators_are_normalised_tr_TaTb_is_delta_over_two():
    for a, b in itertools.product(range(3), repeat=2):
        assert abs(np.trace(T[a] @ T[b]) - 0.5 * (a == b)) < EPS


def test_the_generators_close_the_algebra_with_the_structure_constants():
    for a, b in itertools.product(range(3), repeat=2):                # all of them, also a == b
        commutator = T[a] @ T[b] - T[b] @ T[a]
        expected = sum(1j * sun.structure_constants(a, b, c) * T[c] for c in range(3))
        assert np.abs(commutator - expected).max() < EPS


def test_su2_anticommutator_and_casimir():
    for a, b in itertools.product(range(3), repeat=2):
        assert np.abs(T[a] @ T[b] + T[b] @ T[a] - 0.5 * (a == b) * I2).max() < EPS
    assert np.abs(sum(t @ t for t in T) - 0.75 * I2).max() < EPS      # j (j + 1) = 3/4


def test_the_jacobi_identity():
    f = lambda a, b, c: sun.structure_constants(a, b, c)
    for a, b, c, e in itertools.product(range(3), repeat=4):
        assert sum(f(a, b, d) * f(d, c, e) + f(b, c, d) * f(d, a, e) + f(c, a, d) * f(d, b, e) for d in range(3)) == 0


def test_structure_constants_are_the_levi_civita_symbol():
    for i, j, k in itertools.product(range(3), repeat=3):
        expected = np.linalg.det(np.eye(3)[[i, j, k]]) if len({i, j, k}) == 3 else 0
        assert sun.structure_constants(i, j, k) == round(expected)
    assert [sun.structure_constants(*p) for p in ((0, 1, 2), (1, 2, 0), (2, 0, 1))] == [1, 1, 1]
    assert [sun.structure_constants(*p) for p in ((1, 0, 2), (2, 1, 0), (0, 2, 1))] == [-1, -1, -1]
    assert all(type(sun.structure_constants(0, 1, 2)) is int for _ in range(1))


def test_generators_and_constants_refuse_an_index_that_does_not_exist():
    for bad in (3, -1, 1.5, True):
        with pytest.raises(IndexError):
            sun.get_generator(bad)
    with pytest.raises(IndexError):
        sun.structure_constants(0, 1, 3)


# ------------------------------------------------------------------- random algebra vectors
def test_random_lie_algebra_vector_is_a_real_vector_of_three_in_the_cube():
    v = sun.random_lie_algebra_vector(np.random.default_rng(1))
    assert isinstance(v, VectorReal) and v.value.shape == (3,) and v.value.dtype == float
    many = np.array([sun.random_lie_algebra_vector(np.random.default_rng(s)).value for s in range(2000)])
    assert many.min() >= -0.5 and many.max() < 0.5 and np.abs(many.mean(axis=0)).max() < 0.03
    assert np.abs(many.var(axis=0) - 1 / 12).max() < 0.01


def test_the_random_vectors_follow_the_generator_they_are_given():
    a = sun.random_lie_algebra_vector(np.random.default_rng(5)).value
    b = sun.random_lie_algebra_vector(np.random.default_rng(5)).value
    c = sun.random_lie_algebra_vector(np.random.default_rng(6)).value
    assert np.array_equal(a, b) and not np.array_equal(a, c)


# --------------------------------------------------------------------------- exponential
def test_exp_of_a_real_vector_is_in_su2_to_machine_precision():
    rng = np.random.default_rng(2)
    for scale in (1e-9, 1e-3, 0.5, 3.0, 20.0, 100.0):
        for _ in range(20):
            U = sun.ExpProjToGroup(VectorReal(value=scale * rng.standard_normal(3))).value
            assert_su2(U, 1e-13)


def test_exp_agrees_with_the_matrix_exponential():
    rng = np.random.default_rng(3)
    for _ in range(50):
        w = 4 * rng.standard_normal(3)
        U = sun.ExpProjToGroup(VectorReal(value=w)).value
        assert np.abs(U - expm(1j * sum(w[a] * T[a] for a in range(3)))).max() < 1e-12


def test_exp_known_values():
    exp = lambda *w: sun.ExpProjToGroup(VectorReal(value=np.array(w, dtype=float))).value
    assert np.array_equal(exp(0, 0, 0), I2)
    assert np.abs(exp(0, 0, 2 * np.pi) + I2).max() < EPS                   # a rotation by 2 pi is -1: spinors
    assert np.abs(exp(0, 0, 4 * np.pi) - I2).max() < EPS                   # and by 4 pi is 1
    assert np.abs(exp(np.pi, 0, 0) - 1j * np.array([[0, 1], [1, 0]])).max() < EPS      # i sigma_1
    assert np.abs(exp(0, np.pi, 0) - 1j * np.array([[0, -1j], [1j, 0]])).max() < EPS
    assert np.abs(exp(0, 0, np.pi) - 1j * np.diag([1, -1])).max() < EPS
    assert np.abs(exp(0, 0, 0.3) - np.diag([np.exp(0.15j), np.exp(-0.15j)])).max() < EPS


def test_exp_is_a_one_parameter_subgroup_and_inverts_by_the_adjoint():
    rng = np.random.default_rng(4)
    w = rng.standard_normal(3)
    e = lambda x: sun.ExpProjToGroup(VectorReal(value=x)).value
    assert np.abs(e(0.7 * w) @ e(1.9 * w) - e(2.6 * w)).max() < 1e-12     # same axis: the angles add
    assert np.abs(e(w) @ e(-w) - I2).max() < 1e-12 and np.abs(e(-w) - dagger(e(w))).max() < 1e-12
    assert np.abs(np.linalg.det(e(w)) - 1) < EPS


def test_exp_is_the_quaternion_cos_and_sin_of_half_the_angle():
    w = np.array([0.4, -1.1, 0.7])
    theta = np.linalg.norm(w)
    q = sun.to_quaternion(sun.ExpProjToGroup(VectorReal(value=w)).value)
    assert np.abs(q - np.concatenate([[np.cos(theta / 2)], np.sin(theta / 2) * w / theta])).max() < EPS


def test_exp_is_smooth_at_zero():
    for scale in (0.0, 1e-300, 1e-20, 1e-8):
        U = sun.ExpProjToGroup(VectorReal(value=scale * np.array([1.0, 2.0, 2.0]))).value
        assert np.isfinite(U).all() and np.abs(U - I2).max() <= 4 * scale + EPS


def test_exp_of_a_complex_vector_leaves_su2_but_keeps_the_determinant():
    w = np.array([0.3, 0.1j, -0.2])
    U = sun.ExpProjToGroup(VectorComplex(value=w)).value
    assert np.abs(U - expm(1j * sum(w[a] * T[a] for a in range(3)))).max() < EPS
    assert abs(np.linalg.det(U) - 1) < EPS and np.abs(U @ dagger(U) - I2).max() > 1e-3     # in SL(2,C), not SU(2)


def test_exp_refuses_a_vector_that_is_not_of_three_components():
    with pytest.raises(ValueError):
        sun.ExpProjToGroup(VectorReal(value=np.zeros(4)))


# ------------------------------------------------------------------------ projection and log
def test_proj_to_algebra_is_exact_on_the_algebra():
    rng = np.random.default_rng(5)
    for _ in range(20):
        w = rng.standard_normal(3)
        X = 1j * sum(w[a] * T[a] for a in range(3))
        c = sun.ProjToAlgebra(ComplexMatrix(value=X))
        assert isinstance(c, VectorComplex) and np.abs(c.value - w).max() < EPS
    assert np.abs(sun.ProjToAlgebra(ComplexMatrix(value=I2.astype(complex))).value).max() < EPS    # nothing along the identity


def test_proj_to_algebra_is_linear_and_first_order_on_the_group():
    rng = np.random.default_rng(6)
    A, B = (rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2)) for _ in range(2))
    P = lambda M: sun.ProjToAlgebra(ComplexMatrix(value=M)).value
    assert np.abs(P(2 * A - 3j * B) - (2 * P(A) - 3j * P(B))).max() < EPS
    w = 1e-4 * np.array([1.0, -2.0, 0.5])
    U = sun.ExpProjToGroup(VectorReal(value=w)).value
    assert np.abs(P(U) - w).max() < 1e-12                                   # w to first order
    w = np.array([0.0, 0.0, 1.0])
    assert abs(P(sun.ExpProjToGroup(VectorReal(value=w)).value)[2] - 2 * np.sin(0.5)) < EPS


def test_log_is_the_inverse_of_exp():
    rng = np.random.default_rng(7)
    for _ in range(100):
        w = rng.standard_normal(3)
        w *= rng.uniform(0, 2 * np.pi) / np.linalg.norm(w)                  # |w| up to 2 pi
        back = sun.LogToAlgebra(sun.ExpProjToGroup(VectorReal(value=w)))
        assert isinstance(back, VectorReal) and np.abs(back.value - w).max() < 1e-10


def test_exp_of_log_is_the_element_for_every_element():
    for U in haar(200, 8):
        w = sun.LogToAlgebra(ComplexMatrix(value=U))
        assert np.linalg.norm(w.value) <= 2 * np.pi + EPS
        assert np.abs(sun.ExpProjToGroup(w).value - U).max() < 1e-10


def test_log_at_the_identity_and_at_minus_the_identity():
    assert np.array_equal(sun.LogToAlgebra(ComplexMatrix(value=I2.astype(complex))).value, np.zeros(3))
    w = sun.LogToAlgebra(ComplexMatrix(value=-I2.astype(complex))).value
    assert np.linalg.norm(w) == pytest.approx(2 * np.pi)                    # the axis is arbitrary
    assert np.abs(sun.ExpProjToGroup(VectorReal(value=w)).value + I2).max() < EPS
    U = sun.ExpProjToGroup(VectorReal(value=np.array([1e-13, 0, 0]))).value  # next to the identity
    assert np.abs(sun.LogToAlgebra(ComplexMatrix(value=U)).value - [1e-13, 0, 0]).max() < 1e-15


# ----------------------------------------------------------------------------- unitarize
def test_unitarize_leaves_an_element_of_su2_alone():
    for U in haar(100, 9):
        assert np.abs(sun.unitarize(ComplexMatrix(value=U)).value - U).max() < 1e-13


def test_unitarize_of_any_matrix_is_in_su2_not_only_unitary():
    rng = np.random.default_rng(10)
    for _ in range(300):
        M = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        try:
            V = sun.unitarize(ComplexMatrix(value=M)).value
        except ValueError:                                                  # no SU(2) part: not for a random matrix
            raise AssertionError('a random matrix has a quaternion part')
        assert_su2(V)                                                       # the determinant is 1, not just a phase


def test_unitarize_removes_a_small_perturbation_and_does_not_move_the_element():
    rng = np.random.default_rng(11)
    for U in haar(50, 12):
        noise = 1e-3 * (rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2)))
        V = sun.unitarize(ComplexMatrix(value=U + noise)).value
        assert_su2(V) and np.linalg.norm(V - U) < 3 * np.linalg.norm(noise)  # a nearby element, not a different one


def test_unitarize_keeps_the_sign_minus_one_stays_minus_one():
    assert np.abs(sun.unitarize(ComplexMatrix(value=-1.0 * I2.astype(complex))).value + I2).max() < EPS
    U = haar(1, 13)[0]
    assert np.abs(sun.unitarize(ComplexMatrix(value=-3.0 * U)).value + U).max() < 1e-13
    assert np.abs(sun.unitarize(ComplexMatrix(value=0.01 * U)).value - U).max() < 1e-13


def test_unitarize_is_the_nearest_element_of_su2():
    rng = np.random.default_rng(14)
    for _ in range(20):
        M = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        V = sun.unitarize(ComplexMatrix(value=M)).value
        distance = np.linalg.norm(M - V)
        for W in haar(2000, int(rng.integers(1 << 30))):                    # no element is nearer
            assert np.linalg.norm(M - W) >= distance - 1e-12
        for _ in range(100):                                                # nor one next to it
            W = sun.ExpProjToGroup(VectorReal(value=0.05 * rng.standard_normal(3))).value @ V
            assert np.linalg.norm(M - W) >= distance - 1e-12


def test_unitarize_is_idempotent_and_refuses_what_has_no_su2_part():
    rng = np.random.default_rng(15)
    M = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    V = sun.unitarize(ComplexMatrix(value=M))
    assert np.abs(sun.unitarize(V).value - V.value).max() < 1e-14
    for bad in (np.zeros((2, 2), complex), np.diag([1, -1]).astype(complex), np.diag([1j, 1j]).astype(complex)):
        with pytest.raises(ValueError):
            sun.unitarize(ComplexMatrix(value=bad))


def test_project_su2_works_on_a_batch_of_any_shape():
    rng = np.random.default_rng(16)
    M = rng.standard_normal((3, 4, 2, 2)) + 1j * rng.standard_normal((3, 4, 2, 2))
    P = sun.project_su2(M)
    assert P.shape == M.shape and P.dtype == complex
    assert_su2(P)
    assert np.abs(P[1, 2] - sun.project_su2(M[1, 2])).max() < EPS


# ------------------------------------------------------------------------------ quaternions
def test_quaternion_and_matrix_are_the_same_element():
    assert np.array_equal(sun.to_matrix([1, 0, 0, 0]), I2)
    assert np.array_equal(sun.to_matrix([-1, 0, 0, 0]), -I2)
    for a in range(3):                                                      # i sigma_a
        e = np.zeros(4)
        e[a + 1] = 1
        assert np.array_equal(sun.to_matrix(e), 2j * T[a])
    q = np.array([0.1, 0.2, -0.3, 0.4])
    U = sun.to_matrix(q)
    assert np.abs(U - (q[0] * I2 + 2j * sum(q[a + 1] * T[a] for a in range(3)))).max() < EPS    # a0 + i a.sigma
    assert np.abs(sun.to_quaternion(U) - q).max() < EPS


def test_quaternion_roundtrips_both_ways_on_batches():
    rng = np.random.default_rng(17)
    q = rng.standard_normal((5, 3, 4))
    assert np.abs(sun.to_quaternion(sun.to_matrix(q)) - q).max() < EPS
    assert sun.to_matrix(q).shape == (5, 3, 2, 2) and sun.to_quaternion(sun.to_matrix(q)).shape == (5, 3, 4)
    U = haar(7, 18)
    assert np.abs(sun.to_matrix(sun.to_quaternion(U)) - U).max() < EPS


def test_the_properties_of_the_quaternion_of_an_element():
    for U in haar(50, 19):
        q = sun.to_quaternion(U)
        assert abs(np.linalg.norm(q) - 1) < EPS                             # unit: det U = |q|^2
        assert abs(np.trace(U) / 2 - q[0]) < EPS                            # a0 = tr U / 2
        assert np.abs(sun.to_matrix(sun.quat_conj(q)) - dagger(U)).max() < EPS    # conjugate = adjoint
    q = np.array([0.3, -1.0, 2.0, 0.5])
    assert abs(np.linalg.det(sun.to_matrix(q)) - q @ q) < EPS               # det = |q|^2 for any quaternion


def test_quaternion_product_is_the_matrix_product():
    rng = np.random.default_rng(20)
    p, q = rng.standard_normal((2, 20, 4))
    assert np.abs(sun.to_matrix(sun.quat_mul(p, q)) - sun.to_matrix(p) @ sun.to_matrix(q)).max() < 1e-12
    assert np.abs(sun.quat_mul(p, q) - sun.to_quaternion(sun.to_matrix(p) @ sun.to_matrix(q))).max() < 1e-12


def test_quaternion_algebra_identity_inverse_associativity_noncommutativity():
    rng = np.random.default_rng(21)
    p, q, r = sun.random_quaternion(rng, size=(3, 10))
    one = np.array([1.0, 0, 0, 0])
    assert np.abs(sun.quat_mul(one, q) - q).max() < EPS and np.abs(sun.quat_mul(q, one) - q).max() < EPS
    assert np.abs(sun.quat_mul(q, sun.quat_conj(q)) - one).max() < EPS      # inverse = conjugate
    assert np.abs(sun.quat_mul(sun.quat_mul(p, q), r) - sun.quat_mul(p, sun.quat_mul(q, r))).max() < EPS
    assert np.abs(sun.quat_mul(p, q) - sun.quat_mul(q, p)).max() > 1e-3     # SU(2) is not abelian
    assert np.abs(np.linalg.norm(sun.quat_mul(p, q), axis=-1) - 1).max() < EPS     # the product of unit ones is unit
    i, j = np.eye(4)[1], np.eye(4)[2]
    assert np.array_equal(sun.quat_mul(i, j), -np.eye(4)[3])                # (i sigma_1)(i sigma_2) = -i sigma_3
    assert np.array_equal(sun.quat_mul(j, i), np.eye(4)[3])                 # and the other way round: the sign
    assert np.array_equal(sun.quat_mul(i, i), -one)                         # i^2 = -1


def test_quat_conj_does_not_change_its_argument():
    q = np.array([1.0, 2.0, 3.0, 4.0])
    c = sun.quat_conj(q)
    assert np.array_equal(q, [1, 2, 3, 4]) and np.array_equal(c, [1, -2, -3, -4])


def test_quat_normalize_and_its_refusal():
    q = np.array([[3.0, 0, 4.0, 0], [0, 0, 0, 2.0]])
    assert np.allclose(sun.quat_normalize(q), [[0.6, 0, 0.8, 0], [0, 0, 0, 1]])
    with pytest.raises(ValueError):
        sun.quat_normalize(np.zeros(4))


def test_quaternion_functions_check_the_shapes():
    with pytest.raises(ValueError):
        sun.to_quaternion(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        sun.to_matrix(np.zeros(3))


# -------------------------------------------------------------------------- Haar measure
def test_random_quaternions_are_unit_and_have_the_shape_asked():
    rng = np.random.default_rng(22)
    assert sun.random_quaternion(rng).shape == (4,)
    assert sun.random_quaternion(rng, size=5).shape == (5, 4)
    q = sun.random_quaternion(rng, size=(3, 7))
    assert q.shape == (3, 7, 4) and np.abs(np.linalg.norm(q, axis=-1) - 1).max() < EPS


def test_random_quaternions_are_uniform_on_the_three_sphere():
    q = sun.random_quaternion(np.random.default_rng(23), size=400000)
    assert np.abs(q.mean(axis=0)).max() < 0.005
    second = q.T @ q / len(q)
    assert np.abs(second - np.eye(4) / 4).max() < 0.005                     # <a_i a_j> = delta / 4
    assert abs(np.mean(q[:, 0] ** 4) - 1 / 8) < 0.003                       # the density of a0 is (2/pi) sqrt(1 - a0^2)
    assert kstest(q[:, 0], lambda x: (np.arcsin(x) + x * np.sqrt(1 - x * x)) / np.pi + 0.5).pvalue > 1e-3


def test_haar_measure_is_invariant_under_multiplication():
    q = sun.random_quaternion(np.random.default_rng(24), size=400000)
    g = sun.random_quaternion(np.random.default_rng(25))
    for moved in (sun.quat_mul(g, q), sun.quat_mul(q, g)):
        assert np.abs(moved.mean(axis=0)).max() < 0.005
        assert np.abs(moved.T @ moved / len(q) - np.eye(4) / 4).max() < 0.005
        assert abs(np.mean(moved[:, 0] ** 4) - 1 / 8) < 0.003


def test_random_quaternions_follow_the_generator():
    a = sun.random_quaternion(np.random.default_rng(1), size=4)
    assert np.array_equal(a, sun.random_quaternion(np.random.default_rng(1), size=4))
    assert not np.array_equal(a, sun.random_quaternion(np.random.default_rng(2), size=4))


def test_near_identity_quaternions():
    rng = np.random.default_rng(26)
    for eps in (0.0, 1e-3, 0.3, 1.0):
        q = sun.near_identity_quaternion(eps, rng, size=2000)
        assert np.abs(np.linalg.norm(q, axis=-1) - 1).max() < EPS
        assert np.abs(np.linalg.norm(q[:, 1:], axis=-1) - eps).max() < EPS and (q[:, 0] >= 0).all()
    q = sun.near_identity_quaternion(0.4, rng, size=300000)
    assert np.abs(q[:, 1:].mean(axis=0)).max() < 0.003                      # the direction is uniform: no preferred axis
    assert np.abs(np.cov(q[:, 1:].T) - 0.16 / 3 * np.eye(3)).max() < 0.002
    # symmetric under inversion (what a Metropolis proposal needs): the distribution of X and of X^-1 agree
    assert np.abs(sun.quat_conj(q)[:, 1:].mean(axis=0)).max() < 0.003
    assert sun.near_identity_quaternion(0.1, rng).shape == (4,)
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            sun.near_identity_quaternion(bad)


# -------------------------------------------------------------------- the sum of staples
def test_a_sum_of_su2_elements_is_a_multiple_of_one():
    rng = np.random.default_rng(27)
    for n in (1, 2, 6, 10):
        V = haar((100, n), int(rng.integers(1 << 30))).sum(axis=1)
        k, W = sun.su2_decompose(V)
        assert k.shape == (100,) and W.shape == (100, 2, 2)
        assert (k >= 0).all() and np.abs(k ** 2 - np.linalg.det(V).real).max() < 1e-12     # k = sqrt(det V)
        assert np.abs(np.linalg.det(V).imag).max() < 1e-12
        assert_su2(W)
        assert np.abs(k[:, None, None] * W - V).max() < 1e-12


def test_decompose_of_a_single_element_and_of_zero():
    U = haar(1, 28)[0]
    k, W = sun.su2_decompose(U)
    assert abs(k - 1) < EPS and np.abs(W - U).max() < EPS
    k, W = sun.su2_decompose(np.zeros((2, 2), complex))
    assert k == 0 and np.array_equal(W, I2)                                 # no direction: the identity
    k, W = sun.su2_decompose(np.stack([np.zeros((2, 2), complex), 3 * U]))
    assert np.allclose(k, [0, 3]) and np.abs(W[1] - U).max() < EPS


# ------------------------------------------------------------------------------ heat bath
def mean_a0(alpha):
    """<a0> = I2(alpha) / I1(alpha), through the scaled Bessel functions so that it holds at large alpha"""
    return ive(2, alpha) / ive(1, alpha) if alpha else 0.0


def cdf_of(alpha):
    # the cumulative distribution of sqrt(1 - a0^2) exp(alpha a0), on a grid fine enough at the edge a0 = 1
    u = np.linspace(0, np.pi, 40001)                                         # a0 = cos(u): dx = sin(u) du, smooth in u
    a0 = np.cos(u)
    w = np.sin(u) ** 2 * np.exp(alpha * (a0 - 1))
    cum = np.concatenate([[0], np.cumsum((w[1:] + w[:-1]) / 2 * np.diff(u))])
    return lambda x: np.interp(np.arccos(np.clip(x, -1, 1)), u, 1 - cum / cum[-1])


SAMPLERS = [sun.creutz_a0, sun.kennedy_pendleton_a0, sun.sample_a0]
ALPHAS = [0.0, 0.05, 0.5, 2.0, 5.9, 6.1, 8.0, 40.0, 300.0, 5000.0]


@pytest.mark.parametrize('sampler', SAMPLERS, ids=lambda f: f.__name__)
@pytest.mark.parametrize('alpha', ALPHAS)
def test_a0_samplers_follow_the_distribution(sampler, alpha):
    a0 = sampler(alpha, np.random.default_rng(29), size=100000)
    assert a0.shape == (100000,) and a0.min() >= -1 and a0.max() <= 1
    assert abs(a0.mean() - mean_a0(alpha)) < 0.005 + 0.1 / (1 + alpha)      # <a0> = I2 / I1 (the spread falls with alpha)
    assert kstest(a0, cdf_of(alpha)).pvalue > 1e-3


@pytest.mark.parametrize('alpha', [1.0, 10.0, 200.0])
def test_creutz_and_kennedy_pendleton_give_the_same_distribution(alpha):
    from scipy.stats import ks_2samp
    a = sun.creutz_a0(alpha, np.random.default_rng(1), size=50000)
    b = sun.kennedy_pendleton_a0(alpha, np.random.default_rng(2), size=50000)
    assert ks_2samp(a, b).pvalue > 1e-3
    assert abs(np.var(a) - np.var(b)) < 0.1 * np.var(a) + 1e-6


class CountingRng:
    """a numpy Generator that counts the random numbers it gives"""

    def __init__(self, seed):
        self.rng, self.count = np.random.default_rng(seed), 0

    def random(self, *args, **kwargs):
        out = self.rng.random(*args, **kwargs)
        self.count += out.size
        return out

    def standard_normal(self, *args, **kwargs):
        out = self.rng.standard_normal(*args, **kwargs)
        self.count += out.size
        return out


def test_kennedy_pendleton_is_much_cheaper_than_creutz_at_large_alpha_and_auto_picks_the_cheaper():
    n = 20000
    cost = {}
    for name, sampler in (('creutz', sun.creutz_a0), ('kp', sun.kennedy_pendleton_a0), ('auto', sun.sample_a0)):
        for alpha in (0.3, 2.0, 100.0, 5000.0):
            rng = CountingRng(3)
            sampler(alpha, rng, size=n)
            cost[name, alpha] = rng.count / n                               # random numbers per sample
    assert cost['kp', 5000.0] < 4.2 and cost['kp', 100.0] < 4.5            # 4 per trial, about one trial
    assert cost['creutz', 5000.0] > 100 and cost['creutz', 100.0] > 12     # the 1/sqrt(alpha) acceptance
    assert cost['kp', 0.3] > 3 * cost['creutz', 0.3]                       # and the other way round at small alpha
    for alpha in (0.3, 2.0):
        assert cost['auto', alpha] == cost['creutz', alpha]
    for alpha in (100.0, 5000.0):
        assert cost['auto', alpha] == cost['kp', alpha]


def test_the_samplers_pick_by_alpha_in_one_call_with_a_mixed_array():
    alpha = np.array([0.0, 0.5, sun.KP_THRESHOLD, sun.KP_THRESHOLD + 1e-9, 50.0, 1e4])
    a0 = sun.sample_a0(alpha[:, None], np.random.default_rng(4), size=(6, 40000))
    assert a0.shape == (6, 40000)
    for row, a in zip(a0, alpha):
        assert abs(row.mean() - mean_a0(a)) < 0.006 + 0.1 / (1 + a)
    assert np.array_equal(sun.sample_a0(alpha, np.random.default_rng(5)), sun.sample_a0(alpha, np.random.default_rng(5)))


def test_a0_samplers_refuse_a_bad_alpha_and_accept_a_scalar_and_a_shape():
    for sampler in SAMPLERS:
        for bad in (-1.0, np.nan, np.inf):
            with pytest.raises(ValueError):
                sampler(bad)
        assert sampler(3.0, np.random.default_rng(0)).shape == () and sampler(3.0, np.random.default_rng(0), size=(2, 3)).shape == (2, 3)


def test_creutz_a0_takes_an_array_of_alphas_and_a_shape():
    rng = np.random.default_rng(30)
    alpha = np.array([[0.0, 1.0], [5.0, 30.0]])
    a0 = sun.creutz_a0(alpha, rng)
    assert a0.shape == (2, 2) and (np.abs(a0) <= 1).all()
    many = sun.creutz_a0(alpha[..., None], rng, size=(2, 2, 50000))
    assert many.shape == (2, 2, 50000)
    for idx, a in np.ndenumerate(alpha):
        assert abs(many[idx].mean() - (iv(2, a) / iv(1, a) if a else 0)) < 0.01
    assert sun.creutz_a0(3.0, rng).shape == () and sun.creutz_a0(3.0, rng, size=4).shape == (4,)
    assert np.array_equal(sun.creutz_a0(2.0, np.random.default_rng(1), size=5), sun.creutz_a0(2.0, np.random.default_rng(1), size=5))


@pytest.mark.parametrize('beta_k', [0.0, 0.7, 4.0, 25.0, 400.0])
def test_heatbath_link_has_the_boltzmann_distribution(beta_k):
    # one link, V = k W: tr(U W)/2 must be distributed as a0 above, and <U> = <a0> W^dagger
    rng = np.random.default_rng(31)
    W = haar(1, 32)[0]
    k = 1.3
    n = 100000
    U = sun.heatbath_link(np.broadcast_to(k * W, (n, 2, 2)), beta_k / k, rng)
    assert U.shape == (n, 2, 2)
    assert_su2(U)
    a0 = np.einsum('nij,ji->n', U, W).real / 2                              # tr(U W) / 2
    mean = mean_a0(beta_k)
    assert abs(a0.mean() - mean) < 0.006
    assert np.abs(U.mean(axis=0) - mean * dagger(W)).max() < 0.006         # the other three components average to zero
    assert kstest(a0, cdf_of(beta_k)).pvalue > 1e-3


def test_heatbath_link_of_a_sum_of_staples_and_independent_links():
    rng = np.random.default_rng(33)
    V = haar((200, 5), 34).sum(axis=1)                                      # 200 links with 5 staples each
    U = sun.heatbath_link(V, 2.0, rng)
    assert U.shape == (200, 2, 2)
    assert_su2(U)
    k, W = sun.su2_decompose(V)
    a0 = np.einsum('nij,nji->n', U, W).real / 2
    assert abs(a0.mean() - np.mean([iv(2, 2 * x) / iv(1, 2 * x) for x in k])) < 0.06
    assert len({tuple(np.round(u.ravel(), 12)) for u in U}) == 200           # not copies of each other
    assert np.array_equal(sun.heatbath_link(V, 2.0, np.random.default_rng(1)), sun.heatbath_link(V, 2.0, np.random.default_rng(1)))


def test_heatbath_link_at_zero_coupling_or_zero_staples_is_haar():
    rng = np.random.default_rng(35)
    for V, beta in ((np.broadcast_to(2 * I2, (100000, 2, 2)), 0.0), (np.zeros((100000, 2, 2)), 5.0)):
        q = sun.to_quaternion(sun.heatbath_link(V, beta, rng))
        assert np.abs(q.mean(axis=0)).max() < 0.006 and np.abs(q.T @ q / len(q) - np.eye(4) / 4).max() < 0.006


def test_heatbath_link_makes_the_plaquette_larger_as_beta_grows():
    V = haar((50, 4), 36).sum(axis=1)
    rng = np.random.default_rng(37)
    mean = [np.mean(np.einsum('nij,nji->n', sun.heatbath_link(np.repeat(V, 200, axis=0), b, rng),
                              np.repeat(V, 200, axis=0)).real) for b in (0.0, 1.0, 5.0)]
    assert mean[0] < 0.05 and mean[0] < mean[1] < mean[2]                   # tr(U V) grows with the coupling


def test_heatbath_refuses_a_negative_beta():
    with pytest.raises(ValueError):
        sun.heatbath_link(np.eye(2, dtype=complex), -1.0)


# --------------------------------------------------------------- the group behind the common interface
def test_batched_exp_and_log_agree_with_the_tensor_versions():
    rng = np.random.default_rng(60)
    w = rng.standard_normal((6, 5, 3)) * 2
    U = sun.exp_algebra(w)
    assert U.shape == (6, 5, 2, 2)
    assert_su2(U)
    for idx in np.ndindex(6, 5):
        assert np.abs(U[idx] - sun.ExpProjToGroup(VectorReal(value=w[idx])).value).max() < EPS
    back = sun.log_algebra(U)
    assert back.shape == w.shape
    for idx in np.ndindex(6, 5):
        assert np.abs(back[idx] - sun.LogToAlgebra(ComplexMatrix(value=U[idx])).value).max() < 1e-12
    assert np.abs(sun.exp_algebra(back) - U).max() < 1e-10


def test_batched_log_handles_the_special_points_in_one_array():
    U = np.stack([np.eye(2), -np.eye(2), sun.exp_algebra(np.array([1e-13, 0, 0])), sun.exp_algebra(np.array([0, 0, 1.0]))]).astype(complex)
    w = sun.log_algebra(U)
    assert np.abs(w[0]).max() == 0 and np.abs(w[1] - [0, 0, 2 * np.pi]).max() < EPS
    assert np.abs(w[2] - [1e-13, 0, 0]).max() < 1e-15 and np.abs(w[3] - [0, 0, 1.0]).max() < EPS
    assert np.isfinite(w).all()


def test_su2_object_is_the_same_group_as_the_functions():
    from pylattice.core.groups.lie import LieGroup
    assert isinstance(sun.SU2, LieGroup) and sun.SU2.name == 'SU(2)' and sun.SU2.dim == 3
    assert np.array_equal(sun.SU2.generators(), sun.generators())
    q = sun.random_quaternion(np.random.default_rng(61), size=7)
    assert np.array_equal(sun.SU2.random(np.random.default_rng(62), size=7), sun.to_matrix(sun.random_quaternion(np.random.default_rng(62), size=7)))
    A, B = haar(7, 63), haar(7, 64)
    assert np.array_equal(sun.SU2.multiply(A, B), A @ B) and np.array_equal(sun.SU2.inverse(A), dagger(A))
    assert np.array_equal(sun.SU2.project(A + 1e-3), sun.project_su2(A + 1e-3))
    assert not sun.SU2.contains(2 * A).any() and not sun.SU2.contains(np.stack([np.diag([1, -1])] * 3).astype(complex)).any()
    assert sun.SU2.contains(A).all() and sun.SU2.contains(A[0]).shape == ()
    assert np.array_equal(sun.SU2.identity(3), np.broadcast_to(I2, (3, 2, 2))) and sun.SU2.identity().shape == (2, 2)
    assert sun.SU2.identity(3).flags.writeable                                         # a copy, not a view of one matrix
    n = sun.SU2.near_identity(0.2, np.random.default_rng(65), size=5)
    assert np.array_equal(n, sun.to_matrix(sun.near_identity_quaternion(0.2, np.random.default_rng(65), size=5)))
