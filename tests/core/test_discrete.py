"""Tests of the finite groups (pylattice.core.groups.discrete): Z2 and the cyclic groups Z_q, with their
elements as numbers, and the exact heat bath. No MPI. Run with:  pytest tests/core/test_discrete.py
"""
import itertools

import numpy as np
import pytest
from scipy.stats import chisquare

from pylattice.core.groups.discrete import CyclicGroup, DiscreteGroup, Z, Z2

GROUPS = [Z2, Z(3), Z(4), Z(5), Z(12)]
ids = [g.name for g in GROUPS]


def test_z2_is_plus_and_minus_one_as_real_numbers():
    assert isinstance(Z2, DiscreteGroup) and Z2.name == 'Z2' and Z2.order == 2
    assert list(Z2.elements) == [1.0, -1.0] and Z2.elements.dtype == float
    assert Z2.identity_element == 1.0 and Z2.generator == -1.0
    assert Z2.multiply(-1.0, -1.0) == 1.0 and Z2.multiply(-1.0, 1.0) == -1.0
    assert Z2.inverse(-1.0) == -1.0 and np.array_equal(Z2.multiply(np.array([1.0, -1, -1]), np.array([-1.0, -1, 1])), [-1, 1, -1])


def test_cyclic_groups_are_the_roots_of_unity():
    for q in (3, 4, 7):
        g = Z(q)
        assert g.order == q and g.name == 'Z{}'.format(q) and g.elements.dtype == complex
        assert np.abs(g.elements - np.exp(2j * np.pi * np.arange(q) / q)).max() < 1e-15
        assert np.abs(np.abs(g.elements) - 1).max() < 1e-15 and g.elements[0] == 1
        assert g.generator == g.elements[1]
    assert np.abs(Z(4).elements - [1, 1j, -1, -1j]).max() < 1e-15


@pytest.mark.parametrize('g', GROUPS, ids=ids)
def test_the_group_axioms_on_the_multiplication_table(g):
    n = g.order
    assert g.table.shape == (n, n)
    for k in range(n):
        assert sorted(g.table[k]) == list(range(n)) and sorted(g.table[:, k]) == list(range(n))      # a Latin square
        assert g.table[0, k] == k == g.table[k, 0]                                                   # the identity
        assert g.table[k, g.inverse_labels[k]] == 0
    for a, b, c in itertools.product(range(n), repeat=3):
        assert g.table[g.table[a, b], c] == g.table[a, g.table[b, c]]
    assert np.array_equal(g.table, g.table.T)                                                        # abelian


@pytest.mark.parametrize('g', GROUPS, ids=ids)
def test_multiplication_of_the_numbers_is_the_multiplication_of_the_numbers(g):
    for a, b in itertools.product(g.elements, repeat=2):
        assert abs(g.multiply(a, b) - a * b) < 1e-14
        assert g.multiply(a, b) in g.elements                                    # exactly an element, no rounding drift
    assert all(abs(g.multiply(a, g.inverse(a)) - 1) < 1e-14 for a in g.elements)
    assert all(g.multiply(a, g.inverse(a)) == g.elements[0] for a in g.elements)


@pytest.mark.parametrize('g', GROUPS, ids=ids)
def test_index_and_from_index(g):
    k = np.arange(g.order)
    assert np.array_equal(g.index(g.elements), k) and np.array_equal(g.from_index(k), g.elements)
    assert g.index(g.elements[1]).shape == () and g.from_index(np.array([[0, 1], [1, 0]])).shape == (2, 2)
    for tol in (1e-9, 1e-3):                                                     # nearby values belong to the same element
        assert np.array_equal(g.index(g.elements * (1 + tol) + tol / 3), k)


@pytest.mark.parametrize('g', GROUPS, ids=ids)
def test_project_gives_the_nearest_element_and_is_idempotent(g):
    rng = np.random.default_rng(1)
    x = rng.standard_normal(500) + (1j * rng.standard_normal(500) if g.order > 2 else 0)
    p = g.project(x)
    assert g.contains(p).all() and np.array_equal(g.project(p), p)
    if g.order > 2:
        nearest = np.array([g.elements[np.argmin(np.abs(g.elements - v))] for v in x])
        assert np.abs(p - nearest).max() < 1e-12 or (np.abs(p - x) <= np.abs(nearest - x) + 1e-12).all()
    else:
        assert np.array_equal(p, np.where(x < 0, -1.0, 1.0))
    assert Z2.project(0.3) == 1 and Z2.project(-0.3) == -1 and Z2.project(0.0) == 1          # the tie goes to the identity


@pytest.mark.parametrize('g', GROUPS, ids=ids)
def test_contains(g):
    assert g.contains(g.elements).all()
    assert not g.contains(np.array([0.5, 2.0, 0.3 + 0.3j])).any() and not g.contains(g.elements * 1.001).any()
    assert g.contains(g.elements * (1 + 1e-12)).all() and g.contains(g.elements[0]).shape == ()


@pytest.mark.parametrize('g', GROUPS, ids=ids)
def test_identity_and_power(g):
    assert g.identity() == 1 and g.identity(4).shape == (4,) and g.identity((2, 3)).shape == (2, 3)
    assert (g.identity(5) == 1).all()
    for a in g.elements:
        assert g.power(a, 0) == 1 and abs(g.power(a, g.order) - 1) < 1e-14 and abs(g.power(a, -1) - g.inverse(a)) < 1e-14
        assert abs(g.power(a, 3) - a ** 3) < 1e-12


@pytest.mark.parametrize('g', GROUPS, ids=ids)
def test_random_is_uniform_independent_and_reproducible(g):
    n = 60000
    x = g.random(np.random.default_rng(2), size=n)
    assert g.contains(x).all()
    counts = np.bincount(g.index(x), minlength=g.order)
    assert chisquare(counts).pvalue > 1e-3                                       # each element equally often
    pairs = np.bincount(g.index(x[:-1]) * g.order + g.index(x[1:]), minlength=g.order ** 2)
    assert chisquare(pairs).pvalue > 1e-3                                        # consecutive draws are independent
    assert np.array_equal(g.random(np.random.default_rng(3), size=10), g.random(np.random.default_rng(3), size=10))
    assert g.random(np.random.default_rng(3)).shape == () and g.random(np.random.default_rng(3), size=(2, 3)).shape == (2, 3)


def test_z2_random_uses_one_draw_per_element():
    # the same generator stream as rng.integers: a run is reproducible whatever the code around it does
    a = Z2.random(np.random.default_rng(4), size=100)
    b = np.random.default_rng(4).integers(0, 2, size=100)
    assert np.array_equal(Z2.index(a), b)


# ------------------------------------------------------------------------------ heat bath
@pytest.mark.parametrize('beta,h', [(0.0, 3.0), (0.4, 1.0), (0.4, -2.0), (1.5, 4.0), (50.0, -1.0)])
def test_z2_heatbath_probability(beta, h):
    n = 200000
    x = Z2.heatbath(np.full(n, h), beta, np.random.default_rng(5))
    p_plus = 1 / (1 + np.exp(-2 * beta * h))
    assert x.shape == (n,) and Z2.contains(x).all()
    sigma = np.sqrt(p_plus * (1 - p_plus) / n)
    assert abs((x == 1).mean() - p_plus) < 5 * sigma + 1e-12


@pytest.mark.parametrize('g', GROUPS[1:], ids=ids[1:])
def test_cyclic_heatbath_follows_the_boltzmann_weights(g):
    h = 0.7 * np.exp(0.9j)                                                       # the sum of the staples: complex
    beta, n = 1.3, 300000
    x = g.heatbath(np.full(n, h), beta, np.random.default_rng(6))
    p = np.exp(beta * np.real(g.elements * h))
    p /= p.sum()
    counts = np.bincount(g.index(x), minlength=g.order)
    assert chisquare(counts, p * n).pvalue > 1e-3
    assert g.heatbath(h, beta, np.random.default_rng(7)).shape == ()


def test_heatbath_is_independent_of_the_old_value_and_takes_arrays_of_h_and_beta():
    rng = np.random.default_rng(8)
    h = np.array([[-3.0, 0.0], [0.5, 2.0]])
    beta = np.array([[1.0, 1.0], [0.1, 3.0]])
    n = 100000
    x = Z2.heatbath(np.broadcast_to(h, (n, 2, 2)), beta, rng)                    # beta broadcast against h
    assert x.shape == (n, 2, 2)
    p = 1 / (1 + np.exp(-2 * beta * h))
    assert np.abs((x == 1).mean(axis=0) - p).max() < 0.01
    assert np.array_equal(Z2.heatbath(h, beta, np.random.default_rng(9)), Z2.heatbath(h, beta, np.random.default_rng(9)))


def test_heatbath_at_zero_coupling_is_uniform_and_at_infinity_picks_the_best():
    assert abs((Z2.heatbath(np.full(100000, 5.0), 0.0, np.random.default_rng(10)) == 1).mean() - 0.5) < 0.01
    assert (Z2.heatbath(np.full(1000, 1.0), 1e4, np.random.default_rng(11)) == 1).all()       # no overflow at huge beta
    assert (Z2.heatbath(np.full(1000, -1.0), 1e4, np.random.default_rng(11)) == -1).all()
    g = Z(5)
    best = g.heatbath(np.full(1000, g.elements[3]), 1e4, np.random.default_rng(12))             # Re(g h) is largest for g = conj(h)
    assert (g.index(best) == 2).all()
    with pytest.raises(ValueError):
        Z2.heatbath(np.ones(3), -1.0)


def test_constructing_cyclic_groups():
    for bad in (1, 0, -3, 2.5):
        with pytest.raises(ValueError):
            CyclicGroup(bad)
    assert Z(2).elements.tolist() == Z2.elements.tolist() and Z(6).order == 6
    assert repr(Z2) == 'Z2' and not Z2.elements.flags.writeable
