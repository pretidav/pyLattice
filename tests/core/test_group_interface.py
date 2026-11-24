"""The interface every group has (pylattice.core.groups.base and lie), tested on all of them at once: Z2, Z_q,
SU(2), SU(N) and SO(N). A theory written against this interface must work for each of them. No MPI.
Run with:  pytest tests/core/test_group_interface.py
"""
import numpy as np
import pytest
from abc import ABC

from pylattice.core.groups.base import Group
from pylattice.core.groups.discrete import Z, Z2, DiscreteGroup
from pylattice.core.groups.lie import LieGroup
from pylattice.core.groups.so import SO
from pylattice.core.groups.su import SU
from pylattice.core.groups.su2 import SU2

GROUPS = [Z2, Z(3), Z(7), SU2, SU(3), SU(4), SO(2), SO(3), SO(4)]
ids = [repr(g) for g in GROUPS]


def close(g, a, b, tol=1e-12):
    """a and b are the same element(s), to the tolerance, whatever the shape of an element"""
    ax = tuple(range(-len(g.shape), 0)) if g.shape else None
    return np.abs(np.asarray(a) - np.asarray(b)).max(axis=ax) <= tol if g.shape else np.abs(np.asarray(a) - np.asarray(b)) <= tol


def test_the_classes_form_the_expected_hierarchy():
    assert issubclass(DiscreteGroup, Group) and issubclass(LieGroup, Group) and not issubclass(LieGroup, DiscreteGroup)
    assert isinstance(Z2, Group) and isinstance(SU2, LieGroup) and SU2.dim == 3 and SU2.shape == (2, 2) and Z2.shape == ()
    with pytest.raises(TypeError):
        Group()                                                                   # abstract
    with pytest.raises(TypeError):
        LieGroup()
    assert [repr(g) for g in GROUPS] == ['Z2', 'Z3', 'Z7', 'SU(2)', 'SU(3)', 'SU(4)', 'SO(2)', 'SO(3)', 'SO(4)']


@pytest.mark.parametrize('g', GROUPS, ids=ids)
def test_shapes(g):
    assert g.identity().shape == g.shape and g.identity(5).shape == (5,) + g.shape and g.identity((2, 3)).shape == (2, 3) + g.shape
    rng = np.random.default_rng(1)
    assert g.random(rng).shape == g.shape and g.random(rng, 4).shape == (4,) + g.shape and g.random(rng, (2, 3)).shape == (2, 3) + g.shape


@pytest.mark.parametrize('g', GROUPS, ids=ids)
def test_identity_is_neutral_and_in_the_group(g):
    x = g.random(np.random.default_rng(2), size=20)
    e = g.identity(20)
    assert g.contains(e).all() and close(g, g.multiply(e, x), x).all() and close(g, g.multiply(x, e), x).all()
    assert close(g, g.inverse(g.identity()), g.identity())


@pytest.mark.parametrize('g', GROUPS, ids=ids)
def test_random_elements_are_in_the_group_and_follow_the_generator(g):
    x = g.random(np.random.default_rng(3), size=200)
    assert g.contains(x).all()
    assert np.array_equal(g.random(np.random.default_rng(4), size=10), g.random(np.random.default_rng(4), size=10))
    assert not np.array_equal(g.random(np.random.default_rng(4), size=10), g.random(np.random.default_rng(5), size=10))


@pytest.mark.parametrize('g', GROUPS, ids=ids)
def test_associativity_inverse_and_closure(g):
    a, b, c = g.random(np.random.default_rng(6), size=(3, 30))
    assert close(g, g.multiply(g.multiply(a, b), c), g.multiply(a, g.multiply(b, c))).all()
    assert close(g, g.multiply(a, g.inverse(a)), g.identity(30)).all() and close(g, g.multiply(g.inverse(a), a), g.identity(30)).all()
    assert g.contains(g.multiply(a, b)).all() and g.contains(g.inverse(a)).all()
    assert close(g, g.inverse(g.inverse(a)), a).all()
    assert close(g, g.inverse(g.multiply(a, b)), g.multiply(g.inverse(b), g.inverse(a))).all()      # (ab)^-1 = b^-1 a^-1


@pytest.mark.parametrize('g', GROUPS, ids=ids)
def test_project_is_idempotent_and_fixes_the_elements(g):
    x = g.random(np.random.default_rng(7), size=30)
    assert close(g, g.project(x), x).all()                                         # an element is its own nearest element
    noisy = x * (1 + 1e-4) + 1e-4 * (np.random.default_rng(8).standard_normal(x.shape))
    p = g.project(noisy)
    assert g.contains(p).all() and close(g, g.project(p), p, 1e-13).all() and close(g, p, x, 1e-3).all()


@pytest.mark.parametrize('g', GROUPS, ids=ids)
def test_contains_rejects_what_is_not_an_element(g):
    assert g.contains(g.random(np.random.default_rng(9), size=5)).all()
    far = np.full((3,) + g.shape, 0.37)
    assert not g.contains(far).any()
    assert g.contains(g.random(np.random.default_rng(9)) * 1.01).shape == ()


def test_lie_groups_have_an_exponential_with_the_algebra_properties():
    g = SU2
    assert g.generators().shape == (g.dim,) + g.shape
    w = np.random.default_rng(10).standard_normal((40, g.dim))
    assert g.contains(g.exp(w)).all()
    assert close(g, g.exp(np.zeros(g.dim)), g.identity()) and close(g, g.exp(-w), g.inverse(g.exp(w))).all()
    assert close(g, g.exp(g.log(g.exp(0.3 * w))), g.exp(0.3 * w), 1e-10).all()
    x = g.random(np.random.default_rng(11), size=40)
    assert close(g, g.exp(g.log(x)), x, 1e-10).all()                                # every element is an exponential
    n = g.near_identity(0.1, np.random.default_rng(12), size=100)
    assert g.contains(n).all() and np.abs(n - g.identity(100)).max() < 0.25
    assert np.abs(np.linalg.norm(n - g.identity(100), axis=(-2, -1)) - np.sqrt(4 * (1 - np.sqrt(1 - 0.01)))).max() < 1e-12
