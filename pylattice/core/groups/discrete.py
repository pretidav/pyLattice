"""Finite groups: Z2 (the Ising spin) and the cyclic groups Z_q (the clock model; Z3 is the Potts model of
three states). The elements are numbers, so a coupling is a product of values: the roots of unity
exp(2 pi i k / q), k = 0 .. q-1; for Z2 they are the integers +1 and -1 (int8: a spin takes one byte). `index` and
`from_index` go between the numbers and the labels k, where the group is the addition of the labels mod q.

    Z2.elements  -> [ 1, -1]            Z(3).elements -> [1, exp(2 pi i/3), exp(4 pi i/3)]
    Z2.multiply(-1, -1) == 1            Z2.project(0.3) == 1 (nearest element)
    Z2.heatbath(h, beta, rng)           the exact heat bath of a value with neighbour sum h

All the functions work on arrays of any shape and use the numbers, never the labels, unless named index.
"""
import numpy as np

from pylattice.core.groups.base import Group


def _rng(rng):
    return rng if rng is not None else np.random.default_rng()


class DiscreteGroup(Group):
    """A finite group of order `order`, with its elements as numbers (`elements`, in the order of the
    labels), the multiplication table of the labels, and the heat bath."""
    order = 0
    elements = None
    table = None                                  # table[k, l] = label of element k times element l

    @property
    def identity_element(self):
        return self.elements[0]

    @property
    def generator(self):
        """the element of label 1: the non-trivial element of Z2, exp(2 pi i / q) of Z_q"""
        return self.elements[1]

    def identity(self, size=None):
        shape = () if size is None else ((size,) if np.isscalar(size) else tuple(size))
        return np.full(shape, self.elements[0])

    def index(self, x):
        """the label k of the element nearest to each value x"""
        raise NotImplementedError

    def from_index(self, k):
        """the elements of the labels k (an array of integers)"""
        return self.elements[np.asarray(k)]

    def random(self, rng=None, size=None):
        return self.from_index(_rng(rng).integers(0, self.order, size=size))

    def multiply(self, a, b):
        return self.from_index(self.table[self.index(a), self.index(b)])

    def inverse(self, a):
        k = self.index(a)
        return self.from_index(self.inverse_labels[k])

    def project(self, x):
        return self.from_index(self.index(x))

    def contains(self, x, tol=1e-9):
        x = np.asarray(x)
        return np.abs(x - self.project(x)) <= tol

    def power(self, a, n):
        """a to the integer power n"""
        return self.from_index((self.index(a) * int(n)) % self.order)

    def heatbath(self, h, beta, rng=None):
        """New values drawn independently, element g with probability proportional to exp(beta Re(g h)), where
        h (any shape; real, or complex for Z_q) is the sum of what the value is coupled to: the neighbours of
        a spin, the staples of a link. beta may be an array too. The old value does not matter (exact heat
        bath). For Z2 and a real h this is P(+1) = 1 / (1 + exp(-2 beta h))."""
        h = np.asarray(h)
        beta = np.asarray(beta, dtype=float)
        if np.any(beta < 0):
            raise ValueError("beta must be >= 0")
        e = self.elements.reshape((self.order,) + (1,) * h.ndim)
        logits = beta * np.real(e * h)                                     # (order, ...)
        logits -= logits.max(axis=0)                                       # no overflow
        weights = np.exp(logits)
        cumulative = np.cumsum(weights, axis=0)
        u = _rng(rng).random(h.shape) * cumulative[-1]
        k = np.minimum((u > cumulative).sum(axis=0), self.order - 1)       # first label whose cumulative weight exceeds u
        return self.from_index(k)


class CyclicGroup(DiscreteGroup):
    """Z_q: the numbers exp(2 pi i k / q) under multiplication (the real numbers +1, -1 for q = 2)."""

    def __init__(self, q):
        if int(q) != q or q < 2:
            raise ValueError("the order of a cyclic group must be an integer of at least 2, got {}".format(q))
        self.order = self.q = int(q)
        self.name = 'Z{}'.format(self.q)
        k = np.arange(self.q)
        self.elements = np.array([1, -1], dtype=np.int8) if q == 2 else np.exp(2j * np.pi * k / q)
        self.table = (k[:, None] + k[None, :]) % self.q
        self.inverse_labels = (-k) % self.q
        self.elements.setflags(write=False)

    def index(self, x):
        x = np.asarray(x)
        if self.q == 2:
            return (np.real(x) < 0).astype(np.intp)
        return np.rint(np.angle(x) * self.q / (2 * np.pi)).astype(np.intp) % self.q


def Z(q):
    """the cyclic group of order q"""
    return CyclicGroup(q)


Z2 = CyclicGroup(2)
