"""What every group offers a theory, whatever kind it is (finite or Lie): the elements, the product, the
inverse, a uniform random element, the nearest element to an arbitrary value, and a membership test. A
theory written against this interface (a spin model with values in a group, a gauge theory with links in a
group) works for any group that implements it.

Elements are numpy arrays of any batch shape: the group fixes the shape of ONE element (`shape`, () for a
number, (2, 2) for an SU(2) matrix), and an array of elements has shape size + shape. Functions that draw
take a numpy Generator (pylattice.core.rng); None gives a fresh unseeded one.

Implementations: pylattice.core.groups.discrete (Z2, Z_q), pylattice.core.groups.su2 (SU2).
"""
from abc import ABC, abstractmethod


class Group(ABC):
    name = ''
    shape = ()                                    # shape of one element

    @abstractmethod
    def identity(self, size=None):
        """the identity, repeated to the shape size (None: one element)"""

    @abstractmethod
    def random(self, rng=None, size=None):
        """independent uniformly distributed elements (the Haar measure for a Lie group)"""

    @abstractmethod
    def multiply(self, a, b):
        """the product of the elements a and b (broadcast)"""

    @abstractmethod
    def inverse(self, a):
        """the inverse of the elements a"""

    @abstractmethod
    def project(self, x):
        """the element of the group nearest to the values x: removes the rounding errors of a long
        computation, or maps arbitrary values to the group"""

    @abstractmethod
    def contains(self, x, tol=1e-9):
        """bool array: is each value an element of the group, to the tolerance tol?"""

    def __repr__(self):
        return self.name or type(self).__name__
