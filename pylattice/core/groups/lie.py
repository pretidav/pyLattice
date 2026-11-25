"""Lie groups: the common interface of the groups that have an algebra (SU(N), SO(N), later U(N)). On top of
pylattice.core.groups.base it gives the algebra: its dimension and generators, the exponential map from the
algebra to the group and its inverse, and a random element near the identity (the proposal of a Metropolis
step). An algebra vector has shape size + (dim,), its group element is exp(i w.T).

Implementations: pylattice.core.groups.su2 (SU2) and pylattice.core.groups.su (SU(N), the instances SU(3), SU(4), ...), pylattice.core.groups.so (SO(N)).
"""
from abc import abstractmethod

from pylattice.core.groups.base import Group


class LieGroup(Group):
    dim = 0                                       # dimension of the algebra

    @abstractmethod
    def generators(self):
        """the generators T_a as an array (dim,) + shape"""

    @abstractmethod
    def exp(self, w):
        """the group elements exp(i w.T) of the real algebra vectors w (..., dim)"""

    @abstractmethod
    def log(self, g):
        """algebra vectors w with exp(w) = g (the principal one, see the implementation)"""

    @abstractmethod
    def near_identity(self, eps, rng=None, size=None):
        """random elements at distance eps from the identity, in a uniformly random direction, a set closed
        under inversion (so a Metropolis step g -> x g with them is symmetric)"""
