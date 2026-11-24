"""SU(N) built from its SU(2) subgroups (Cabibbo-Marinari): the 2x2 blocks of an N x N matrix are picked
(su2_block), updated with the SU(2) heat bath of pylattice.core.groups.su2, and put back (embed_su2). Every
SU(N) update is a sequence of these steps over the subgroups (i, j), which together generate SU(N). N = 2 is
SU(2) itself, with the single subgroup (0, 1).

Matrices have shape (..., N, N): a batch of links at once. This module needs no action: a step takes the sum of
the staples V of the links, which the gauge theory computes.
"""
import numpy as np

from pylattice.core.groups.su2 import heatbath_link, su2_decompose


def embed_su2(U, i, j, N):
    """The N x N matrix that is U (2x2, shape (..., 2, 2)) on the rows and columns i and j and the identity
    elsewhere: an element of the SU(2) subgroup (i, j) of SU(N). Returns shape (..., N, N)."""
    U = np.asarray(U)
    _check_block(i, j, N)
    out = np.zeros(U.shape[:-2] + (N, N), dtype=complex)
    out[..., np.arange(N), np.arange(N)] = 1.0
    out[..., i, i], out[..., i, j] = U[..., 0, 0], U[..., 0, 1]
    out[..., j, i], out[..., j, j] = U[..., 1, 0], U[..., 1, 1]
    return out


def su2_block(M, i, j):
    """The 2x2 block of rows and columns i, j of the N x N matrices M, written as k W with k >= 0 and W in
    SU(2) (the SU(2) part of the block, see to_quaternion; returns (k, W) as su2_decompose does). It is
    what a Cabibbo-Marinari step needs: the update of the subgroup (i, j) of a link U with staple sum V
    uses the block (i, j) of U V."""
    M = np.asarray(M)
    _check_block(i, j, M.shape[-1])
    return su2_decompose(M[..., [i, j], :][..., [i, j]])


def _check_block(i, j, N):
    if not (0 <= i < N and 0 <= j < N and i != j):
        raise ValueError("a block needs two different indices in 0..{}, got {}, {}".format(N - 1, i, j))


def cabibbo_marinari_step(U, V, c, i, j, rng=None):
    """One heat-bath step of the SU(2) subgroup (i, j) of SU(N) for the link weight exp(c Re tr(U V)), with V
    the sum of the staples of the link (so a plaquette is tr(U staple)): U -> X U with X in the subgroup,
    drawn from its exact conditional distribution. Only the block (i, j) of U V matters, since tr(X M) sees
    only that block. U, V: (..., N, N); c >= 0. Returns the new U, still in SU(N).

    A sweep over the links is a loop over the subgroups, e.g. (0, 1), (1, 2), (0, 2) for SU(3)."""
    U = np.asarray(U)
    k, W = su2_block(U @ np.asarray(V), i, j)
    X = heatbath_link(k[..., None, None] * W, 2 * c, rng)       # weight exp((2c/2) Re tr(X kW)) = exp(c Re tr(X M))
    return embed_su2(X, i, j, U.shape[-1]) @ U
