"""SU(N) and SO(N) gauge fields on 2 MPI ranks: the plaquette, the action, the staples, the Wilson loops, the Polyakov loop and the gauge
transformation of a lattice split over the ranks equal a serial numpy computation (np.roll) on the same global configuration, whichever
direction the lattice is split in (the links across the rank boundaries come from the neighbour rank). Run with:
    mpiexec -n 2 python tests/theories/gauge/test_gauge_field_mpi.py
"""
import numpy as np
import pytest
from mpi4py import MPI

from pylattice.core.groups import so, su
from pylattice.core.parallel import CartesianComm, pprint
from pylattice.theories.gauge.field import SONField, SUNField

pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.Get_size() != 2, reason='needs exactly 2 MPI ranks (mpiexec -n 2)')

dag = lambda M: np.conj(np.swapaxes(M, -1, -2))


def local_block(full, cc, local):
    """the part of a global array (*global grid, ...) that lives on this rank, flattened over the sites"""
    lo = [c * l for c, l in zip(cc.mpicoord, local)]
    return full[tuple(slice(a, a + l) for a, l in zip(lo, local))].reshape((int(np.prod(local)),) + full.shape[len(local):])


def loops_and_polyakov(A, U, G, N):
    """Wilson loops (also the ones that cross the boundary between the ranks) and the Polyakov loop along the last direction"""
    Nd = len(G)

    def line(mu, R):
        out = moved = U[..., mu, :, :]
        for _ in range(R - 1):
            moved = np.roll(moved, -1, axis=mu)
            out = out @ moved
        return out

    def loop(mu, nu, R, T):
        a, b = line(mu, R), line(nu, T)
        return a @ np.roll(b, -R, axis=mu) @ dag(np.roll(a, -T, axis=nu)) @ dag(b)

    ordered = [(m, n) for m in range(Nd) for n in range(Nd) if m != n]
    for R, T in [(R, T) for R, T in ((1, 2), (2, 2), (2, 3)) if max(R, T) <= min(G)]:      # only the loops that fit
        want = np.mean([np.trace(loop(m, n, R, T), axis1=-2, axis2=-1) / N for m, n in ordered])
        assert abs(A.Wilson(R, T) - want) < 1e-11, (R, T, A.Wilson(R, T), want)
    t = Nd - 1
    want = (np.trace(line(t, G[t]), axis1=-2, axis2=-1) / N).mean()                    # around the whole lattice, across the ranks
    assert abs(A.Polyakov(t) - want) < 1e-11, (A.Polyakov(t), want)
    assert abs(A.Polyakov(t)) > 1e-5


def check(mpigrid, local, N=2, group='su'):
    random, Field = (su.random_su, SUNField) if group == 'su' else (so.random_so, SONField)
    cc = CartesianComm(mpigrid=mpigrid)
    Nd, G = len(local), [m * l for m, l in zip(mpigrid, local)]
    U = random(N, np.random.default_rng(42), size=tuple(G) + (Nd,))                              # the same on every rank
    A = Field(grid=local, cartesiancomm=cc, Nd=Nd, N=N)
    A.value = local_block(U, cc, local).copy()
    shift = lambda X, mu: np.roll(X, -1, axis=mu)                                                      # value at x + mu
    link = lambda mu: U[..., mu, :, :]
    P = lambda mu, nu: link(mu) @ shift(link(nu), mu) @ dag(shift(link(mu), nu)) @ dag(link(nu))
    tr = lambda M: np.trace(M, axis1=-2, axis2=-1)
    ordered = [(m, n) for m in range(Nd) for n in range(Nd) if m != n]
    planes = [(m, n) for m in range(Nd) for n in range(m + 1, Nd)]
    plaq = sum(tr(P(m, n)).sum() for m, n in ordered) / (N * len(ordered) * np.prod(G))
    action = 2.3 * sum((1 - tr(P(m, n)).real / N).sum() for m, n in planes)
    assert abs(A.Plaq() - plaq) < 1e-12, (A.Plaq(), plaq)
    assert abs(A.Action(2.3) - action) < 1e-9, (A.Action(2.3), action)
    for mu in range(Nd):
        S = sum(shift(link(nu), mu) @ dag(shift(link(mu), nu)) @ dag(link(nu))
                + dag(np.roll(shift(link(nu), mu), 1, axis=nu)) @ dag(np.roll(link(mu), 1, axis=nu)) @ np.roll(link(nu), 1, axis=nu)
                for nu in range(Nd) if nu != mu)
        assert np.abs(A.Staple(mu).value - local_block(S, cc, local)).max() < 1e-12
    loops_and_polyakov(A, U, G, N)
    g = random(N, np.random.default_rng(7), size=tuple(G))
    B = A.gauge_transform(local_block(g, cc, local))
    for mu in range(Nd):
        want = g @ link(mu) @ dag(shift(g, mu))
        assert np.abs(B.value[:, mu] - local_block(want, cc, local)).max() < 1e-12
    assert abs(B.Plaq() - A.Plaq()) < 1e-12 and abs(B.Action(2.3) - A.Action(2.3)) < 1e-9
    assert abs(B.Wilson(1, 2) - A.Wilson(1, 2)) < 1e-11 and abs(B.Polyakov(Nd - 1) - A.Polyakov(Nd - 1)) < 1e-11    # invariant
    assert abs(A.Plaq().imag) < 1e-12 and type(B) is Field


def test_split_along_the_first_direction():
    check([2, 1], [3, 4])


def test_split_along_the_second_direction():
    check([1, 2], [3, 4])


def test_3d_lattice_split_along_one_direction():
    check([2, 1, 1], [2, 3, 3])
    check([1, 1, 2], [2, 3, 3])


def test_su3_split_along_each_direction():
    check([2, 1], [3, 4], N=3)
    check([1, 2], [3, 4], N=3)
    check([1, 1, 2], [2, 3, 3], N=3)                                                    # the Polyakov loop runs across the ranks


def test_su4_split_along_one_direction():
    check([2, 1], [3, 4], N=4)


def test_so_n_split_along_each_direction():
    check([2, 1], [3, 4], N=2, group='so')
    check([1, 2], [3, 4], N=3, group='so')
    check([2, 1], [3, 4], N=4, group='so')
    check([1, 1, 2], [2, 3, 3], N=3, group='so')                                        # the Polyakov loop runs across the ranks


if __name__ == '__main__':
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() != 2:                  # the rank grids hold 2 ranks: any other number would hang
        pprint(comm=MPI.COMM_WORLD, msg='[-] skipped: this script needs exactly 2 ranks (mpiexec -n 2)')
        raise SystemExit(0)
    for name, f in list(globals().items()):
        if name.startswith('test_'):
            f()
            pprint(comm=MPI.COMM_WORLD, msg='[+] {}'.format(name))
