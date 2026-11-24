from pylattice.core.grid import LatticeReal, LatticeComplex, LatticeRealMatrix, LatticeComplexMatrix, LatticeVectorReal, LatticeVectorComplex, LatticeVectorRealMatrix, LatticeVectorComplexMatrix
from pylattice.core.tensors import RealMatrix, ComplexMatrix, VectorReal, VectorComplex, VectorRealMatrix, VectorComplexMatrix
from pylattice.core.parallel import CartesianComm
import numpy as np
import pytest

mpigrid = [1, 1]
grid = [4, 4]
tol = 10e-6
CC = CartesianComm(mpigrid=mpigrid)


def test_realfield():
    Lattice = LatticeReal(grid=grid, cartesiancomm=CC)
    Lattice.fill_value( n = np.array([i for i in range(np.prod(grid))], dtype='float32') )
    Lattice_E, Lattice_O = Lattice.peek_EO_lattices()
    # even sites: coordinate sum even, so the origin (value 0) is in E
    assert( (np.abs(Lattice_E.value - np.array([ 0.,  2.,  5.,  7.,  8., 10., 13., 15.]))<tol ).all())
    assert( (np.abs(Lattice_O.value - np.array([ 1.,  3.,  4.,  6.,  9., 11., 12., 14.]))<tol ).all())
    Lattice_E = Lattice_E*-1
    Lattice.poke_EO_lattices(E_lattice=Lattice_E, O_lattice=Lattice_O)
    assert( (np.abs(Lattice.value - np.array([0.,   1.,  -2.,   3.,   4.,  -5.,   6.,
                                               -7.,  -8.,   9., -10.,  11.,  12., -13.,
                                               14., -15.]))<tol).all())

    print('[+] TEST grid={} mpigrid={}: LatticeReal EO passed'.format('x'.join([str(a)
          for a in grid]), 'x'.join([str(a) for a in [1, 1]])))


def test_half_lattice_bookkeeping():
    # each half is a LatticeReal whose grid matches its storage (last dimension halved),
    # so sizes, reducesum and average are those of the real half-lattice
    for g in ([4, 4], [2, 6], [4, 2, 2], [2, 2, 2, 2]):
        L = LatticeReal(grid=g, cartesiancomm=CartesianComm(mpigrid=[1]*len(g)))
        vals = np.arange(np.prod(g), dtype='float32')**2
        L.fill_value(n=vals.copy())
        parity = np.indices(g).sum(axis=0).flatten() % 2
        E, O = L.peek_EO_lattices()
        for half, ref in ((E, vals[parity == 0]), (O, vals[parity == 1])):
            assert(list(half.grid) == g[:-1] + [g[-1]//2])
            assert(len(half.value) == len(half.flat_idx) == np.prod(g)//2)
            assert( (half.value == ref).all() )
            assert( np.abs(half.reducesum() - ref.sum()) < tol*ref.sum() )
            assert( np.abs(half.average() - ref.mean()) < tol*ref.mean() )
        assert(E.value[0] == vals[0])                       # the origin is even
        L2 = LatticeReal(grid=g, cartesiancomm=CartesianComm(mpigrid=[1]*len(g)))
        L2.fill_value(n=np.zeros(np.prod(g), dtype='float32'))
        L2.poke_EO_lattices(E_lattice=E, O_lattice=O)       # peek then poke is the identity
        assert( (L2.value == vals).all() )
    print('[+] TEST half-lattice grid, sums and averages passed')


def test_eo_shifts():
    # A shift by one step hops between the sublattices. A field on the sites of parity p, moved
    # forward (backward) along mu, lives on the sites of parity 1-p and holds the value at x + mu
    # (x - mu): the same as np.roll of the full periodic lattice, read on the other sublattice.
    for g in ([4, 4], [2, 6], [4, 2, 2]):
        d = len(g)
        L = LatticeReal(grid=g, cartesiancomm=CartesianComm(mpigrid=[1]*d))
        vals = np.random.default_rng(7).standard_normal(np.prod(g)).astype('float32')
        glob = vals.reshape(g)
        par = np.indices(g).sum(axis=0).flatten() % 2
        for p in (0, 1):
            fresh = lambda: (L.fill_value(n=vals.copy()), L.peek_EO_lattices()[p])[1]
            for mu in range(d):
                for move, back, roll in (('moveforward', 'movebackward', -1), ('movebackward', 'moveforward', 1)):
                    H = fresh()
                    getattr(H, move)(mu=mu)
                    assert(H.eo_parity == 1 - p)
                    assert( (H.value == np.roll(glob, roll, axis=mu).flatten()[par == 1 - p]).all() )
                    getattr(H, back)(mu=mu)                     # hop back to where it came from
                    assert(H.eo_parity == p and (H.value == vals[par == p]).all())
            # nearest-neighbour sum: every site gets the sum of its 2d neighbours, taken from the other sublattice
            acc = None
            for mu in range(d):
                for move in ('moveforward', 'movebackward'):
                    H = fresh()
                    getattr(H, move)(mu=mu)
                    acc = H if acc is None else acc + H         # parities match, tags are kept by +
            ref = sum(np.roll(glob, r, axis=mu) for mu in range(d) for r in (-1, 1)).flatten()[par == 1 - p]
            assert(acc.eo_parity == 1 - p and np.abs(acc.value - ref).max() < 1e-5)
            # the result can be poked into the sublattice it lives on, and only there
            target = LatticeReal(grid=g, cartesiancomm=CartesianComm(mpigrid=[1]*d))
            target.poke_EO_lattices(**{('E_lattice', 'O_lattice')[1 - p]: acc})
            assert( (target.value[par == 1 - p] == acc.value).all() and not target.value[par == p].any() )
            with pytest.raises(ValueError):
                target.poke_EO_lattices(**{('E_lattice', 'O_lattice')[p]: acc})
        E, O = L.peek_EO_lattices()
        with pytest.raises(ValueError):                         # E + O would add values of different sites
            E + O
        with pytest.raises(ValueError):                         # only one step exists
            E.moveforward(mu=0, step=2)
        assert(E.eo_parity == 0)                                # untouched by the failed call
        with pytest.raises(ValueError):                         # a half-lattice has no halves
            E.peek_EO_lattices()
    print('[+] TEST even/odd shifts hop between sublattices')


def test_odd_local_grid_raises():
    # every local dimension must be even, otherwise the local parity is not the global one
    for odd in ([3, 4], [4, 3], [3, 3], [4, 4, 5], [1, 4]):
        L = LatticeReal(grid=odd, cartesiancomm=CartesianComm(mpigrid=[1]*len(odd)))
        for call in (L.get_EO_idx, L.peek_EO_lattices, lambda: L.poke_EO_lattices(E_lattice=L)):
            with pytest.raises(ValueError):
                call()
    for even in ([2, 2], [2, 6], [4, 2, 2]):
        L = LatticeReal(grid=even, cartesiancomm=CartesianComm(mpigrid=[1]*len(even)))
        E_idx, O_idx = L.get_EO_idx()
        assert(E_idx.sum() == O_idx.sum() == np.prod(even)//2)
    print('[+] TEST odd local grids raise ValueError, even ones pass')


test_realfield()
test_half_lattice_bookkeeping()
test_eo_shifts()
test_odd_local_grid_raises()