"""Read half of a cross-layout test of pylattice.core.hdf5io: see test_hdf5io_w.py, which must have been run
first, in the same directory and (usually) with a different number of ranks.

    mpiexec -n 4 python test_hdf5io_w.py
    mpiexec -n 2 python test_hdf5io_r.py            # reads with rank grid 1x1x2

Another layout can be given as an argument, e.g. `mpiexec -n 8 python test_hdf5io_r.py 2x2x2`.
"""
import sys

import numpy as np

from pylattice.core.hdf5io import Reader
from pylattice.core.parallel import CartesianComm
from pylattice.core.grid import LatticeComplex, LatticeReal, LatticeComplexMatrix, LatticeRealMatrix, LatticeVectorReal, LatticeVectorComplex, LatticeVectorRealMatrix, LatticeVectorComplexMatrix

# the same definitions as in test_hdf5io_w.py
GLOBAL = [4, 4, 4]
TYPES = [('LatticeReal', LatticeReal, {}, (), 'float64'),
         ('LatticeComplex', LatticeComplex, {}, (), 'complex128'),
         ('LatticeRealMatrix', LatticeRealMatrix, {'N': 2}, (2, 2), 'float64'),
         ('LatticeComplexMatrix', LatticeComplexMatrix, {'N': 2}, (2, 2), 'complex128'),
         ('LatticeVectorReal', LatticeVectorReal, {'Nd': 3}, (3,), 'float64'),
         ('LatticeVectorComplex', LatticeVectorComplex, {'Nd': 3}, (3,), 'complex128'),
         ('LatticeVectorRealMatrix', LatticeVectorRealMatrix, {'Nd': 3, 'N': 2}, (3, 2, 2), 'float64'),
         ('LatticeVectorComplexMatrix', LatticeVectorComplexMatrix, {'Nd': 3, 'N': 2}, (3, 2, 2), 'complex128')]
PREFIX = './test_file_cross_'


def global_field(site_shape, dtype):
    base = np.arange(int(np.prod(GLOBAL + list(site_shape))), dtype='float64').reshape(GLOBAL + list(site_shape))
    return (base + 1j*(2*base + 1) if dtype.startswith('complex') else base).astype(dtype)


mpigrid = [int(a) for a in sys.argv[1].split('x')] if len(sys.argv) > 1 else [1, 1, 2]
assert all(g % m == 0 for g, m in zip(GLOBAL, mpigrid)), 'rank grid {} does not divide {}'.format(mpigrid, GLOBAL)
grid = [g//m for g, m in zip(GLOBAL, mpigrid)]
CC = CartesianComm(mpigrid=mpigrid)


def test_read():
    block = tuple(slice(c*g, (c+1)*g) for c, g in zip(CC.mpicoord, grid))
    problems = []
    for name, cls, kw, site_shape, dtype in TYPES:
        Lattice = Reader(cartesiancomm=CC).read(filename=PREFIX + name)
        expected = global_field(site_shape, dtype)[block].reshape((-1,) + site_shape)
        if type(Lattice) is not cls:
            problems.append('{}: read back as {}'.format(name, type(Lattice).__name__))
        elif list(Lattice.grid) != grid:
            problems.append('{}: local grid {} instead of {}'.format(name, list(Lattice.grid), grid))
        elif Lattice.value.dtype != dtype or Lattice.value.shape != expected.shape:
            problems.append('{}: value is {} {}'.format(name, Lattice.value.shape, Lattice.value.dtype))
        elif not np.array_equal(Lattice.value, expected):
            problems.append('{}: values differ from the known global field'.format(name))
        elif any(getattr(Lattice, key) != val for key, val in kw.items()):
            problems.append('{}: sizes {} instead of {}'.format(name, {k: getattr(Lattice, k) for k in kw}, kw))
    # the file written under a name that already ended in .hdf5 is read with that name as it is
    Lattice = Reader(cartesiancomm=CC).read(filename=PREFIX + 'ext.hdf5')
    if not np.array_equal(Lattice.value, global_field((), 'float64')[block].reshape(-1)):
        problems.append('LatticeReal: values read from a name ending in .hdf5 differ')
    # the file with the full class path in its header, as older versions wrote it
    Lattice = Reader(cartesiancomm=CC).read(filename=PREFIX + 'legacy')
    if type(Lattice) is not LatticeReal or not np.array_equal(Lattice.value, global_field((), 'float64')[block].reshape(-1)):
        problems.append('LatticeReal: a file with the old-style type attribute is not read back')
    allproblems = CC.comm.allgather(problems)                   # every rank fails together
    allproblems = [p for rank, ps in enumerate(allproblems) for p in ['rank {}: {}'.format(rank, q) for q in ps]]
    assert not allproblems, allproblems
    if CC.rank == 0:
        print('[+] TEST read {} lattice types with rank grid {}: all match the known global field'.format(
            len(TYPES), 'x'.join(map(str, mpigrid))))


if __name__ == '__main__':
    test_read()
