"""Write half of a cross-layout test of IO.MPI_IO (the read half is test_mpi_io_r.py).

A known global 4x4x4 field is written for every lattice type, split over one MPI layout; the
reader then reads it back with a *different* layout and checks it against the same field.
Both scripts must run in the same directory:

    mpiexec -n 4 python test_mpi_io_w.py            # writes with rank grid 1x2x2
    mpiexec -n 2 python test_mpi_io_r.py            # reads with rank grid 1x1x2

Another layout can be given as an argument, e.g. `mpiexec -n 8 python test_mpi_io_w.py 2x2x2`.
"""
import os
import sys

import h5py
import numpy as np

from IO.MPI_IO import Writer
from mpi.parallel import CartesianComm
from lattice.grid import LatticeComplex, LatticeReal, LatticeComplexMatrix, LatticeRealMatrix, LatticeVectorReal, LatticeVectorComplex, LatticeVectorRealMatrix, LatticeVectorComplexMatrix

GLOBAL = [4, 4, 4]
# name, class, constructor arguments, shape of the value at one site, dtype
TYPES = [('LatticeReal', LatticeReal, {}, (), 'float32'),
         ('LatticeComplex', LatticeComplex, {}, (), 'complex64'),
         ('LatticeRealMatrix', LatticeRealMatrix, {'N': 2}, (2, 2), 'float32'),
         ('LatticeComplexMatrix', LatticeComplexMatrix, {'N': 2}, (2, 2), 'complex64'),
         ('LatticeVectorReal', LatticeVectorReal, {'Nd': 3}, (3,), 'float32'),
         ('LatticeVectorComplex', LatticeVectorComplex, {'Nd': 3}, (3,), 'complex64'),
         ('LatticeVectorRealMatrix', LatticeVectorRealMatrix, {'Nd': 3, 'N': 2}, (3, 2, 2), 'float32'),
         ('LatticeVectorComplexMatrix', LatticeVectorComplexMatrix, {'Nd': 3, 'N': 2}, (3, 2, 2), 'complex64')]
PREFIX = './test_file_cross_'


def global_field(site_shape, dtype):
    # every entry of every site is different (the same definition is used by the reader)
    base = np.arange(int(np.prod(GLOBAL + list(site_shape))), dtype='float64').reshape(GLOBAL + list(site_shape))
    return (base + 1j*(2*base + 1) if dtype.startswith('complex') else base).astype(dtype)


mpigrid = [int(a) for a in sys.argv[1].split('x')] if len(sys.argv) > 1 else [1, 2, 2]
assert all(g % m == 0 for g, m in zip(GLOBAL, mpigrid)), 'rank grid {} does not divide {}'.format(mpigrid, GLOBAL)
grid = [g//m for g, m in zip(GLOBAL, mpigrid)]
CC = CartesianComm(mpigrid=mpigrid)


def local_block():
    return tuple(slice(c*g, (c+1)*g) for c, g in zip(CC.mpicoord, grid))


def test_write():
    for name, cls, kw, site_shape, dtype in TYPES:
        Lattice = cls(grid=grid, cartesiancomm=CC, **kw)
        Lattice.fill_value(n=global_field(site_shape, dtype)[local_block()].reshape((-1,) + site_shape).copy())
        Writer(lattice=Lattice, cartesiancomm=CC).write(filename=PREFIX + name)
    # a filename that already ends in .hdf5 must be used as it is
    Lattice = LatticeReal(grid=grid, cartesiancomm=CC)
    Lattice.fill_value(n=global_field((), 'float32')[local_block()].reshape(-1).copy())
    Writer(lattice=Lattice, cartesiancomm=CC).write(filename=PREFIX + 'ext.hdf5')
    CC.comm.Barrier()

    problems = []
    if CC.rank == 0:                                    # inspect the files serially
        for name, cls, kw, site_shape, dtype in TYPES:
            with h5py.File(PREFIX + name + '.hdf5', 'r') as f:
                if list(f.attrs['grid']) != GLOBAL:
                    problems.append('{}: grid attribute {}'.format(name, list(f.attrs['grid'])))
                if f.attrs['type'] != "<class 'lattice.grid.{}'>".format(name):
                    problems.append('{}: type attribute {}'.format(name, f.attrs['type']))
                for key, val in kw.items():
                    if int(f.attrs[key]) != val:
                        problems.append('{}: attribute {} = {}'.format(name, key, f.attrs[key]))
                if f['value'].shape != (int(np.prod(GLOBAL)),) + site_shape or f['value'].dtype != dtype:
                    problems.append('{}: dataset {} {}'.format(name, f['value'].shape, f['value'].dtype))
        if not os.path.exists(PREFIX + 'ext.hdf5') or os.path.exists(PREFIX + 'ext.hdf5.hdf5'):
            problems.append('a name ending in .hdf5 must not get a second extension')
    problems = CC.comm.bcast(problems, root=0)
    assert not problems, problems
    if CC.rank == 0:
        print('[+] TEST wrote {} lattice types with rank grid {}'.format(len(TYPES), 'x'.join(map(str, mpigrid))))


if __name__ == '__main__':
    test_write()
