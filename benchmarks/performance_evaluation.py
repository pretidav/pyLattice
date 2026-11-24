"""
Strong-scaling benchmark: (A*B).reducesum() on a LatticeRealMatrix (N=2).
The global grid is fixed (default 40x40x40x40) and split over the MPI grid.

  mpiexec -np 4 python benchmarks/performance_evaluation.py --mpigrid=2x2x1x1
  python benchmarks/performance_evaluation.py --naive        # no MPI, plain loop

Prints one JSON line (rank 0) with the best and all timings in seconds.
"""
import argparse
import json
import time

import numpy as np

from pylattice.core.grid import LatticeRealMatrix
from pylattice.core.parallel import CartesianComm


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--global-grid', default='40x40x40x40')
    p.add_argument('--mpigrid', default='1x1x1x1')
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--naive', action='store_true',
                   help='plain per-site np.dot loop on the whole grid, no MPI')
    return p.parse_args()


def dims(s):
    return [int(a) for a in s.split('x')]


def naive(global_grid, repeat):
    n = int(np.prod(global_grid))
    a = np.arange(n * 4, dtype='float32').reshape(n, 2, 2)
    b = a.copy()
    out = np.empty_like(a)
    times = []
    for _ in range(repeat):
        start = time.time()
        for i in range(n):
            out[i] = np.dot(a[i], b[i])
        out.sum()
        times.append(time.time() - start)
    return times


def mpi_run(global_grid, mpigrid, repeat):
    CC = CartesianComm(mpigrid=mpigrid)
    local = [g // m for g, m in zip(global_grid, mpigrid)]
    assert [l * m for l, m in zip(local, mpigrid)] == global_grid
    n = int(np.prod(local))
    data = np.arange(n * 4, dtype='float32').reshape(n, 2, 2)
    A = LatticeRealMatrix(grid=local, cartesiancomm=CC, N=2)
    B = LatticeRealMatrix(grid=local, cartesiancomm=CC, N=2)
    A.fill_value(n=data)
    B.fill_value(n=data)
    times = []
    for _ in range(repeat):
        CC.comm.Barrier()
        start = time.time()
        (A * B).reducesum()
        times.append(time.time() - start)
    return CC.rank, times


if __name__ == '__main__':
    args = parse()
    gg = dims(args.global_grid)
    if args.naive:
        rank, times, procs = 0, naive(gg, args.repeat), 1
    else:
        mg = dims(args.mpigrid)
        rank, times = mpi_run(gg, mg, args.repeat)
        procs = int(np.prod(mg))
    if rank == 0:
        print(json.dumps({'procs': procs, 'naive': args.naive,
                          'best': min(times), 'times': times}))
