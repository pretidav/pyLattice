"""
Creates an HDF5 file with a single dataset of shape (channels, n),
filled with random numbers.
Writing to the different channels (rows) is parallelized using MPI.
Usage:
  mpirun -np 8 python demo.py
Small shell script to run timings with different numbers of MPI processes:
  for np in 1 2 4 8 12 16 20 24 28 32; do
      echo -n "$np ";
      /usr/bin/time --format="%e" mpirun -np $np python demo.py;
  done
"""

from mpi4py import MPI
import h5py
import numpy as np


n = 10000000
channels = 32
num_processes = MPI.COMM_WORLD.size
rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)

np.random.seed(746574366 + rank)

f = h5py.File('parallel_test.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD)
dset = f.create_dataset('test', (channels, n), dtype='f')

for i in range(channels):
    if i % num_processes == rank:
       print("rank = {}, i = {}".format(rank, i))
       data = np.random.uniform(size=n)
       dset[i] = data

f.close()
