"""Independent random number streams for the MPI ranks."""
import numpy as np
from mpi4py import MPI


def rank_rng(seed=0, comm=None):
    """The numpy Generator of this rank.

    It is stream number `rank` of the seed: SeedSequence(seed, spawn_key=(rank,)), the same as
    SeedSequence(seed).spawn(size)[rank]. Streams of different ranks are statistically independent
    (that is what SeedSequence is for), and nothing is communicated: every rank builds its own.
    Rank r gets the same stream whatever the number of ranks, so a run is reproducible for a given
    seed and layout of the lattice.

    seed  a non-negative integer (or a sequence of them)
    comm  default COMM_WORLD
    """
    comm = MPI.COMM_WORLD if comm is None else comm
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(comm.Get_rank(),)))
