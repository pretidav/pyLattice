"""Tests of pylattice.core.rng: an independent random stream for every rank. Run on one process or
on several ranks:

    pytest tests/core/test_rng.py
    mpiexec -n 4 python tests/core/test_rng.py        # 1, 2 or 4 ranks
"""
import numpy as np
import pytest
from mpi4py import MPI

from pylattice.core.parallel import CartesianComm
from pylattice.core.rng import rank_rng

COMM = MPI.COMM_WORLD
RANK, SIZE = COMM.Get_rank(), COMM.Get_size()


class FakeComm:
    """a communicator that only knows its rank, to look at many ranks from one process"""

    def __init__(self, rank):
        self.rank = rank

    def Get_rank(self):
        return self.rank


def test_the_stream_of_a_rank_is_the_one_of_seedsequence_spawn():
    for seed in (0, 1, 12345):
        expected = np.random.default_rng(np.random.SeedSequence(seed).spawn(SIZE)[RANK])
        assert np.array_equal(rank_rng(seed).random(8), expected.random(8))


def test_a_rank_has_the_same_stream_whatever_the_number_of_ranks():
    # nothing is communicated and the size does not enter: rank r of 2 ranks, of 4, or of 1000
    for seed in (3, 99):
        expected = lambda: np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(RANK,))).random(8)
        assert np.array_equal(rank_rng(seed, COMM).random(8), expected())
        assert np.array_equal(rank_rng(seed, FakeComm(RANK)).random(8), expected())     # seen from a job of any size


def test_reproducible_and_dependent_on_the_seed():
    a, b, c = rank_rng(5), rank_rng(5), rank_rng(6)
    x, y, z = a.random(100), b.random(100), c.random(100)
    assert np.array_equal(x, y) and not np.array_equal(x, z)
    assert isinstance(rank_rng(1), np.random.Generator)


def test_the_ranks_have_independent_streams():
    n = 5000
    draws = COMM.allgather(rank_rng(7).random(n))
    assert len(draws) == SIZE
    for i in range(SIZE):
        assert abs(draws[i].mean() - 0.5) < 5 * np.sqrt(1 / 12 / n)                 # uniform
        for j in range(i + 1, SIZE):
            assert not np.array_equal(draws[i], draws[j])
            assert abs(np.corrcoef(draws[i], draws[j])[0, 1]) < 5 / np.sqrt(n)    # not correlated


def test_many_ranks_never_share_a_stream():
    # a thousand ranks looked at from one process; the old scheme (a seed from 0..9999 per rank)
    # would have given some of them the same stream
    firsts = [rank_rng(11, FakeComm(r)).integers(0, 2 ** 62) for r in range(1000)]
    assert len(set(firsts)) == 1000


def test_cartesiancomm_gives_the_generator_of_its_rank():
    cc = CartesianComm(mpigrid={1: [1, 1], 2: [2, 1], 4: [2, 2]}[SIZE])
    assert np.array_equal(cc.rng(4).random(6), rank_rng(4, cc.comm).random(6))
    assert np.array_equal(cc.rng().random(6), rank_rng(0).random(6))                # seed 0 by default


def test_parallel_RNG_gives_distinct_integer_seeds_without_communicating():
    cc = CartesianComm(mpigrid={1: [1, 1], 2: [2, 1], 4: [2, 2]}[SIZE])
    seed = cc.parallel_RNG(seed=8)
    assert type(seed) is int and 0 <= seed < 2 ** 32 and cc.parallel_RNG(seed=8) == seed
    assert len(set(COMM.allgather(seed))) == SIZE
    assert cc.parallel_RNG(seed=9) != seed


def test_an_invalid_seed_is_refused():
    for bad in (-1, 1.5, 'x'):
        with pytest.raises((ValueError, TypeError)):
            rank_rng(bad)


if __name__ == '__main__':
    import traceback
    tests = [(name, f) for name, f in sorted(globals().items()) if name.startswith('test_') and callable(f)]
    for name, f in tests:
        try:
            f()
        except BaseException:
            # a rank that fails must stop them all: the others would wait for it in a collective call
            print('[x] {} failed on rank {}'.format(name, RANK), flush=True)
            traceback.print_exc()
            COMM.Abort(1)
        if RANK == 0:
            print('[+] {} passed on {} rank(s)'.format(name, SIZE))
