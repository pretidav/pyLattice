"""Tests of pylattice.core.callbacks: RunLog and Checkpointer, driven by pylattice.core.montecarlo.
The checkpoint tests need h5py built with parallel HDF5 and are skipped without it. Run on one
process or on several ranks:

    pytest tests/core/test_callbacks.py
    mpiexec -n 2 python tests/core/test_callbacks.py        # 1, 2 or 4 ranks
"""
import contextlib
import os
import shutil
import tempfile
import time

import numpy as np
import psutil
import pytest
from mpi4py import MPI

from pylattice.core import montecarlo
from pylattice.core.callbacks import Checkpointer, RunLog
from pylattice.core.grid import LatticeReal
from pylattice.core.parallel import CartesianComm

COMM = MPI.COMM_WORLD
RANK, SIZE = COMM.Get_rank(), COMM.Get_size()


def parallel_hdf5():
    try:
        import h5py
        return bool(h5py.get_config().mpi)
    except ImportError:
        return False


def needs_hdf5(f):
    f.needs_hdf5 = True
    return pytest.mark.skipif(not parallel_hdf5(), reason='h5py with parallel HDF5 is not installed')(f)


@contextlib.contextmanager
def shared_tempdir():
    """one directory for all the ranks (the checkpoint is written by all of them)"""
    path = COMM.bcast(tempfile.mkdtemp() if RANK == 0 else None, root=0)
    try:
        yield path
    finally:
        COMM.Barrier()
        if RANK == 0:
            shutil.rmtree(path, ignore_errors=True)


def spin(seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


class Toy:
    """counts its sweeps; observables x = sweeps and y = 2 sweeps. With a field, every sweep sets the
    field to 100 * sweeps + rank, so a checkpoint tells at which sweep it was taken."""

    def __init__(self, work=None, field=None):
        self.comm = COMM
        self.sweeps = 0
        self.calls = []
        self.work = work
        self.field = field

    def sweep(self, rng):
        self.sweeps += 1
        self.calls.append('s')
        if self.work:
            self.work()
        if self.field is not None:
            self.field.value[:] = 100.0 * self.sweeps + RANK
        return 1, 4

    def observables(self):
        self.calls.append('o')
        return {'x': float(self.sweeps), 'y': 2.0 * self.sweeps}


def make_field():
    cc = CartesianComm(mpigrid={1: [1, 1], 2: [2, 1], 4: [2, 2]}[SIZE])
    return LatticeReal(grid=[4, 4], cartesiancomm=cc)


def parse(line):
    return [tuple(item.split('=')) for item in line.split()]


# ============================================================================ RunLog
def test_runlog_one_line_per_log_step_with_the_requested_fields():
    with shared_tempdir() as d:
        path = os.path.join(d, 'run.log')
        montecarlo.run(Toy(), steps=10, thermalization=2, progress=False, callbacks=[RunLog(3, path)])
        everyone = COMM.allreduce(psutil.Process().memory_info().rss) / 2 ** 20           # all the ranks together
        if RANK != 0:
            return
        lines = open(path).read().splitlines()
        assert len(lines) == 4                                           # sweeps 3, 6, 9, 12 (thermalization counts); not one set per rank
        for line, step in zip(lines, (3, 6, 9, 12)):
            fields = parse(line)
            assert [k for k, _ in fields] == ['step', 'x', 'y', 'mem_MB', 'cpu_pct', 'steps_per_s']
            values = dict(fields)
            assert int(values['step']) == step
            assert float(values['x']) == step and float(values['y']) == 2 * step     # the observables of that sweep
            assert 0.7 * everyone < float(values['mem_MB']) < 1.3 * everyone     # the memory of all the ranks, not of one
            assert 0.0 <= float(values['cpu_pct']) <= 1000.0
            assert float(values['steps_per_s']) > 0


def test_runlog_observable_format():
    class Odd(Toy):
        def observables(self):
            return {'m': 0.123456789, 'e': -2, 'big': 1234567.0}
    with shared_tempdir() as d:
        path = os.path.join(d, 'run.log')
        montecarlo.run(Odd(), steps=1, progress=False, callbacks=[RunLog(1, path)])
        if RANK == 0:
            assert open(path).read().split()[:4] == ['step=1', 'm=0.123457', 'e=-2', 'big=1.23457e+06']


def test_runlog_speed_is_sweeps_per_second():
    with shared_tempdir() as d:
        path = os.path.join(d, 'run.log')
        montecarlo.run(Toy(work=lambda: time.sleep(0.01)), steps=9, progress=False, callbacks=[RunLog(3, path)])
        if RANK == 0:
            speeds = [float(dict(parse(l))['steps_per_s']) for l in open(path).read().splitlines()]
            assert len(speeds) == 3 and all(20 < v <= 100.5 for v in speeds), speeds       # 0.01 s per sweep


def test_runlog_cpu_follows_the_use_of_the_processor():
    with shared_tempdir() as d:
        busy, idle = os.path.join(d, 'busy.log'), os.path.join(d, 'idle.log')
        montecarlo.run(Toy(work=lambda: spin(0.02)), steps=6, progress=False, callbacks=[RunLog(3, busy)])
        montecarlo.run(Toy(work=lambda: time.sleep(0.02)), steps=6, progress=False, callbacks=[RunLog(3, idle)])
        if RANK == 0:
            cpu = lambda path: [float(dict(parse(l))['cpu_pct']) for l in open(path).read().splitlines()]
            assert all(50 < c < 150 for c in cpu(busy)), cpu(busy)       # a core, spinning: the mean over the ranks, not the sum
            assert all(c < 30 for c in cpu(idle)), cpu(idle)             # asleep


def test_runlog_lines_are_flushed_as_they_are_written():
    seen = []
    with shared_tempdir() as d:
        path = os.path.join(d, 'run.log')
        peek = lambda model, step, observables: seen.append(len(open(path).read().splitlines())) if RANK == 0 else None
        montecarlo.run(Toy(), steps=4, progress=False, callbacks=[RunLog(2, path), peek])
        if RANK == 0:
            assert seen == [0, 1, 1, 2]                                  # the file is read while the run goes on


def test_runlog_appends_by_default_and_can_start_afresh():
    with shared_tempdir() as d:
        path = os.path.join(d, 'run.log')
        for _ in range(2):
            montecarlo.run(Toy(), steps=4, progress=False, callbacks=[RunLog(2, path)])
        if RANK == 0:
            assert len(open(path).read().splitlines()) == 4
        montecarlo.run(Toy(), steps=4, progress=False, callbacks=[RunLog(2, path, mode='w')])
        if RANK == 0:
            assert len(open(path).read().splitlines()) == 2
        COMM.Barrier()


def test_runlog_does_not_ask_for_the_observables_twice():
    toy = Toy()
    with shared_tempdir() as d:
        montecarlo.run(toy, steps=6, progress=False, callbacks=[RunLog(1, os.path.join(d, 'run.log'))])
    assert toy.calls.count('o') == 7                                     # sweep 0 and one per sweep, shared with the History


def test_every_must_be_a_positive_integer():
    for bad in (0, -1, 1.5):
        with pytest.raises(ValueError):
            RunLog(bad)
        with pytest.raises(ValueError):
            Checkpointer(bad)
    with pytest.raises(ValueError):
        Checkpointer(5, keep=0)


# ====================================================================== Checkpointer
def list_checkpoints(d, prefix='t'):
    return sorted(f for f in os.listdir(d) if f.startswith(prefix + '_'))


@needs_hdf5
def test_checkpoints_every_n_sweeps_hold_the_configuration_of_that_sweep():
    from pylattice.core.hdf5io import Reader
    import h5py
    field = make_field()
    with shared_tempdir() as d:
        ck = os.path.join(d, 'ck')                                       # created by the Checkpointer
        montecarlo.run(Toy(field=field), steps=15, thermalization=2, progress=False,
                       callbacks=[Checkpointer(5, directory=ck, prefix='t')])
        assert list_checkpoints(ck) == ['t_00000005.hdf5', 't_00000010.hdf5', 't_00000015.hdf5']   # 17 sweeps in all
        for step in (5, 10, 15):
            path = os.path.join(ck, 't_{:08d}.hdf5'.format(step))
            back = Reader(cartesiancomm=field.cartesiancomm).read(filename=path[:-5])
            assert type(back) is LatticeReal and (back.value == 100.0 * step + RANK).all()
            if RANK == 0:
                with h5py.File(path, 'r') as f:
                    assert int(f.attrs['step']) == step and list(f.attrs['grid']) == list(field.grid * field.cartesiancomm.mpigrid)
        COMM.Barrier()


@needs_hdf5
def test_checkpoint_keep_and_final():
    field = make_field()
    with shared_tempdir() as d:
        montecarlo.run(Toy(field=field), steps=17, progress=False,
                       callbacks=[Checkpointer(5, directory=d, prefix='t', keep=2, final=True)])
        assert list_checkpoints(d) == ['t_00000015.hdf5', 't_00000017.hdf5']                 # the newest two, the last one is the final
    with shared_tempdir() as d:
        saves = []
        class Counting(Checkpointer):
            def save(self, model, step):
                saves.append(step)
                return super().save(model, step)
        montecarlo.run(Toy(field=field), steps=15, progress=False, callbacks=[Counting(5, directory=d, prefix='t', final=True)])
        assert saves == [5, 10, 15] and len(list_checkpoints(d)) == 3                         # not saved twice at the end
    with shared_tempdir() as d:
        montecarlo.run(Toy(field=field), steps=7, progress=False, callbacks=[Checkpointer(5, directory=d, prefix='t')])
        assert list_checkpoints(d) == ['t_00000005.hdf5']                                    # final is off by default


@needs_hdf5
def test_checkpoint_files_are_complete_or_absent():
    field = make_field()
    with shared_tempdir() as d:
        montecarlo.run(Toy(field=field), steps=6, progress=False, callbacks=[Checkpointer(2, directory=d, prefix='t')])
        assert not any('partial' in f for f in os.listdir(d))                                # renamed when complete
        if RANK == 0:                                                                        # left by a job that was killed
            open(os.path.join(d, 't_00000099.partial.hdf5'), 'w').close()
        COMM.Barrier()
        montecarlo.run(Toy(field=field), steps=2, progress=False, callbacks=[Checkpointer(2, directory=d, prefix='t')])
        assert not any('partial' in f for f in os.listdir(d))                                # and removed by the next run


@needs_hdf5
def test_a_failed_write_leaves_no_checkpoint():
    if SIZE != 1:
        return                                                                               # a failure of one rank would hang the others
    from pylattice.core import hdf5io
    field = make_field()
    original = hdf5io.Writer.write
    def failing(self, filename):
        open(filename, 'w').write('half a file')
        raise OSError('disk full')
    hdf5io.Writer.write = failing
    try:
        with shared_tempdir() as d:
            with pytest.raises(OSError):
                montecarlo.run(Toy(field=field), steps=4, progress=False, callbacks=[Checkpointer(2, directory=d, prefix='t')])
            assert not [f for f in os.listdir(d) if not f.endswith('.partial.hdf5')]         # nothing under a final name
    finally:
        hdf5io.Writer.write = original


@needs_hdf5
def test_the_lattice_to_save_can_be_chosen():
    field = make_field()
    toy = Toy()
    toy.state = field                                                                        # not called `field`
    with shared_tempdir() as d:
        montecarlo.run(toy, steps=2, progress=False, callbacks=[Checkpointer(2, directory=d, prefix='t', lattice=lambda m: m.state)])
        assert list_checkpoints(d) == ['t_00000002.hdf5']
        toy2 = Toy()                                                                         # no field and no choice: a clear error
        with pytest.raises(TypeError, match='no `field`'):
            montecarlo.run(toy2, steps=2, progress=False, callbacks=[Checkpointer(2, directory=d, prefix='u')])
        assert toy2.sweeps == 0 and list_checkpoints(d, 'u') == []


@needs_hdf5
def test_a_model_field_that_is_a_subclass_is_saved_as_its_lattice():
    from pylattice.core.hdf5io import Reader
    from pylattice.theories.spin.ising import IsingField, IsingModel
    cc = CartesianComm(mpigrid={1: [1, 1], 2: [2, 1], 4: [2, 2]}[SIZE])
    model = IsingModel(field=IsingField(grid=[4, 4], cartesiancomm=cc, initialization='random', seed=cc.rng(7)),
                       logger=None, beta=0.5)
    with shared_tempdir() as d:
        model.run_mc(steps=6, progress=False, callbacks=[Checkpointer(3, directory=d, prefix='t', final=True),
                                                         RunLog(2, os.path.join(d, 'run.log'))])
        back = Reader(cartesiancomm=cc).read(filename=os.path.join(d, 't_00000006'))
        assert type(back) is LatticeReal and np.array_equal(back.value, model.field.value)    # the end configuration
        if RANK == 0:
            lines = open(os.path.join(d, 'run.log')).read().splitlines()
            assert len(lines) == 3 and [k for k, _ in parse(lines[0])][:3] == ['step', 'm', 'e']
        COMM.Barrier()


if __name__ == '__main__':
    import traceback
    tests = [(name, f) for name, f in sorted(globals().items()) if name.startswith('test_') and callable(f)]
    for name, f in tests:
        if getattr(f, 'needs_hdf5', False) and not parallel_hdf5():
            if RANK == 0:
                print('[-] {} skipped: no h5py with parallel HDF5'.format(name))
            continue
        try:
            f()
        except BaseException:
            # a rank that fails must stop them all: the others would wait for it in a collective call
            print('[x] {} failed on rank {}'.format(name, RANK), flush=True)
            traceback.print_exc()
            COMM.Abort(1)
        if RANK == 0:
            print('[+] {} passed on {} rank(s)'.format(name, SIZE))
