"""Tests of the offline measurements of Ising configurations (pylattice.core.measure and
pylattice.theories.spin.ising_meas). The checkpoints need h5py with parallel HDF5 (skipped without).
Run on one process or on several ranks:

    pytest tests/theories/spin/test_ising_meas.py
    mpiexec -n 4 python tests/theories/spin/test_ising_meas.py        # 1, 2 or 4 ranks
"""
import contextlib
import json
import os
import shutil
import tempfile

import numpy as np
import pytest
from mpi4py import MPI

from pylattice.core.callbacks import Checkpointer, MeasurementFile
from pylattice.core.grid import LatticeReal
from pylattice.core.hdf5io import Writer
from pylattice.core.measure import measure_files
from pylattice.core.parallel import CartesianComm
from pylattice.theories.spin import ising_meas
from pylattice.theories.spin.ising import IsingField, IsingModel

COMM = MPI.COMM_WORLD
RANK, SIZE = COMM.Get_rank(), COMM.Get_size()
MPIGRID = {1: [1, 1], 2: [2, 1], 4: [2, 2]}[SIZE]
HDF5 = pytest.mark.skipif(not (lambda: __import__('h5py').get_config().mpi)(), reason='h5py with parallel HDF5 is not installed')


@contextlib.contextmanager
def shared_tempdir():
    path = COMM.bcast(tempfile.mkdtemp() if RANK == 0 else None, root=0)
    try:
        yield path
    finally:
        COMM.Barrier()
        if RANK == 0:
            shutil.rmtree(path, ignore_errors=True)


def setup(G):
    cc = CartesianComm(mpigrid=MPIGRID)
    return cc, [g // m for g, m in zip(G, MPIGRID)]


def model_with(cc, loc, glob, step_rng=1):
    field = IsingField(grid=loc, cartesiancomm=cc)
    block = tuple(slice(c * g, (c + 1) * g) for c, g in zip(cc.mpicoord, loc))
    field.value[:] = glob[block].reshape(-1)
    return IsingModel(field=field, logger=None, beta=0.5, rng=cc.rng(step_rng))


def exact(glob):
    """per site: m and the energy of H = -sum_<ij> s_i s_j with periodic boundaries"""
    bonds = sum((glob * np.roll(glob, -1, axis=mu)).sum() for mu in range(glob.ndim))
    return glob.mean(), -bonds / glob.size


def read(path):
    return json.load(open(path))['measurements']


@HDF5
def test_the_observables_of_known_configurations_and_the_order_of_the_steps():
    G = (8, 4)
    cc, loc = setup(G)
    configs = {5: np.ones(G), 1: 1.0 - 2.0 * (np.indices(G).sum(axis=0) % 2),
               9: np.random.default_rng(3).choice([-1.0, 1.0], size=G)}
    with shared_tempdir() as d:
        paths = []
        for step, glob in configs.items():                          # saved out of order
            paths.append(Checkpointer(1, directory=d, prefix='c').save(model_with(cc, loc, glob), step))
        out = os.path.join(d, 'meas.json')
        measure_files(paths, lambda p: ising_meas.measure_ising(p, cc, loc), COMM, filename=out)
        COMM.Barrier()
        records = read(out)
        assert [r['sweep'] for r in records] == [1, 5, 9]                   # in the order of the steps
        for r in records:
            m, e = exact(configs[r['sweep']])
            assert r['m'] == pytest.approx(m, abs=1e-6) and r['e'] == pytest.approx(e, abs=1e-6)
            assert r['abs_m'] == pytest.approx(abs(m), abs=1e-6) and r['m2'] == pytest.approx(m * m, abs=1e-6)
            assert r['e2'] == pytest.approx(e * e, abs=1e-5) and r['acceptance'] is None
        assert records[1]['m'] == 1.0 and records[1]['e'] == -2.0           # all up: every bond is -1, 2 bonds per site


@HDF5
def test_offline_measurements_equal_the_ones_made_during_the_run():
    cc, loc = setup([8, 4])
    rng = cc.rng(7)
    field = IsingField(grid=loc, cartesiancomm=cc, initialization='random', seed=rng)
    model = IsingModel(field=field, logger=None, beta=0.4, rng=rng)
    with shared_tempdir() as d:
        online = os.path.join(d, 'online.json')
        model.run_mc(steps=6, online_obs=True, progress=False,
                     callbacks=[Checkpointer(1, directory=d, prefix='c'), MeasurementFile(online)])
        COMM.Barrier()
        files = [os.path.join(d, 'c_{:08d}.hdf5'.format(k)) for k in range(1, 7)]
        offline = os.path.join(d, 'offline.json')
        measure_files(files, lambda p: ising_meas.measure_ising(p, cc, loc), COMM, filename=offline)
        COMM.Barrier()
        a, b = read(online)[1:], read(offline)                              # sweep 0 (no checkpoint) left out
        assert [r['sweep'] for r in a] == [r['sweep'] for r in b] == list(range(1, 7))
        for x, y in zip(a, b):
            assert y['m'] == pytest.approx(x['m'], abs=1e-6) and y['e'] == pytest.approx(x['e'], abs=1e-6)


@HDF5
def test_a_file_that_is_not_an_ising_configuration_stops_the_run_and_keeps_what_was_measured():
    cc, loc = setup([8, 4])
    good = model_with(cc, loc, np.ones((8, 4)))
    with shared_tempdir() as d:
        p1 = Checkpointer(1, directory=d, prefix='c').save(good, 1)
        notspins = LatticeReal(grid=loc, cartesiancomm=cc)
        notspins.value[:] = 0.5
        p2 = os.path.join(d, 'bad.hdf5')
        Writer(lattice=notspins, cartesiancomm=cc).write(p2)
        out = os.path.join(d, 'meas.json')
        with pytest.raises(ValueError, match='not an Ising configuration'):
            measure_files([p1, p2], lambda p: ising_meas.measure_ising(p, cc, loc), COMM, filename=out)
        COMM.Barrier()
        assert [r['sweep'] for r in read(out)] == [1]                      # valid JSON, the good one is in
        with pytest.raises(ValueError, match='no configuration files'):
            measure_files([], lambda p: {}, COMM, filename=out)


@HDF5
def test_command_line():
    cc, loc = setup([8, 4])
    with shared_tempdir() as d:
        paths = [Checkpointer(1, directory=d, prefix='c').save(model_with(cc, loc, np.ones((8, 4))), s) for s in (2, 1)]
        out = os.path.join(d, 'm.json')
        ising_meas.main(paths + ['--mpigrid=' + 'x'.join(map(str, MPIGRID)), '--out', out])
        COMM.Barrier()
        records = read(out)
        assert [r['sweep'] for r in records] == [1, 2] and all(r['m'] == 1.0 for r in records)
        with pytest.raises(SystemExit):                                        # a layout that does not fit the ranks
            ising_meas.main(paths + ['--mpigrid=3x3', '--out', out])
        with pytest.raises(SystemExit):
            ising_meas.main([os.path.join(d, 'missing.hdf5'), '--mpigrid=' + 'x'.join(map(str, MPIGRID)), '--out', out])


if __name__ == '__main__':
    import traceback
    tests = [(name, f) for name, f in sorted(globals().items()) if name.startswith('test_') and callable(f)]
    for name, f in tests:
        try:
            f()
        except BaseException:
            print('[x] {} failed on rank {}'.format(name, RANK), flush=True)
            traceback.print_exc()
            COMM.Abort(1)
        if RANK == 0:
            print('[+] {} passed on {} rank(s)'.format(name, SIZE))
