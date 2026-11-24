"""Tests of the offline measurements of SU(2) configurations (pylattice.theories.gauge.sun_meas): the plaquettes, the
Polyakov loop, the Wilson loops and the Creutz ratios, compared with a serial numpy computation on the whole lattice, and
their measurement on stored configurations (checkpoints, which need h5py with parallel HDF5: skipped without).
They run on one process or on a lattice split over 2 or 4 ranks:

    pytest tests/theories/gauge/test_gauge_sun_meas.py
    mpiexec -n 2 python tests/theories/gauge/test_gauge_sun_meas.py        # 2 ranks (1 or 4 also work)
"""
import contextlib
import json
import logging
import os
import shutil
import tempfile

import numpy as np
import pytest
from mpi4py import MPI

from pylattice.core.groups import su2 as group
from pylattice.core.parallel import CartesianComm
from pylattice.theories.gauge.field import SU2Field
from pylattice.theories.gauge.sun import SUNModel, plaquette_observables
from pylattice.theories.gauge.sun_meas import loop_observables, main, measure_su

COMM = MPI.COMM_WORLD
dag = lambda M: np.conj(np.swapaxes(M, -1, -2))


# ----------------------------------------------------------------------------- helpers
def setup(G):
    """communicator and local grid for a global lattice G, split over the ranks of this run (1, 2 or 4)"""
    size, dims = COMM.Get_size(), len(G)
    assert size in (1, 2, 4), 'run with 1, 2 or 4 ranks'
    mg = {1: [1] * dims, 2: [2] + [1] * (dims - 1), 4: [2, 2] + [1] * (dims - 2)}[size]
    return CartesianComm(mpigrid=mg), [g // m for g, m in zip(G, mg)]


def local_block(full, cc, local):
    """the part of a global array (*global grid, ...) that lives on this rank, flattened over the sites"""
    lo = [c * l for c, l in zip(cc.mpicoord, local)]
    return full[tuple(slice(a, a + l) for a, l in zip(lo, local))].reshape((int(np.prod(local)),) + full.shape[len(local):])


def global_field(G, seed=3, eps=0.5):
    """an SU2Field holding a random global configuration, and that configuration as a numpy array U[x..., mu, 2, 2].
    The links are random at a distance eps from the identity (like a configuration at weak coupling): the loops are positive
    and decrease with their size, so the Creutz ratios are defined (for Haar random links every loop is about 0)"""
    cc, loc = setup(list(G))
    U = group.to_matrix(group.near_identity_quaternion(eps, np.random.default_rng(seed), size=tuple(G) + (len(G),)))
    field = SU2Field(grid=loc, cartesiancomm=cc, Nd=len(G))
    field.value = local_block(U, cc, loc).copy()
    return cc, field, U


def reference_line(U, mu, R):
    out = moved = U[..., mu, :, :]
    for _ in range(R - 1):
        moved = np.roll(moved, -1, axis=mu)
        out = out @ moved
    return out


def reference_loop(U, mu, nu, R, T):
    a, b = reference_line(U, mu, R), reference_line(U, nu, T)
    return a @ np.roll(b, -R, axis=mu) @ dag(np.roll(a, -T, axis=nu)) @ dag(b)


def reference_average(U, R, T, pairs=None):
    Nd = U.ndim - 3
    pairs = [(m, n) for m in range(Nd) for n in range(Nd) if m != n] if pairs is None else pairs
    return np.mean([0.5 * np.trace(reference_loop(U, m, n, R, T), axis1=-2, axis2=-1) for m, n in pairs])


@contextlib.contextmanager
def shared_tempdir():
    path = COMM.bcast(tempfile.mkdtemp() if COMM.Get_rank() == 0 else None, root=0)
    old = os.getcwd()
    try:
        yield path
    finally:
        os.chdir(old)
        for handler in list(logging.getLogger('pylattice').handlers):
            logging.getLogger('pylattice').removeHandler(handler)
        COMM.Barrier()
        if COMM.Get_rank() == 0:
            shutil.rmtree(path, ignore_errors=True)                  # the directory this test made with mkdtemp


def parallel_hdf5():
    try:
        import h5py
        return bool(h5py.get_config().mpi)
    except ImportError:
        return False


def needs_hdf5(f):
    f.needs_hdf5 = True                                  # the script runner below skips it too
    return pytest.mark.skipif(not parallel_hdf5(), reason='h5py with parallel HDF5 is not installed')(f)


def read(path):
    return json.load(open(path))['measurements']


# ----------------------------------------------------------------------------- the observables, exactly
@pytest.mark.parametrize('G,max_loop', [([8, 4, 4], 3), ([4, 4], 2)])
def test_the_loops_equal_a_serial_computation_on_the_whole_lattice(G, max_loop):
    cc, field, U = global_field(G)
    o = loop_observables(field, max_loop)
    Nd, t = len(G), len(G) - 1
    pairs = [(m, n) for m in range(Nd) for n in range(Nd) if m != n]
    W = {(1, 1): reference_average(U, 1, 1).real}
    for T in range(1, max_loop + 1):
        for R in range(1, T + 1):
            if (R, T) != (1, 1):
                W[R, T] = reference_average(U, R, T).real
                assert abs(o['W{}{}'.format(R, T)] - W[R, T]) < 1e-12, (R, T)
                assert abs(reference_average(U, T, R).real - W[R, T]) < 1e-12        # the orientations are the same average
    for R in range(2, max_loop + 1):
        want = -np.log(W[R, R] * W[R - 1, R - 1] / W[R - 1, R] ** 2)
        assert abs(o['creutz{}{}'.format(R, R)] - want) < 1e-9, R
    assert abs(o['plaq_t'] - reference_average(U, 1, 1, [p for p in pairs if t in p]).real) < 1e-12
    assert ('plaq_s' in o) == (Nd >= 3)
    if Nd >= 3:
        assert abs(o['plaq_s'] - reference_average(U, 1, 1, [p for p in pairs if t not in p]).real) < 1e-12
    polyakov = 0.5 * np.trace(reference_line(U, t, G[t]), axis1=-2, axis2=-1).mean()    # the loop closes around the lattice
    assert abs(o['polyakov'] - polyakov.real) < 1e-12 and abs(o['polyakov_abs'] - abs(polyakov)) < 1e-12
    assert abs(o['W12']) > 1e-4 and abs(o['polyakov']) > 1e-4                            # not zero against zero
    assert abs(field.Wilson(1, 2).imag) < 1e-12 and abs(field.Polyakov(t).imag) < 1e-12   # SU(2): real
    assert set(o) == ({'plaq_s'} if Nd >= 3 else set()) | {'plaq_t', 'polyakov', 'polyakov_abs'} | \
        {'W{}{}'.format(R, T) for T in range(1, max_loop + 1) for R in range(1, T + 1) if (R, T) != (1, 1)} | \
        {'creutz{0}{0}'.format(R) for R in range(2, max_loop + 1)}


def test_the_plaquette_observables_and_the_plaquette_split_into_space_and_time():
    cc, field, U = global_field([8, 4, 4])
    p = plaquette_observables(field, 2.1)
    assert set(p) == {'plaq_re', 'plaq_im', 'action'} and abs(p['plaq_im']) < 1e-12
    assert abs(p['plaq_re'] - reference_average(U, 1, 1).real) < 1e-12
    assert abs(p['action'] - 2.1 * 3 * (1 - p['plaq_re'])) < 1e-9                       # 3 planes per site in 3 dimensions
    o = loop_observables(field, 1)
    assert abs(p['plaq_re'] - (1 * o['plaq_s'] + 2 * o['plaq_t']) / 3) < 1e-12           # 1 spatial plane, 2 with the time
    assert abs(field.Wilson(1, 1) - field.Plaq()) < 1e-12 and abs(field.Wilson(1, 2) - field.Wilson(2, 1)) < 1e-12
    assert set(o) == {'plaq_s', 'plaq_t', 'polyakov', 'polyakov_abs'}                    # no loops beyond the plaquette


def test_a_cold_configuration_has_every_loop_equal_to_one():
    cc, loc = setup([8, 4, 4])
    field = SU2Field(grid=loc, cartesiancomm=cc, Nd=3)
    o = loop_observables(field, 3)
    assert all(abs(v - 1) < 1e-14 for k, v in o.items() if not k.startswith('creutz'))
    assert all(abs(v) < 1e-14 for k, v in o.items() if k.startswith('creutz'))           # -ln 1
    p = plaquette_observables(field, 2.3)
    assert p == {'plaq_re': 1.0, 'plaq_im': 0.0, 'action': 0.0}


def test_every_observable_is_gauge_invariant():
    cc, field, U = global_field([8, 4, 4])
    moved = field.gauge_transform()
    for before, after in ((loop_observables(field, 3), loop_observables(moved, 3)),
                          (plaquette_observables(field, 2.0), plaquette_observables(moved, 2.0))):
        for key in before:
            assert abs(before[key] - after[key]) < 1e-11, key


def test_sizes_that_do_not_fit_are_refused():
    cc, field, U = global_field([8, 4])
    for max_loop in (0, 5, 10):
        with pytest.raises(ValueError, match='loop'):
            loop_observables(field, max_loop)
    with pytest.raises(ValueError, match='at least 1'):
        field.line(0, 0)
    with pytest.raises(ValueError, match='different'):
        field.loop_field(1, 1, 1, 1)
    with pytest.raises(ValueError, match='does not fit'):
        field.loop_field(0, 1, 9, 1)                                                    # 9 > 8 sites along the direction 0
    with pytest.raises(ValueError, match='does not fit'):
        field.Wilson(1, 5)
    assert field.loop_field(0, 1, 8, 4) is not None                                     # the whole lattice: fits


# ----------------------------------------------------------------------------- stored configurations
@needs_hdf5
def test_stored_configurations_are_measured_in_the_order_of_their_steps():
    from pylattice.core.callbacks import Checkpointer
    from pylattice.core.measure import measure_files
    cc, field, U = global_field([8, 4, 4])
    other = SU2Field(grid=field.grid, cartesiancomm=cc, Nd=3)                            # a cold configuration
    with shared_tempdir() as d:
        configs = {5: other, 2: field}                                                   # saved out of order
        paths = [Checkpointer(1, directory=d, prefix='c').save(SUNModel(f, None, beta=2.0), step)
                 for step, f in configs.items()]
        out = os.path.join(d, 'meas.json')
        measure_files(paths, lambda p: measure_su(p, cc, field.grid, 2.0, 2, 2), COMM, filename=out)
        COMM.Barrier()
        records = read(out)
        assert [r['sweep'] for r in records] == [2, 5] and records[0]['acceptance'] is None
        want = {**plaquette_observables(field, 2.0), **loop_observables(field, 2)}
        assert set(records[0]) == set(want) | {'sweep', 'acceptance'}
        for key, value in want.items():
            assert records[0][key] == pytest.approx(value, abs=1e-12), key               # what is stored is what is measured
        assert records[1]['plaq_re'] == 1.0 and records[1]['W22'] == 1.0 and records[1]['creutz22'] == 0.0


@needs_hdf5
def test_offline_measurements_equal_the_ones_made_during_the_run():
    from pylattice.core.callbacks import Checkpointer, MeasurementFile
    from pylattice.core.measure import measure_files
    cc, loc = setup([8, 4])
    rng = cc.rng(7)
    field = SU2Field(grid=loc, cartesiancomm=cc, Nd=2, rng=rng)
    field.random_config(rng)
    model = SUNModel(field, None, beta=2.0, eps=0.5, rng=rng, algorithm='heatbath', overrelax=1)
    from pylattice.core import montecarlo
    with shared_tempdir() as d:
        online = os.path.join(d, 'online.json')
        montecarlo.run(model, 3, progress=False, callbacks=[Checkpointer(1, directory=d, prefix='c'), MeasurementFile(online)])
        COMM.Barrier()
        offline = os.path.join(d, 'offline.json')
        measure_files([os.path.join(d, 'c_{:08d}.hdf5'.format(k)) for k in (1, 2, 3)],
                      lambda p: measure_su(p, cc, loc, 2.0, 2, 1), COMM, filename=offline)
        COMM.Barrier()
        a, b = read(online)[1:], read(offline)                                           # sweep 0 (no checkpoint) left out
        assert [r['sweep'] for r in a] == [r['sweep'] for r in b] == [1, 2, 3]
        for x, y in zip(a, b):
            for key in ('plaq_re', 'plaq_im', 'action'):
                assert y[key] == pytest.approx(x[key], abs=1e-12), key
            assert 'polyakov' in y and 'polyakov' not in x                               # only offline


@needs_hdf5
def test_a_file_that_is_not_an_su2_configuration_stops_the_run_and_keeps_what_was_measured():
    from pylattice.core.callbacks import Checkpointer
    from pylattice.core.hdf5io import Writer
    from pylattice.core.measure import measure_files
    cc, field, U = global_field([8, 4])
    bad = SU2Field(grid=field.grid, cartesiancomm=cc, Nd=2)
    bad.value = bad.value * 1.5                                                          # not unitary
    with shared_tempdir() as d:
        p1 = Checkpointer(1, directory=d, prefix='c').save(SUNModel(field, None, beta=2.0), 1)
        p2 = os.path.join(d, 'bad.hdf5')
        Writer(lattice=bad, cartesiancomm=cc).write(p2)
        out = os.path.join(d, 'meas.json')
        with pytest.raises(ValueError, match='not an SU\\(2\\) configuration'):
            measure_files([p1, p2], lambda p: measure_su(p, cc, field.grid, 2.0, 2, 2), COMM, filename=out)
        COMM.Barrier()
        assert [r['sweep'] for r in read(out)] == [1]                                    # valid JSON, the good one is in


@needs_hdf5
def test_command_line_measures_the_stored_configurations():
    from pylattice.core.callbacks import Checkpointer
    cc, field, U = global_field([8, 4, 4])
    with shared_tempdir() as d:
        paths = [Checkpointer(1, directory=d, prefix='c').save(SUNModel(field, None, beta=2.0), s) for s in (4, 8)]
        out = os.path.join(d, 'm.json')
        base = paths + ['--beta=2.0', '--out=' + out, '--mpigrid=' + 'x'.join(map(str, cc.mpigrid))]
        main(base + ['--max-loop=3'])
        COMM.Barrier()
        records = read(out)
        assert [r['sweep'] for r in records] == [4, 8]
        assert {'plaq_s', 'plaq_t', 'polyakov', 'W12', 'W13', 'W22', 'W23', 'W33', 'creutz22', 'creutz33'} <= set(records[0])
        assert records[0]['W33'] == pytest.approx(reference_average(U, 3, 3).real, abs=1e-12)
        for argv, code in (([paths[0], '--out=' + out], 2),                                   # --beta is required
                           (base + ['--max-loop=5'], 1),                                       # a 5x5 loop does not fit (4 sites)
                           (base + ['--max-loop=0'], 1),
                           ([paths[0], '--beta=-1'], 2),
                           (paths + ['--beta=2.0', '--out=' + out, '--mpigrid=3x1x1'], 1)):   # does not fit 1, 2 or 4 ranks
            with pytest.raises(SystemExit) as exc:
                main(argv)
            assert exc.value.code == code, argv
            COMM.Barrier()
        assert [r['sweep'] for r in read(out)] == [4, 8]                                       # the good output is untouched


if __name__ == '__main__':
    import traceback
    tests = [(name, f) for name, f in sorted(globals().items()) if name.startswith('test_') and callable(f)]
    for name, f in tests:
        if getattr(f, 'needs_hdf5', False) and not parallel_hdf5():
            if COMM.Get_rank() == 0:
                print('[-] {} skipped: no h5py with parallel HDF5'.format(name))
            continue
        marks = getattr(f, 'pytestmark', None)
        cases = [()]
        if marks and any(m.name == 'parametrize' for m in marks):
            cases = list(next(m.args[1] for m in marks if m.name == 'parametrize'))
        try:
            for case in cases:
                f(*case)
        except BaseException:
            # a rank that fails must stop them all: the others would wait for it in a collective call
            print('[x] {} failed on rank {}'.format(name, COMM.Get_rank()), flush=True)
            traceback.print_exc()
            COMM.Abort(1)
        if COMM.Get_rank() == 0:
            print('[+] {} passed on {} rank(s)'.format(name, COMM.Get_size()))
