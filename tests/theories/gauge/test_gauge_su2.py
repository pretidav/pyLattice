"""Tests of pylattice.theories.gauge.su2: the Metropolis sweep of the SU(2) gauge theory with the Wilson action, its
observables, the restart and the command line driver.

The change of the action that the sweep uses is compared with the change of the whole action, and the sampling with
the exact heat bath and with the limits of strong and weak coupling. They run on one process or on a lattice split
over 2 or 4 ranks:

    pytest tests/theories/gauge/test_gauge_su2.py
    mpiexec -n 2 python tests/theories/gauge/test_gauge_su2.py        # 2 ranks (1 or 4 also work)
"""
import contextlib
import io
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
from pylattice.theories.gauge.su2 import SU2Model, SU2Parser, load, main
from pylattice.theories.gauge.sun import SU2Field

I2 = np.eye(2)


# ----------------------------------------------------------------------------- helpers
def mpigrid_for(dims):
    size = MPI.COMM_WORLD.Get_size()
    assert size in (1, 2, 4), 'run with 1, 2 or 4 ranks'
    return {1: [1] * dims, 2: [2] + [1] * (dims - 1), 4: [2, 2] + [1] * (dims - 2)}[size]


def setup(G):
    """communicator and local grid for a global lattice G (every local size must be even)"""
    mg = mpigrid_for(len(G))
    return CartesianComm(mpigrid=mg), [g // m for g, m in zip(G, mg)]


def make(G, beta=2.0, eps=0.5, hits=1, seed=1, hot=True, algorithm='metropolis', overrelax=0):
    cc, loc = setup(list(G))
    rng = cc.rng(seed)
    field = SU2Field(grid=loc, cartesiancomm=cc, Nd=len(G), rng=rng)
    if hot:
        field.random_config(rng)
    return cc, field, SU2Model(field=field, logger=None, beta=beta, eps=eps, hits=hits, rng=rng,
                               algorithm=algorithm, overrelax=overrelax)


def assert_su2(U, tol=1e-12):
    assert np.abs(U @ np.conj(np.swapaxes(U, -1, -2)) - I2).max() < tol and np.abs(np.linalg.det(U) - 1).max() < tol


def total(cc, x):
    return cc.comm.allreduce(float(x))


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
        yield out


@contextlib.contextmanager
def shared_tempdir():
    comm = MPI.COMM_WORLD
    path = comm.bcast(tempfile.mkdtemp() if comm.Get_rank() == 0 else None, root=0)
    old = os.getcwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(old)
        for handler in list(logging.getLogger('pylattice').handlers):
            logging.getLogger('pylattice').removeHandler(handler)
        comm.Barrier()
        if comm.Get_rank() == 0:
            shutil.rmtree(path, ignore_errors=True)


def parallel_hdf5():
    try:
        import h5py
        return bool(h5py.get_config().mpi)
    except ImportError:
        return False


def needs_hdf5(f):
    f.needs_hdf5 = True                                  # the script runner below skips it too
    return pytest.mark.skipif(not parallel_hdf5(), reason='h5py with parallel HDF5 is not installed')(f)


# ----------------------------------------------------------------------------- the change of the action
@pytest.mark.parametrize('G', [[8, 4], [4, 4, 4]])
def test_the_action_change_of_a_link_update_is_that_of_the_whole_action(G):
    # all the links of one direction on the sites of one parity are replaced at once: they share no plaquette, so
    # the change of the whole action is the sum of the changes each one reports
    beta = 1.7
    cc, field, model = make(G, beta=beta, eps=0.8)
    rng = cc.rng(5)
    for mu in range(len(G)):
        for parity in (model.even, model.odd):
            staple = field.Staple(mu)
            link = field.peek_index(mu)
            proposed = model.proposal(link, rng)
            change = model.link_action_change(link, proposed, staple).value
            before = field.Action(beta)
            kept = field.value[:, mu].copy()
            link.value = np.where(parity[:, None, None], proposed.value, link.value)
            field.poke_index(mu, link)
            after = field.Action(beta)
            expected = total(cc, change[parity].sum())
            assert abs((after - before) - expected) < 1e-9, (mu, after - before, expected)
            assert abs(expected) > 1e-3                                              # a real change, not zero against zero
            field.value[:, mu] = kept


def test_updating_the_two_parities_together_would_not_be_independent():
    # the reason for the checkerboard: neighbouring links of one direction share a plaquette, so replacing all of
    # them at once is NOT the sum of the single changes (this is what the sweep must not do)
    beta = 1.7
    cc, field, model = make([8, 4], beta=beta, eps=0.8)
    mu = 0
    staple, link = field.Staple(mu), field.peek_index(mu)
    proposed = model.proposal(link, cc.rng(6))
    summed = total(cc, model.link_action_change(link, proposed, staple).value.sum())
    before = field.Action(beta)
    field.poke_index(mu, proposed)
    assert abs((field.Action(beta) - before) - summed) > 1e-3


# ----------------------------------------------------------------------------- the sweep
def test_a_sweep_keeps_every_link_in_su2_and_counts_its_proposals():
    cc, field, model = make([8, 4], hits=2)
    sites = int(np.prod(field.grid))
    for _ in range(5):
        accepted, attempted = model.sweep()
        assert attempted == 2 * sites * 2 and 0 <= accepted <= attempted            # Nd sites hits
        assert_su2(field.value)
    acceptance = total(cc, accepted) / total(cc, attempted)
    assert 0.2 < acceptance < 0.9


def test_a_small_proposal_is_almost_always_accepted_and_a_large_one_much_less():
    for eps, low, high in ((0.02, 0.97, 1.0), (1.0, 0.0, 0.3)):
        cc, field, model = make([8, 4], beta=2.3, eps=eps)
        for _ in range(10):
            model.sweep()                                                            # thermalize
        a = t = 0
        for _ in range(5):
            x, y = model.sweep()
            a, t = a + x, t + y
        assert low <= total(cc, a) / total(cc, t) <= high, eps


def test_at_beta_zero_every_proposal_is_accepted_and_the_links_become_random():
    cc, field, model = make([8, 4, 4], beta=0.0, eps=0.9, hot=False)
    accepted = attempted = 0
    for _ in range(40):
        a, t = model.sweep()
        accepted, attempted = accepted + a, attempted + t
    assert accepted == attempted                                                     # dS = 0
    plaquette = model.observables()['plaq_re']
    assert abs(plaquette) < 0.1 and plaquette != 1.0                                 # from the cold start 1 to about 0
    assert model.observables()['action'] == 0.0                                      # the action is beta times something


def test_the_same_seed_gives_the_same_sweeps_and_another_seed_another():
    def run(seed):
        cc, field, model = make([8, 4], seed=seed)
        for _ in range(3):
            model.sweep()
        return field.value.copy()
    assert np.array_equal(run(3), run(3)) and not np.array_equal(run(3), run(4))


def test_observables_of_the_cold_start_and_the_identities_between_them():
    cc, field, model = make([8, 4, 4], beta=2.3, hot=False)
    o = model.observables()
    assert o['plaq_re'] == 1.0 and o['plaq_im'] == 0.0 and o['action'] == 0.0
    for _ in range(5):
        model.sweep()
    o = model.observables()
    assert set(o) == {'plaq_re', 'plaq_im', 'action'}                                # the loops are measured offline
    assert all(type(v) is float for v in o.values())
    assert abs(o['plaq_im']) < 1e-12                                                 # the imaginary part is rounding
    planes = 3 * 2 // 2
    assert abs(o['action'] - 2.3 * planes * (1 - o['plaq_re'])) < 1e-9               # the action per site, from the plaquette
    assert 0 < o['plaq_re'] < 1 and o['action'] > 0


def test_the_model_checks_its_arguments():
    cc, loc = setup([8, 4])
    field = SU2Field(grid=loc, cartesiancomm=cc, Nd=2)
    for kwargs in (dict(beta=float('nan')), dict(beta=-1.0), dict(beta=1.0, eps=0.0), dict(beta=1.0, eps=1.5),
                   dict(beta=1.0, hits=0), dict(beta=1.0, hits=1.5), dict(beta=1.0, algorithm='gibbs'),
                   dict(beta=1.0, overrelax=-1), dict(beta=1.0, overrelax=1.5)):
        with pytest.raises(ValueError):
            SU2Model(field=field, logger=None, **kwargs)
    odd = SU2Field(grid=[3, 4], cartesiancomm=CartesianComm(mpigrid=[1, 1]), Nd=2) if MPI.COMM_WORLD.Get_size() == 1 else None
    if odd is not None:
        with pytest.raises(ValueError, match='checkerboard'):
            SU2Model(field=odd, logger=None, beta=1.0)
        with pytest.raises(ValueError, match='at least 2'):
            SU2Model(field=SU2Field(grid=[4], cartesiancomm=CartesianComm(mpigrid=[1]), Nd=1), logger=None, beta=1.0)


# ----------------------------------------------------------------------------- the sweep, exactly
def test_a_sweep_changes_the_action_by_the_sum_of_the_accepted_changes():
    # no statistics: over one sweep the action of the whole lattice changes by exactly the sum of the changes of the
    # links that were accepted. This holds only if the links updated together share no plaquette (the checkerboard),
    # and with the right staples, sign and factor in the change of the action
    from pylattice.core import montecarlo
    beta = 1.9
    for G, hits in (([8, 4], 1), ([4, 4, 4], 2)):
        cc, field, model = make(G, beta=beta, eps=0.7, hits=hits)
        seen, original = [], model.link_action_change
        accept_of = montecarlo.metropolis_accept

        def spy_change(link, proposed, staple):
            change = original(link, proposed, staple)
            seen.append([change.value.copy(), None])
            return change

        def spy_accept(delta, rng):
            accept = accept_of(delta, rng)
            seen[-1][1] = accept
            return accept

        model.link_action_change = spy_change
        montecarlo.metropolis_accept = spy_accept
        try:
            before = field.Action(beta)
            accepted, attempted = model.sweep()
            after = field.Action(beta)
        finally:
            montecarlo.metropolis_accept = accept_of
        assert len(seen) == len(G) * 2 * hits                                         # directions, parities, hits
        parities = [model.even, model.odd] * (len(G))
        expected = 0.0
        for k, (change, accept) in enumerate(seen):
            expected += change[accept & parities[k // hits]].sum()
        assert accepted == sum(int((a & parities[k // hits]).sum()) for k, (c, a) in enumerate(seen))
        assert 0 < total(cc, accepted) < total(cc, attempted)
        assert abs((after - before) - total(cc, expected)) < 1e-8, (G, after - before, total(cc, expected))


def test_a_cold_start_at_large_beta_stays_ordered_and_a_hot_start_at_beta_zero_stays_disordered():
    cc, field, model = make([4, 4], beta=30.0, eps=0.1, hot=False)
    for _ in range(3):
        model.sweep()
    assert model.observables()['plaq_re'] > 0.9
    cc, field, model = make([4, 4, 4], beta=0.0, eps=0.9, hot=True)
    for _ in range(3):
        model.sweep()
    assert abs(model.observables()['plaq_re']) < 0.2


# ----------------------------------------------------------------------------- heat bath and overrelaxation
def test_overrelaxation_leaves_the_action_unchanged_and_is_its_own_inverse():
    for G in ([8, 4], [4, 4, 4]):
        cc, field, model = make(G, beta=1.8)
        before_links, before = field.value.copy(), field.Action(1.8)
        for mu in range(len(G)):
            for parity in (model.even, model.odd):
                once = field.value.copy()
                model.overrelaxation_update(mu, parity)
                assert abs(field.Action(1.8) - before) < 1e-9                        # the action does not change
                assert_su2(field.value, 1e-11)
                changed = np.abs(field.value[:, mu] - once[:, mu]).max(axis=(-1, -2)) > 1e-8
                assert changed[parity].mean() > 0.9 and not changed[~parity].any()   # moves the links of its parity only
                other = [d for d in range(len(G)) if d != mu]
                assert all(np.array_equal(field.value[:, d], once[:, d]) for d in other)
                model.overrelaxation_update(mu, parity)                              # twice: the same links again
                assert np.abs(field.value - once).max() < 1e-12
                model.overrelaxation_update(mu, parity)                              # go on from the changed one
        assert np.abs(field.value - before_links).max() > 1e-3                       # the sweep as a whole did move them


def test_heat_bath_updates_only_its_links_counts_them_and_aligns_them_with_the_staple_at_large_beta():
    cc, field, model = make([8, 4, 4], beta=40.0, algorithm='heatbath', hot=True)
    for mu in range(3):
        for parity in (model.even, model.odd):
            before = field.value.copy()
            staple = field.Staple(mu).value
            assert model.heatbath_update(mu, parity, cc.rng(1)) == int(parity.sum())
            assert_su2(field.value)
            assert np.array_equal(field.value[~parity], before[~parity])               # the other sites keep their links
            k, W = group.su2_decompose(staple)
            alignment = 0.5 * np.einsum('sij,sji->s', field.value[:, mu], W).real      # (1/2) Re tr(U W): 1 = aligned
            assert total(cc, alignment[parity].sum()) / total(cc, parity.sum()) > 0.9  # exp((beta/2) k Re tr(UW)), beta k ~ 100


def test_every_algorithm_sweeps_keep_su2_count_and_reproduce():
    for algorithm, overrelax in (('metropolis', 0), ('metropolis', 2), ('heatbath', 0), ('heatbath', 3)):
        def run(seed):
            cc, field, model = make([8, 4], beta=2.0, algorithm=algorithm, overrelax=overrelax, seed=seed)
            counts = [model.sweep() for _ in range(2)]
            assert_su2(field.value)
            return field.value.copy(), counts, int(np.prod(field.grid))
        links, counts, sites = run(3)
        assert np.array_equal(links, run(3)[0]) and not np.array_equal(links, run(4)[0])
        for accepted, attempted in counts:
            assert attempted == 2 * sites                                              # Nd sites, one proposal per link
            assert (accepted == attempted) if algorithm == 'heatbath' else (0 < accepted < attempted)
    cc, field, model = make([4, 4], beta=0.0, algorithm='heatbath')                    # beta = 0: the Haar measure
    model.sweep()
    assert_su2(field.value)


# ----------------------------------------------------------------------------- restart
@needs_hdf5
def test_load_gives_back_the_saved_links_and_refuses_what_is_not_su2():
    from pylattice.core.hdf5io import Writer
    cc, field, _ = make([8, 4])
    with shared_tempdir() as d:
        path = os.path.join(d, 'conf.hdf5')
        Writer(lattice=field, cartesiancomm=cc).write(path)
        new, lines = SU2Field(grid=field.grid, cartesiancomm=cc, Nd=2), []
        load(new, path, log=lines.append)
        assert np.array_equal(new.value, field.value) and new.value.dtype == np.complex128
        assert 'SU(2)' in '\n'.join(lines)
        bad = SU2Field(grid=field.grid, cartesiancomm=cc, Nd=2)
        bad.value = bad.value * 1.5                                                  # unitarity broken
        Writer(lattice=bad, cartesiancomm=cc).write(os.path.join(d, 'bad.hdf5'))
        target = SU2Field(grid=field.grid, cartesiancomm=cc, Nd=2)
        with pytest.raises(ValueError, match='not an SU\\(2\\) configuration'):
            load(target, os.path.join(d, 'bad.hdf5'))
        assert np.array_equal(target.value, SU2Field(grid=field.grid, cartesiancomm=cc, Nd=2).value)   # untouched


@needs_hdf5
def test_a_restart_continues_the_run_exactly():
    import h5py
    cc, loc = setup([8, 4])
    base = ['--grid=' + 'x'.join(map(str, loc)), '--mpigrid=' + 'x'.join(map(str, cc.mpigrid)), '--beta=2.0',
            '--eps=0.5', '--seed=4', '--init=hot', '--logfile=off']

    def last(directory):
        path = os.path.join(directory, sorted(os.listdir(directory))[-1])
        with h5py.File(path, 'r') as f:
            return f['value'][...]

    with shared_tempdir() as d:
        if cc.rank == 0:                                                              # one rank makes the directories
            for name in ('straight', 'split', 'second'):
                os.makedirs(os.path.join(d, name))
        MPI.COMM_WORLD.Barrier()
        os.chdir(os.path.join(d, 'straight'))
        with quiet():
            main(base + ['--steps=6', '--checkpoint-every=6'])
        MPI.COMM_WORLD.Barrier()
        straight = last('checkpoints') if cc.rank == 0 else None
        os.chdir(os.path.join(d, 'split'))
        with quiet():
            main(base + ['--steps=3', '--checkpoint-every=3'])
        MPI.COMM_WORLD.Barrier()
        first = os.path.abspath(os.path.join('checkpoints', 'su2_00000003.hdf5'))
        os.chdir(d)
        os.chdir(os.path.join(d, 'second'))
        with quiet():
            main(base + ['--steps=3', '--checkpoint-every=3', '--start-from=' + first])
        MPI.COMM_WORLD.Barrier()
        if cc.rank == 0:
            assert np.array_equal(last('checkpoints'), straight)                    # bit for bit


# ----------------------------------------------------------------------------- command line
def test_parser_defaults_options_and_errors():
    P = SU2Parser([])
    assert (P.grid, P.mpigrid, P.beta, P.eps, P.hits, P.init) == ([4, 4, 4, 4], [1, 1, 1, 2], 2.3, 0.4, 1, 'cold')
    assert (P.algorithm, P.overrelax, P.thermalization, P.measure_every) == ('metropolis', 0, 0, 1)
    P = SU2Parser(['--grid=6x6', '--mpigrid=1x1', '--beta=1.1', '--eps=0.25', '--hits=3', '--init=hot'])
    assert (P.grid, P.beta, P.eps, P.hits, P.init) == ([6, 6], 1.1, 0.25, 3, 'hot')
    assert dict(P.extra_info()) == {'algorithm': 'metropolis', 'overrelaxation sweeps': 0, 'proposal eps': 0.25,
                                    'hits per link': 3, 'init': 'hot'}
    Q = SU2Parser(['--algorithm=heatbath', '--overrelax=4', '--thermalization=10', '--measure-every=5'])
    assert (Q.algorithm, Q.overrelax, Q.thermalization, Q.measure_every) == ('heatbath', 4, 10, 5)
    assert dict(Q.extra_info()) == {'algorithm': 'heatbath', 'overrelaxation sweeps': 4, 'init': 'cold'}   # no eps for the heat bath
    for argv, option in ((['--eps=0'], '--eps'), (['--eps=1.5'], '--eps'), (['--eps=x'], '--eps'), (['--hits=0'], '--hits'),
                         (['--init=warm'], '--init'), (['--algorithm=gibbs'], '--algorithm'), (['--overrelax=-1'], '--overrelax'),
                         (['--thermalization=-1'], '--thermalization'), (['--measure-every=0'], '--measure-every'), (['--grid=4x4', '--mpigrid=1x1x2'], '--grid')):
        with pytest.raises(SystemExit) as exc:
            SU2Parser(argv)
        assert exc.value.code == 2


def test_command_line_driver_writes_the_measurements_the_log_and_the_summary():
    cc, loc = setup([4, 4])
    argv = ['--grid=' + 'x'.join(map(str, loc)), '--mpigrid=' + 'x'.join(map(str, cc.mpigrid)), '--steps=6', '--beta=2.0',
            '--seed=3', '--runlog-every=3']
    with shared_tempdir():
        with quiet() as out:
            main(argv)
        if cc.rank == 0:
            records = json.load(open('meas.json'))['measurements']
            assert [r['sweep'] for r in records] == list(range(7))
            assert set(records[1]) == {'sweep', 'plaq_re', 'plaq_im', 'action', 'acceptance'}
            assert records[0]['acceptance'] is None and all(0 < r['acceptance'] < 1 for r in records[1:])
            assert all(abs(r['plaq_im']) < 1e-12 for r in records)
            content = open('run.log', encoding='utf-8').read()
            lines = [l for l in content.splitlines() if l.startswith('step=')]
            assert [l.split()[0] for l in lines] == ['step=3', 'step=6'] and 'plaq_re=' in lines[0]
            assert 'proposal eps' in content and '╭─ done' in content
            shown = out.getvalue()
            assert '╭─ system' in shown and 'step=3' in shown


if __name__ == '__main__':
    import traceback
    tests = [(name, f) for name, f in sorted(globals().items()) if name.startswith('test_') and callable(f)]
    for name, f in tests:
        if getattr(f, 'needs_hdf5', False) and not parallel_hdf5():
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('[-] {} skipped: no h5py with parallel HDF5'.format(name))
            continue
        params = getattr(f, 'pytestmark', None)
        cases = [()]
        if params and any(m.name == 'parametrize' for m in params):
            cases = [(G,) for G in next(m.args[1] for m in params if m.name == 'parametrize')]
        try:
            for case in cases:
                f(*case)
        except BaseException:
            # a rank that fails must stop them all: the others would wait for it in a collective call
            print('[x] {} failed on rank {}'.format(name, MPI.COMM_WORLD.Get_rank()), flush=True)
            traceback.print_exc()
            MPI.COMM_WORLD.Abort(1)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('[+] {} passed on {} rank(s)'.format(name, MPI.COMM_WORLD.Get_size()))
