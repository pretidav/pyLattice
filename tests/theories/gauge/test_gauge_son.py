"""Tests of pylattice.theories.gauge.son: the SO(N) gauge theory with the Wilson action and the Cabibbo-Marinari sweeps over the
rotation planes (heat bath, Metropolis, overrelaxation), its observables, restart, command line and the offline measurement of its
checkpoints. N = 3 mostly, and N = 2 (SO(2) = U(1): one plane, the exact heat bath).

Nothing runs a real Monte Carlo: the change of the action of every update is compared with the change of the whole action, and the
sweeps are checked on a few sweeps of tiny lattices. They run on one process or on a lattice split over 2 or 4 ranks:

    pytest tests/theories/gauge/test_gauge_son.py
    mpiexec -n 2 python tests/theories/gauge/test_gauge_son.py        # 2 ranks (1 or 4 also work)
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

from pylattice.core import montecarlo
from pylattice.core.groups import so
from pylattice.core.parallel import CartesianComm
from pylattice.theories.gauge.field import SONField, gauge_field, load_links
from pylattice.theories.gauge.son import SONModel, SONParser, main
from pylattice.theories.gauge.sun import plaquette_observables

dag = lambda M: np.conj(np.swapaxes(M, -1, -2))


# ----------------------------------------------------------------------------- helpers
def mpigrid_for(dims):
    size = MPI.COMM_WORLD.Get_size()
    assert size in (1, 2, 4), 'run with 1, 2 or 4 ranks'
    return {1: [1] * dims, 2: [2] + [1] * (dims - 1), 4: [2, 2] + [1] * (dims - 2)}[size]


def setup(G):
    """communicator and local grid for a global lattice G (every local size must be even)"""
    mg = mpigrid_for(len(G))
    return CartesianComm(mpigrid=mg), [g // m for g, m in zip(G, mg)]


def make(G, N=3, beta=5.5, eps=0.4, hits=1, seed=1, hot=True, algorithm='heatbath', overrelax=0):
    cc, loc = setup(list(G))
    rng = cc.rng(seed)
    field = gauge_field(loc, cc, len(G), N, rng, group='so')
    if hot:
        field.random_config(rng)
    return cc, field, SONModel(field=field, logger=None, beta=beta, eps=eps, hits=hits, rng=rng, algorithm=algorithm,
                               overrelax=overrelax)


def special_orthogonal(U, tol=1e-12):
    N = U.shape[-1]
    return np.abs(U @ dag(U) - np.eye(N)).max() < tol and np.abs(np.linalg.det(U) - 1).max() < tol and not np.iscomplexobj(U)


def run_name(N, loc, cc, beta):
    """the start of the name of every file of a run: <theory>_grid<grid>_mpi<mpigrid>_b<beta>"""
    return 'so{}_grid{}_mpi{}_b{}'.format(N, 'x'.join(map(str, loc)), 'x'.join(map(str, cc.mpigrid)), beta)


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
            shutil.rmtree(path, ignore_errors=True)               # the directory this test made with mkdtemp


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
@pytest.mark.parametrize('G,N', [([8, 4], 3), ([4, 4, 4], 3), ([8, 4], 4), ([8, 4], 2)])
def test_the_action_change_of_a_link_update_is_that_of_the_whole_action(G, N):
    # all the links of one direction on the sites of one parity are replaced at once: they share no plaquette, so the change of the
    # whole action is the sum of the changes each one reports (with the factor beta / N of the link weight)
    beta = 4.3
    cc, field, model = make(G, N=N, beta=beta, eps=0.8)
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
            expected = total(cc, change[parity].sum())
            assert abs((field.Action(beta) - before) - expected) < 1e-9, (mu, N)
            assert abs(expected) > 1e-3                                              # a real change, not zero against zero
            field.value[:, mu] = kept


def test_a_metropolis_sweep_changes_the_action_by_the_sum_of_the_accepted_changes():
    beta = 5.0
    for G, hits, N in (([8, 4], 1, 3), ([4, 4, 4], 2, 3), ([8, 4], 1, 2)):
        cc, field, model = make(G, N=N, beta=beta, eps=0.5, hits=hits, algorithm='metropolis')
        seen, original, accept_of = [], model.link_action_change, montecarlo.metropolis_accept

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
        assert len(seen) == len(G) * 2 * hits
        parities = [model.even, model.odd] * len(G)
        expected = sum(change[accept & parities[k // hits]].sum() for k, (change, accept) in enumerate(seen))
        assert 0 < total(cc, accepted) < total(cc, attempted)
        assert abs((after - before) - total(cc, expected)) < 1e-8, (G, after - before, total(cc, expected))


# ----------------------------------------------------------------------------- overrelaxation and heat bath
@pytest.mark.parametrize('G,N', [([8, 4], 3), ([4, 4, 4], 3), ([8, 4], 4), ([8, 4], 2)])
def test_overrelaxation_leaves_the_action_unchanged_and_moves_only_its_links(G, N):
    cc, field, model = make(G, N=N, beta=3.1)
    before = field.Action(3.1)
    for mu in range(len(G)):
        for parity in (model.even, model.odd):
            once = field.value.copy()
            model.overrelaxation_update(mu, parity)
            assert abs(field.Action(3.1) - before) < 1e-9                             # the action does not change
            assert special_orthogonal(field.value, 1e-11)
            changed = np.abs(field.value[:, mu] - once[:, mu]).max(axis=(-1, -2)) > 1e-8
            assert changed[parity].mean() > 0.9 and not changed[~parity].any()        # the links of its parity only
            assert all(np.array_equal(field.value[:, d], once[:, d]) for d in range(len(G)) if d != mu)


def test_the_so2_overrelaxation_of_the_model_is_its_own_inverse_and_the_plane_step_of_so():
    # N = 2 has one plane, so applying the update twice gives the links back (for N >= 3 each plane step is its own inverse, not the sequence)
    cc, field, model = make([8, 4], N=2, beta=1.0)
    for mu in range(2):
        for parity in (model.even, model.odd):
            once = field.value.copy()
            staple = field.Staple(mu).value
            model.overrelaxation_update(mu, parity)
            assert np.abs(field.value - once).max() > 1e-3
            assert np.abs(field.value[:, mu][parity] - so.cabibbo_marinari_overrelax_link(once[:, mu], staple)[parity]).max() < 1e-12
            model.overrelaxation_update(mu, parity)
            assert np.abs(field.value - once).max() < 1e-12
            model.overrelaxation_update(mu, parity)


@pytest.mark.parametrize('N', [2, 3, 4])
def test_heat_bath_updates_only_its_links_counts_them_and_aligns_them_with_the_staple_at_large_beta(N):
    cc, field, model = make([8, 4, 4], N=N, beta=100.0 * N)                         # beta / N = 100: sharp conditional weight
    for mu in range(3):
        for parity in (model.even, model.odd):
            before = field.value.copy()
            staple = field.Staple(mu).value
            assert model.heatbath_update(mu, parity, cc.rng(1)) == int(parity.sum())
            assert special_orthogonal(field.value)
            assert np.array_equal(field.value[~parity], before[~parity])           # the other sites keep their links
            s = np.linalg.svd(staple, compute_uv=False)
            best = s.sum(axis=-1) - 2 * s.min(axis=-1) * (np.linalg.det(staple) < 0)        # max over SO(N) of tr(U S)
            alignment = np.trace(field.value[:, mu] @ staple, axis1=-2, axis2=-1) / best
            assert total(cc, alignment[parity].sum()) / total(cc, parity.sum()) > 0.8     # a random link gives about 0


@pytest.mark.parametrize('N', [2, 3, 4])
def test_the_heat_bath_uses_the_weight_exp_beta_over_n_tr_u_s_of_the_action(N):
    # the link weight from S = beta sum (1 - tr P / N) is exp((beta / N) tr(U S)) (the action-change test above fixes this factor): the
    # coupling given to the Cabibbo-Marinari step must be beta / N, and the staple the field's
    from pylattice.theories.gauge import son as theory
    beta = 4.6
    cc, field, model = make([8, 4], N=N, beta=beta)
    seen, original = [], theory.so.cabibbo_marinari_link

    def spy(U, V, c, rng=None, subgroups=None):
        seen.append((c, V.copy()))
        return original(U, V, c, rng, subgroups)

    theory.so.cabibbo_marinari_link = spy
    try:
        staple = field.Staple(0).value.copy()
        model.heatbath_update(0, model.even, cc.rng(2))
    finally:
        theory.so.cabibbo_marinari_link = original
    assert len(seen) == 1 and seen[0][0] == pytest.approx(beta / N) and np.array_equal(seen[0][1], staple)


@pytest.mark.parametrize('N,eps', [(2, 0.3), (3, 0.3), (4, 0.7), (3, 1.0)])
def test_the_metropolis_proposal_is_at_the_distance_eps_from_the_identity(N, eps):
    cc, field, model = make([8, 4], N=N, eps=eps, hot=False)                         # cold: every link is the identity
    X = model.proposal(field.peek_index(0), cc.rng(3)).value                          # so the proposal is X itself
    assert special_orthogonal(X) and X.shape == (int(np.prod(field.grid)), N, N)
    assert np.abs(np.linalg.norm(so.log_algebra(X), axis=-1) - 2 * np.arcsin(eps)).max() < 1e-8
    assert np.abs(so.log_algebra(X)).max() > 0


@pytest.mark.parametrize('N,algorithm,overrelax', [(3, 'heatbath', 0), (3, 'heatbath', 2), (3, 'metropolis', 0), (3, 'metropolis', 2),
                                                   (4, 'heatbath', 1), (2, 'heatbath', 0), (2, 'heatbath', 2), (2, 'metropolis', 0),
                                                   (2, 'metropolis', 2)])
def test_every_algorithm_sweeps_keep_so_n_count_and_reproduce(N, algorithm, overrelax):
    def run(seed):
        cc, field, model = make([8, 4], N=N, beta=5.0, algorithm=algorithm, overrelax=overrelax, seed=seed)
        counts = [model.sweep() for _ in range(2)]
        assert special_orthogonal(field.value)
        return field.value.copy(), counts, int(np.prod(field.grid))
    cc = setup([8, 4])[0]
    links, counts, sites = run(3)
    assert np.array_equal(links, run(3)[0]) and not np.array_equal(links, run(4)[0])
    for accepted, attempted in counts:
        assert attempted == 2 * sites                                                  # Nd sites, one proposal per link
        assert (accepted == attempted) if algorithm == 'heatbath' else (0 < total(cc, accepted) < total(cc, attempted))   # all ranks: few links on each
    cc, field, model = make([4, 4], N=N, beta=0.0, algorithm='heatbath')               # beta = 0: the Haar measure
    model.sweep()
    assert special_orthogonal(field.value)


@pytest.mark.parametrize('N', [2, 3])
def test_a_cold_start_at_large_beta_stays_ordered_and_a_hot_start_at_beta_zero_stays_disordered(N):
    cc, field, model = make([4, 4], N=N, beta=60.0, eps=0.1, hot=False, algorithm='metropolis')
    for _ in range(3):
        model.sweep()
    assert model.observables()['plaq_re'] > 0.9
    cc, field, model = make([4, 4, 4], N=N, beta=0.0, hot=True)
    for _ in range(3):
        model.sweep()
    assert abs(model.observables()['plaq_re']) < 0.2


@pytest.mark.parametrize('N', [2, 3])
def test_at_beta_zero_every_metropolis_proposal_is_accepted_and_the_links_become_random(N):
    cc, field, model = make([8, 4, 4], N=N, beta=0.0, eps=0.9, hot=False, algorithm='metropolis')
    accepted = attempted = 0
    for _ in range(20):
        a, t = model.sweep()
        accepted, attempted = accepted + a, attempted + t
    assert accepted == attempted                                                       # dS = 0
    o = model.observables()
    assert abs(o['plaq_re']) < 0.15 and o['plaq_re'] != 1.0 and o['action'] == 0.0


# ----------------------------------------------------------------------------- observables and arguments
@pytest.mark.parametrize('N', [2, 3])
def test_observables_of_the_cold_start_and_their_identities(N):
    cc, field, model = make([8, 4, 4], N=N, beta=5.7, hot=False)
    assert model.observables() == {'plaq_re': 1.0, 'plaq_im': 0.0, 'action': 0.0}
    for _ in range(3):
        model.sweep()
    o = model.observables()
    assert set(o) == {'plaq_re', 'plaq_im', 'action'} and all(type(v) is float for v in o.values())
    assert o['plaq_im'] == 0.0 and 0 < o['plaq_re'] < 1                                 # real links: exactly real
    assert abs(o['action'] - 5.7 * 3 * (1 - o['plaq_re'])) < 1e-9                      # the action per site: 3 planes in 3 dimensions
    assert plaquette_observables(field, 5.7) == o


def test_the_model_checks_its_arguments_and_takes_n_from_the_field():
    cc, loc = setup([8, 4])
    field = gauge_field(loc, cc, 2, 3, group='so')
    for kwargs in (dict(beta=float('nan')), dict(beta=-1.0), dict(beta=1.0, eps=0.0), dict(beta=1.0, eps=1.5),
                   dict(beta=1.0, hits=0), dict(beta=1.0, algorithm='gibbs'), dict(beta=1.0, overrelax=-1)):
        with pytest.raises(ValueError):
            SONModel(field=field, logger=None, **kwargs)
    assert SONModel(field=field, logger=None, beta=1.0).N == 3
    assert SONModel(field=gauge_field(loc, cc, 2, 2, group='so'), logger=None, beta=1.0).N == 2
    assert type(gauge_field(loc, cc, 2, 2, group='so')) is SONField
    if MPI.COMM_WORLD.Get_size() == 1:
        with pytest.raises(ValueError, match='checkerboard'):
            SONModel(field=gauge_field([3, 4], CartesianComm(mpigrid=[1, 1]), 2, 3, group='so'), logger=None, beta=1.0)


# ----------------------------------------------------------------------------- restart
@needs_hdf5
@pytest.mark.parametrize('N', [2, 3])
def test_load_links_gives_back_the_saved_links_and_refuses_what_is_not_so_n(N):
    from pylattice.core.hdf5io import Writer
    cc, field, _ = make([8, 4], N=N)
    other = 5 - N
    with shared_tempdir() as d:
        path = os.path.join(d, 'conf.hdf5')
        Writer(lattice=field, cartesiancomm=cc).write(path)
        new, lines = gauge_field(field.grid, cc, 2, N, group='so'), []
        load_links(new, path, log=lines.append)
        assert np.array_equal(new.value, field.value) and 'SO({})'.format(N) in '\n'.join(lines)
        bad = gauge_field(field.grid, cc, 2, N, group='so')
        bad.value = bad.value * 1.5                                                    # orthogonality broken
        Writer(lattice=bad, cartesiancomm=cc).write(os.path.join(d, 'bad.hdf5'))
        target = gauge_field(field.grid, cc, 2, N, group='so')
        with pytest.raises(ValueError, match='not an SO\\({}\\) configuration'.format(N)):
            load_links(target, os.path.join(d, 'bad.hdf5'))
        assert np.array_equal(target.value, gauge_field(field.grid, cc, 2, N, group='so').value)    # untouched
        with pytest.raises(ValueError):                                                # a field of another N does not fit the file
            load_links(gauge_field(field.grid, cc, 2, other, group='so'), path)


@needs_hdf5
@pytest.mark.parametrize('N', [2, 3])
def test_a_restart_continues_the_run_exactly(N):
    import h5py
    cc, loc = setup([8, 4])
    base = ['--nc={}'.format(N), '--grid=' + 'x'.join(map(str, loc)), '--mpigrid=' + 'x'.join(map(str, cc.mpigrid)), '--beta=5.5',
            '--seed=4', '--init=hot', '--overrelax=1', '--logfile=off']

    def last(directory):
        path = os.path.join(directory, sorted(os.listdir(directory))[-1])
        with h5py.File(path, 'r') as f:
            return f['value'][...]

    with shared_tempdir() as d:
        if cc.rank == 0:
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
        first = os.path.abspath(os.path.join('checkpoints', run_name(N, loc, cc, '5.5') + '_00000003.hdf5'))
        os.chdir(os.path.join(d, 'second'))
        with quiet():
            main(base + ['--steps=3', '--checkpoint-every=3', '--start-from=' + first])
        MPI.COMM_WORLD.Barrier()
        if cc.rank == 0:
            assert np.array_equal(last('checkpoints'), straight)                       # bit for bit


# ----------------------------------------------------------------------------- command line
def test_the_parser_defaults_options_and_errors():
    P = SONParser(['--beta=6'])
    assert (P.N, P.theory, P.grid, P.mpigrid, P.beta, P.algorithm, P.overrelax, P.init) == (
        3, 'so3', [4, 4, 4, 4], [1, 1, 1, 2], 6.0, 'heatbath', 0, 'cold')
    assert SONParser(['--nc=2']).beta == 1.0 and SONParser(['--nc=2', '--beta=1.2']).beta == 1.2 and SONParser(['--nc=2']).theory == 'so2'
    Q = SONParser(['--nc=4', '--beta=10', '--grid=6x6', '--mpigrid=1x1', '--algorithm=metropolis', '--eps=0.2', '--hits=2', '--overrelax=3'])
    assert (Q.N, Q.algorithm, Q.eps, Q.hits, Q.overrelax) == (4, 'metropolis', 0.2, 2, 3)
    assert dict(Q.extra_info()) == {'gauge group': 'SO(4)', 'algorithm': 'metropolis', 'overrelaxation sweeps': 3,
                                    'proposal eps': 0.2, 'hits per link': 2, 'init': 'cold'}
    assert dict(P.extra_info())['gauge group'] == 'SO(3)'
    assert Q.run_name() == 'so4_grid6x6_mpi1x1_b10' and Q.meas_filename() == 'so4_grid6x6_mpi1x1_b10_meas.json'
    for argv in (['--nc=1'], ['--nc=x'], ['--eps=2'], ['--hits=0'], ['--overrelax=-1'], ['--algorithm=gibbs'], ['--grid=4x4', '--mpigrid=1x1x2'],
                 [], ['--nc=3'], ['--nc=5', '--steps=3']):                              # N >= 3 needs --beta
        with pytest.raises(SystemExit) as exc:
            SONParser(argv)
        assert exc.value.code == 2


@pytest.mark.parametrize('N', [2, 3])
def test_command_line_driver_writes_the_measurements_the_log_and_the_summary(N):
    cc, loc = setup([4, 4])
    argv = ['--nc={}'.format(N), '--grid=' + 'x'.join(map(str, loc)), '--mpigrid=' + 'x'.join(map(str, cc.mpigrid)), '--steps=6',
            '--thermalization=2', '--measure-every=2', '--beta=5.5', '--overrelax=1', '--seed=3', '--runlog-every=4']
    with shared_tempdir():
        with quiet() as out:
            main(argv)
        if cc.rank == 0:
            name = run_name(N, loc, cc, '5.5')
            records = json.load(open(name + '_meas.json'))['measurements']
            assert [r['sweep'] for r in records] == [0, 2, 4, 6]
            assert set(records[1]) == {'sweep', 'plaq_re', 'plaq_im', 'action', 'acceptance'}
            assert records[0]['acceptance'] is None and all(r['acceptance'] == 1.0 for r in records[1:])      # the heat bath
            assert all(r['plaq_im'] == 0.0 and 0 < r['plaq_re'] < 1 for r in records)
            content = open(name + '.log', encoding='utf-8').read()
            assert sorted(os.listdir('.')) == [name + '.log', name + '_meas.json']                  # every file says the run
            assert [l.split()[0] for l in content.splitlines() if l.startswith('step=')] == ['step=4', 'step=8']
            assert 'SO({})'.format(N) in content and 'gauge group' in content and '╭─ done' in content
            assert 'SO({})'.format(N) in out.getvalue()


# ----------------------------------------------------------------------------- the offline measurements
@needs_hdf5
def test_stored_so3_configurations_are_measured_and_the_files_of_another_group_or_lattice_are_refused():
    from pylattice.core.callbacks import Checkpointer
    from pylattice.core.measure import measure_files
    from pylattice.theories.gauge import son_meas, sun_meas
    cc, field, model = make([8, 4, 4], N=3, beta=5.5, hot=False)
    for _ in range(2):
        model.sweep()
    with shared_tempdir() as d:
        name = run_name(3, field.grid, cc, '5.5')
        paths = [Checkpointer(1, directory=d, prefix=name).save(model, s) for s in (4, 9)]
        out = os.path.join(d, 'm.json')
        measure_files(paths, lambda p: sun_meas.measure_su(p, cc, field.grid, 5.5, 3, 2, group='so'), MPI.COMM_WORLD, filename=out)
        MPI.COMM_WORLD.Barrier()
        want = {**plaquette_observables(field, 5.5), **sun_meas.loop_observables(field, 2)}
        records = json.load(open(out))['measurements']
        assert [r['sweep'] for r in records] == [4, 9]
        for key, value in want.items():
            assert records[0][key] == pytest.approx(value, abs=1e-12), key
        assert records[0]['plaq_im'] == 0.0 and abs(records[0]['polyakov']) > 0
        grid = '--mpigrid=' + 'x'.join(map(str, cc.mpigrid))
        son_meas.main(paths + ['--beta=5.5', '--max-loop=2', '--logfile', grid])
        MPI.COMM_WORLD.Barrier()
        assert sorted(f for f in os.listdir('.') if not f.endswith('.hdf5')) == [
            'm.json', name + '_meas_offline.json', name + '_meas_offline.log']
        offline = json.load(open(name + '_meas_offline.json'))['measurements']
        assert [r['sweep'] for r in offline] == [4, 9] and 'W22' in offline[0] and 'creutz22' in offline[0]
        assert 'SO(3)' in open(name + '_meas_offline.log', encoding='utf-8').read() if cc.rank == 0 else True
        bad = gauge_field(field.grid, cc, 3, 3, group='so')
        bad.value = bad.value * 1.5
        from pylattice.core.hdf5io import Writer
        Writer(lattice=bad, cartesiancomm=cc).write(os.path.join(d, 'bad.hdf5'))
        with pytest.raises(SystemExit) as exc:                                             # not orthogonal
            son_meas.main([os.path.join(d, 'bad.hdf5'), '--beta=5.5', '--out=' + os.path.join(d, 'x.json'), grid])
        assert exc.value.code == 1


if __name__ == '__main__':
    import traceback
    tests = [(name, f) for name, f in sorted(globals().items()) if name.startswith('test_') and callable(f)]
    for name, f in tests:
        if getattr(f, 'needs_hdf5', False) and not parallel_hdf5():
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('[-] {} skipped: no h5py with parallel HDF5'.format(name))
            continue
        marks = getattr(f, 'pytestmark', None)
        cases = [()]
        if marks and any(m.name == 'parametrize' for m in marks):
            cases = list(next(m.args[1] for m in marks if m.name == 'parametrize'))
            cases = [c if isinstance(c, tuple) else (c,) for c in cases]
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
