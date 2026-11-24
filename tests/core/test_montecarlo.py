"""Tests of pylattice.core.montecarlo: the driver run() and History, with a toy model that only
counts. Run on one process or on several ranks:

    pytest tests/core/test_montecarlo.py
    mpiexec -n 2 python tests/core/test_montecarlo.py        # 1, 2 or 4 ranks
"""
import contextlib
import io
import os
import shutil
import tempfile

import numpy as np
import pytest
from mpi4py import MPI

from pylattice.core import montecarlo
from pylattice.core.montecarlo import History
from pylattice.core.rng import rank_rng

COMM = MPI.COMM_WORLD
RANK, SIZE = COMM.Get_rank(), COMM.Get_size()


class Toy:
    """sweeps do nothing but count. On rank r a sweep accepts r+1 of its 4 updates; the observables
    are the number of sweeps made so far (and twice that)."""

    def __init__(self, with_comm=True, alternating=False):
        if with_comm:
            self.comm = COMM
        self.alternating = alternating                    # accept only on the even sweeps (by number)
        self.sweeps = 0
        self.calls = []
        self.rngs = []

    def sweep(self, rng):
        self.sweeps += 1
        self.calls.append('s')
        self.rngs.append(rng)
        if self.alternating and self.sweeps % 2:
            return 0, 4
        return RANK + 1, 4

    def observables(self):
        self.calls.append('o')
        return {'x': float(self.sweeps), 'y': 2.0 * self.sweeps}


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()) as out:
        yield out


# the acceptance of a sweep over all the ranks: sum(r + 1) / (4 * SIZE)
GLOBAL_ACCEPTANCE = (SIZE + 1) / 8.0


def test_schedule():
    toy = Toy()
    h = montecarlo.run(toy, steps=10, thermalization=3, measure_every=2, progress=False)
    assert toy.sweeps == 13                                           # thermalization + steps
    assert h.sweeps == [0, 2, 4, 6, 8, 10] and len(h) == 6            # 0 is the end of the thermalization
    assert list(h['x']) == [3, 5, 7, 9, 11, 13] and list(h['y']) == [6, 10, 14, 18, 22, 26]
    assert len(h.acceptance) == 5


def test_every_sweep_is_measured_by_default():
    toy = Toy()
    h = montecarlo.run(toy, steps=5, progress=False)
    assert h.sweeps == [0, 1, 2, 3, 4, 5] and list(h['x']) == [0, 1, 2, 3, 4, 5]


def test_acceptance_is_the_global_fraction():
    for every in (1, 2, 5):
        h = montecarlo.run(Toy(), steps=10, measure_every=every, progress=False)
        assert np.allclose(h.acceptance, GLOBAL_ACCEPTANCE), (every, h.acceptance)
        assert len(h.acceptance) == 10 // every


def test_acceptance_covers_the_sweeps_since_the_previous_measurement():
    # sweeps 2, 4, 6, ... accept (SIZE+1)/8, the others nothing. With the thermalization sweep (number 1)
    # the measurements cover sweeps (2,3,4), (5,6,7), (8,9,10): 2, 1 and 2 of the accepting ones
    h = montecarlo.run(Toy(alternating=True), steps=9, thermalization=1, measure_every=3, progress=False)
    assert np.allclose(h.acceptance, np.array([2, 1, 2]) * GLOBAL_ACCEPTANCE / 3), h.acceptance


def test_the_last_sweeps_are_not_measured_if_measure_every_does_not_divide_steps():
    toy = Toy()
    h = montecarlo.run(toy, steps=7, measure_every=3, progress=False)
    assert toy.sweeps == 7 and h.sweeps == [0, 3, 6]


def test_without_measuring_only_the_sweeps_are_made():
    toy = Toy()
    h = montecarlo.run(toy, steps=6, thermalization=2, progress=False, measure=False)
    assert toy.sweeps == 8 and len(h) == 0 and h.sweeps == [] and h.acceptance == []
    assert 'o' not in toy.calls                                       # the observables were never asked for


def test_zero_steps():
    toy = Toy()
    h = montecarlo.run(toy, steps=0, progress=False)
    assert toy.sweeps == 0 and h.sweeps == [0] and h.acceptance == [] and list(h['x']) == [0]
    toy = Toy()
    h = montecarlo.run(toy, steps=0, thermalization=4, progress=False)
    assert toy.sweeps == 4 and list(h['x']) == [4]                    # the thermalization still runs


def test_sweeps_and_measurements_are_called_in_the_same_order_on_every_rank():
    # both involve collective MPI calls
    toy = Toy()
    montecarlo.run(toy, steps=4, thermalization=2, measure_every=2, progress=False)
    assert toy.calls == ['s', 's', 'o', 's', 's', 'o', 's', 's', 'o']
    assert all(c == toy.calls for c in COMM.allgather(toy.calls))


def test_the_communicator_is_optional():
    toy = Toy(with_comm=False)
    assert not hasattr(toy, 'comm')
    h = montecarlo.run(toy, steps=3, progress=False)                  # falls back to COMM_WORLD
    assert np.allclose(h.acceptance, GLOBAL_ACCEPTANCE)


def test_invalid_arguments():
    for kw in (dict(steps=-1), dict(steps=1, thermalization=-1), dict(steps=1, measure_every=0)):
        toy = Toy()
        with pytest.raises(ValueError):
            montecarlo.run(toy, progress=False, **kw)
        assert toy.sweeps == 0                                        # nothing was run


def test_progress_bar_is_drawn_by_rank_zero_only():
    with quiet() as out:
        montecarlo.run(Toy(), steps=6, thermalization=4)
    text = out.getvalue()
    if RANK == 0:
        assert 'MONTE CARLO' in text and '10/10' in text              # one line when the output is not a terminal
    else:
        assert text == ''
    with quiet() as out:
        montecarlo.run(Toy(), steps=6, progress=False)
    assert out.getvalue() == ''


class Hooks:
    """a callback with all three hooks, recording what it is given"""

    def __init__(self):
        self.events = []

    def start(self, model):
        self.events.append(('start', model.sweeps))

    def __call__(self, model, step, observables):
        self.events.append(('step', step, model.sweeps, observables()['x']))

    def finish(self, model, step):
        self.events.append(('finish', step, model.sweeps))


def test_callbacks_are_called_after_every_sweep_thermalization_included():
    toy, hooks = Toy(), Hooks()
    montecarlo.run(toy, steps=3, thermalization=2, progress=False, callbacks=[hooks])
    assert hooks.events[0] == ('start', 0)                              # before the first sweep
    assert hooks.events[1:-1] == [('step', k, k, float(k)) for k in range(1, 6)]   # step 1..5; the model and its observables are at that sweep
    assert hooks.events[-1] == ('finish', 5, 5) and len(hooks.events) == 7


class OnlyMeasurements:
    """not callable: it has just the measurement hook"""

    def __init__(self):
        self.events = []

    def measurement(self, model, sweep, values, acceptance):
        self.events.append((model.sweeps, sweep, dict(values), acceptance))


def test_the_measurement_hook_is_called_when_a_measurement_is_made():
    toy, hook = Toy(), OnlyMeasurements()
    h = montecarlo.run(toy, steps=6, thermalization=3, measure_every=2, progress=False, callbacks=[hook])
    # the model is at sweep 3 + n when measurement n is made; the thermalization is not measured
    assert [e[:2] for e in hook.events] == [(3, 0), (5, 2), (7, 4), (9, 6)]
    assert [e[1] for e in hook.events] == h.sweeps                              # the same sweeps as the History
    assert [e[2] for e in hook.events] == [{'x': 3.0 + n, 'y': 6.0 + 2 * n} for n in (0, 2, 4, 6)]
    assert [e[3] for e in hook.events] == [None] + h.acceptance                 # no acceptance for the first measurement
    # no measurement, no call
    hook2 = OnlyMeasurements()
    montecarlo.run(Toy(), steps=4, progress=False, measure=False, callbacks=[hook2])
    assert hook2.events == []


def test_the_measurement_hook_comes_before_the_per_sweep_callbacks_of_that_sweep():
    order = []

    class Both(OnlyMeasurements):
        def __call__(self, model, step, observables):
            order.append(('sweep', step))

        def measurement(self, model, sweep, values, acceptance):
            order.append(('measurement', sweep))

    montecarlo.run(Toy(), steps=4, thermalization=1, measure_every=2, progress=False, callbacks=[Both()])
    # 'sweep' is the step, which counts the thermalization sweep; 'measurement' is the sweep after it:
    # measurement 2 is made during step 3, before the callbacks of that step
    assert order == [('sweep', 1), ('measurement', 0), ('sweep', 2), ('measurement', 2), ('sweep', 3),
                     ('sweep', 4), ('measurement', 4), ('sweep', 5)]


def test_the_hook_sees_the_observables_without_computing_them_again():
    toy, hook = Toy(), OnlyMeasurements()
    montecarlo.run(toy, steps=5, progress=False, callbacks=[hook])
    assert toy.calls.count('o') == 6 and len(hook.events) == 6                  # one call per measurement, shared with the History
    assert all(c == toy.calls for c in COMM.allgather(toy.calls))


def test_callbacks_can_be_plain_functions_and_any_iterable():
    seen = []
    montecarlo.run(Toy(), steps=3, progress=False, callbacks=(lambda model, step, observables: seen.append(step),))
    montecarlo.run(Toy(), steps=2, progress=False, callbacks=(f for f in [lambda model, step, observables: seen.append(-step)]))
    assert seen == [1, 2, 3, -1, -2]


def test_callbacks_run_in_the_given_order():
    order = []
    montecarlo.run(Toy(), steps=2, progress=False, callbacks=[lambda m, s, o: order.append(('a', s)),
                                                              lambda m, s, o: order.append(('b', s))])
    assert order == [('a', 1), ('b', 1), ('a', 2), ('b', 2)]


def test_observables_are_computed_once_per_sweep_however_many_ask():
    toy = Toy()
    ask = lambda model, step, observables: (observables(), observables())
    h = montecarlo.run(toy, steps=6, progress=False, callbacks=[ask, ask])   # + the measurement of every sweep
    assert toy.calls.count('o') == 7 and len(h) == 7                        # sweep 0 and one per sweep, not more
    toy = Toy()
    montecarlo.run(toy, steps=6, thermalization=3, measure_every=2, progress=False, callbacks=[ask])
    # asked at every one of the 9 sweeps; the measurement of sweep 0 is the state of sweep 3, already computed
    assert toy.calls.count('o') == 9
    toy = Toy()
    montecarlo.run(toy, steps=4, progress=False, measure=False, callbacks=[ask])    # the callbacks alone ask for them
    assert toy.calls.count('o') == 4


def test_callbacks_see_the_same_collective_calls_on_every_rank():
    toy = Toy()
    montecarlo.run(toy, steps=4, thermalization=1, measure_every=2, progress=False,
                   callbacks=[lambda m, s, o: o() if s % 2 == 0 else None])
    assert all(c == toy.calls for c in COMM.allgather(toy.calls))


def test_an_error_in_a_callback_stops_the_run():
    def boom(model, step, observables):
        if step == 3:
            raise RuntimeError('stop')
    toy = Toy()
    with pytest.raises(RuntimeError):
        montecarlo.run(toy, steps=10, progress=False, callbacks=[boom])
    assert toy.sweeps == 3


# ----------------------------------------------------------------- random numbers
def test_every_sweep_gets_the_generator_of_this_rank():
    toy = Toy()
    montecarlo.run(toy, steps=4, thermalization=2, progress=False)
    assert len(toy.rngs) == 6 and all(r is toy.rngs[0] for r in toy.rngs)         # one generator for the whole run
    assert isinstance(toy.rngs[0], np.random.Generator)
    assert np.array_equal(toy.rngs[0].random(5), rank_rng(0).random(5))             # by default the stream of this rank, seed 0
    # the model's own generator, if it has one
    toy = Toy()
    toy.rng = rank_rng(9)
    montecarlo.run(toy, steps=2, progress=False)
    assert all(r is toy.rng for r in toy.rngs)
    # and one given to run() wins over both
    mine = rank_rng(4)
    toy = Toy()
    toy.rng = rank_rng(9)
    montecarlo.run(toy, steps=2, progress=False, rng=mine)
    assert all(r is mine for r in toy.rngs)


class Noisy(Toy):
    """the observable is a sum of the random numbers of the sweeps"""
    total = 0.0

    def sweep(self, rng):
        super().sweep(rng)
        self.total += rng.random()
        return 1, 1

    def observables(self):
        return {'x': self.total}


def test_a_run_is_reproducible_from_its_generator():
    a = montecarlo.run(Noisy(), steps=10, progress=False, rng=rank_rng(3))
    b = montecarlo.run(Noisy(), steps=10, progress=False, rng=rank_rng(3))
    c = montecarlo.run(Noisy(), steps=10, progress=False, rng=rank_rng(4))
    assert np.array_equal(a['x'], b['x']) and not np.array_equal(a['x'], c['x'])
    np.random.seed(1)                                                               # the global numpy state plays no part
    d = montecarlo.run(Noisy(), steps=10, progress=False, rng=rank_rng(3))
    assert np.array_equal(a['x'], d['x'])


def test_the_ranks_of_a_run_draw_different_numbers():
    final = COMM.allgather(montecarlo.run(Noisy(), steps=10, progress=False, rng=rank_rng(3))['x'][-1])
    assert len(set(final)) == SIZE


# ------------------------------------------------------------- Metropolis test
def test_metropolis_accept_probability_is_exp_of_minus_the_action_change():
    rng = np.random.default_rng(1)
    n = 200000
    for dS in (0.2, 0.5, 1.0, 2.0, 4.0):
        p = np.exp(-dS)
        accepted = montecarlo.metropolis_accept(np.full(n, dS), rng)
        assert accepted.dtype == bool and accepted.shape == (n,)
        assert abs(accepted.mean() - p) < 5 * np.sqrt(p * (1 - p) / n), (dS, accepted.mean(), p)


def test_metropolis_accept_edge_cases():
    rng = np.random.default_rng(2)
    with np.errstate(over='raise', invalid='raise', divide='raise'):                # no overflow, whatever the action change (a tiny exp underflowing to 0 is the point)
        assert montecarlo.metropolis_accept(np.full(5000, -3.0), rng).all()         # lowering the action: always
        assert montecarlo.metropolis_accept(np.zeros(5000), rng).all()              # keeping it: always
        assert montecarlo.metropolis_accept(np.full(5000, -1e4), rng).all()         # exp(+1e4) would overflow
        assert not montecarlo.metropolis_accept(np.full(5000, 1e4), rng).any()      # exp(-1e4) underflows to 0
        assert not montecarlo.metropolis_accept(np.full(200000, 40.0), rng).any()
    out = montecarlo.metropolis_accept(np.full((3, 4), 0.5), rng)
    assert out.shape == (3, 4)                                                      # the shape of the input
    assert montecarlo.metropolis_accept(np.array([0.1, -0.1], dtype='float32'), rng).shape == (2,)
    assert montecarlo.metropolis_accept([-1.0, -2.0], rng).all()                    # a list will do
    assert bool(montecarlo.metropolis_accept(-1.0, rng)) is True                    # and a number


def test_metropolis_accept_draws_one_number_per_update_whatever_the_outcome():
    for dS in (-5.0, 0.0, 2.0, 50.0):
        used, reference = np.random.default_rng(5), np.random.default_rng(5)
        montecarlo.metropolis_accept(np.full(37, dS), used)
        reference.random(37)
        assert used.random() == reference.random()                                  # the streams are at the same place


def test_metropolis_accept_uses_the_given_generator_only():
    dS = np.full(1000, 0.7)
    a = montecarlo.metropolis_accept(dS, np.random.default_rng(8))
    np.random.seed(123)
    b = montecarlo.metropolis_accept(dS, np.random.default_rng(8))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, montecarlo.metropolis_accept(dS, np.random.default_rng(9)))


def test_metropolis_accept_beta_belongs_inside_the_action():
    # a theory with energy change dE at inverse temperature beta passes beta * dE: the colder, the fewer flips
    rng = np.random.default_rng(3)
    dE = np.full(100000, 2.0)
    rates = [montecarlo.metropolis_accept(beta * dE, rng).mean() for beta in (0.1, 0.5, 1.0, 2.0)]
    assert rates == sorted(rates, reverse=True) and abs(rates[1] - np.exp(-1.0)) < 0.01


# ---------------------------------------------------------------------- plot
def test_history_plot_makes_one_figure_per_observable_and_one_for_the_acceptance():
    import pylattice.core.parallel as parallel
    h = montecarlo.run(Toy(), steps=6, progress=False)
    calls = []
    original = parallel.pplot
    parallel.pplot = lambda comm, x, y, title, file: calls.append((list(x), list(y), title, file))
    try:
        files = h.plot(COMM)
        assert files == ['./x.png', './y.png', './acceptance.png']                  # the observables in order, then the acceptance
        assert [c[3] for c in calls] == files and [c[2] for c in calls] == ['x', 'y', 'acceptance']
        assert calls[0][0] == [0, 1, 2, 3, 4, 5, 6] and calls[0][1] == [0, 1, 2, 3, 4, 5, 6]     # x against the sweeps
        assert calls[2][0] == [1, 2, 3, 4, 5, 6] and len(calls[2][1]) == 6          # the acceptance starts at sweep 1
        calls.clear()
        files = h.plot(COMM, files={'x': 'a.png', 'acceptance': 'acc.png'}, titles={'x': 'The X', 'y': 'The Y'})
        assert files == ['a.png', './y.png', 'acc.png'] and [c[2] for c in calls] == ['The X', 'The Y', 'acceptance']
        calls.clear()
        assert History().plot(COMM) == [] and calls == []                           # nothing to plot
        no_acceptance = montecarlo.run(Toy(), steps=0, progress=False)
        assert no_acceptance.plot(COMM) == ['./x.png', './y.png']
    finally:
        parallel.pplot = original


def test_history_plot_writes_png_files_from_rank_zero_only():
    h = montecarlo.run(Toy(), steps=6, progress=False)
    old = os.getcwd()
    d = tempfile.mkdtemp()
    os.chdir(d)
    try:
        with quiet():
            h.plot(COMM)
        names = sorted(os.listdir(d))
        if RANK == 0:
            assert names == ['acceptance.png', 'x.png', 'y.png']
            with open('x.png', 'rb') as f:
                assert f.read(8) == b'\x89PNG\r\n\x1a\n'
        else:
            assert names == []
    finally:
        os.chdir(old)
        shutil.rmtree(d, ignore_errors=True)


def test_history_record_and_arrays():
    h = History()
    assert len(h) == 0 and h.arrays()['sweep'].size == 0
    h.record(0, {'a': 1.0, 'b': 5})
    h.record(2, {'a': 2.0, 'b': 6}, acceptance=0.5)
    assert len(h) == 2 and h.sweeps == [0, 2] and h.acceptance == [0.5]
    arrays = h.arrays()
    assert set(arrays) == {'sweep', 'acceptance', 'a', 'b'}
    assert arrays['sweep'].dtype == int and arrays['a'].dtype == float and list(arrays['b']) == [5.0, 6.0]
    assert h['a'].dtype == float and list(h['a']) == [1.0, 2.0]
    with pytest.raises(ValueError):
        h.record(3, {'a': 3.0})                                       # the observables must not change
    with pytest.raises(ValueError):
        History().record(0, {'sweep': 1.0})                           # reserved names
    with pytest.raises(ValueError):
        History().record(0, {'acceptance': 1.0})
    assert len(h) == 2                                                # the failed records changed nothing


def test_history_save():
    h = montecarlo.run(Toy(), steps=4, thermalization=1, progress=False)
    d = tempfile.mkdtemp()
    path = os.path.join(d, 'run.npz')
    h.save(path)
    loaded = np.load(path)
    assert set(loaded.files) == {'sweep', 'acceptance', 'x', 'y'}
    for name, arr in h.arrays().items():
        assert np.array_equal(loaded[name], arr)
    assert list(loaded['x']) == [1, 2, 3, 4, 5]


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
