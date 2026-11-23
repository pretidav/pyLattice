"""Monte Carlo driver, independent of the theory.

The theories own the update (Metropolis, heatbath, an HMC trajectory, ...); this module owns what
is around it: the loop, the thermalization / measurement schedule, the bookkeeping of the
observables and of the acceptance, and the progress bar.

A model is any object with

    sweep(rng) -> (accepted, attempted)   one sweep over the lattice: the counts of this rank
    observables() -> dict                 name -> value, global (the same on every rank)
    comm                                  (optional) its MPI communicator, default COMM_WORLD
    rng                                   (optional) the numpy Generator of this rank, see run()

All the randomness of a sweep must come from the rng it is given: that is what makes a run
reproducible, with a different independent stream on every rank (pylattice.core.rng).

sweep() and observables() involve collective MPI calls, so run() calls them on every rank, in the
same order. Only the progress bar is drawn by rank 0 alone.

Callbacks (see pylattice.core.callbacks: Checkpointer, RunLog) are objects called as
callback(model, step, observables) after every sweep, thermalization included. step counts the
sweeps since run() started (1, 2, ...); observables() gives the observables of the current
configuration, computed at most once per sweep however many ask. A callback may also have
start(model), called before the first sweep, and finish(model, step), called after the last one.
Callbacks run on every rank at the same steps, so they may use collective MPI calls.
"""
import numpy as np
from mpi4py import MPI

from pylattice.core.rng import rank_rng
from rich.progress import Progress, BarColumn, MofNCompleteColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

_RESERVED = ('sweep', 'acceptance')


def metropolis_accept(delta_action, rng):
    """Metropolis test of many independent updates at once: update i is accepted with probability
    min(1, exp(-delta_action[i])).

    delta_action is the change of the action S, the quantity in the weight exp(-S): the inverse
    temperature is already inside it (S = beta * H, so a theory with an energy H passes
    beta * delta_H). Returns a boolean array of the shape of delta_action.

    One random number is drawn per update, whether or not it is needed, so the stream of the
    generator does not depend on the outcomes.
    """
    delta_action = np.asarray(delta_action)
    # exp(-max(dS, 0)) is 1 for dS <= 0 and never overflows, however negative dS is
    return rng.random(delta_action.shape) < np.exp(-np.maximum(delta_action, 0))


class _Once:
    """model.observables() for the current configuration, computed at most once per sweep however
    many (the History, the callbacks) ask for it. It is a collective call: every rank must ask for
    it at the same sweeps."""

    def __init__(self, observables):
        self._observables = observables
        self._value = None

    def reset(self):
        self._value = None

    def __call__(self):
        if self._value is None:
            self._value = self._observables()
        return self._value


class History:
    """Time series of a run.

    sweeps       sweep after which each measurement was made: 0 is the configuration at the end of
                 the thermalization, then every measure_every sweeps
    values       observable name -> list, one value per measurement
    acceptance   fraction of the updates accepted over the sweeps since the previous measurement,
                 so it has one entry fewer than sweeps (acceptance[i] goes with sweeps[i+1])
    """

    def __init__(self):
        self.sweeps = []
        self.values = {}
        self.acceptance = []

    def record(self, sweep, observables, acceptance=None):
        names = set(observables)
        if names & set(_RESERVED):
            raise ValueError("observable names cannot be {}".format(_RESERVED))
        if self.sweeps and names != set(self.values):
            raise ValueError("the observables changed: {} instead of {}".format(sorted(names), sorted(self.values)))
        for name, value in observables.items():
            self.values.setdefault(name, []).append(value)
        self.sweeps.append(sweep)
        if acceptance is not None:
            self.acceptance.append(acceptance)

    def __len__(self):
        return len(self.sweeps)

    def __getitem__(self, name):
        """the values of an observable, as a float array"""
        return np.array(self.values[name], dtype=float)

    def arrays(self):
        """everything as arrays: 'sweep', 'acceptance' and one per observable"""
        out = {name: np.array(v, dtype=float) for name, v in self.values.items()}
        out['sweep'] = np.array(self.sweeps, dtype=int)
        out['acceptance'] = np.array(self.acceptance, dtype=float)
        return out

    def save(self, path):
        """write the arrays to a .npz file (it is the same on every rank: call it on rank 0)"""
        np.savez(path, **self.arrays())

    def plot(self, comm=None, files=None, titles=None):
        """One figure per observable, and one for the acceptance (called 'acceptance'), against the
        sweep; rank 0 writes them (pylattice.core.parallel.pplot, in the style of plot_style).

        files   name -> file name; default './<name>.png'
        titles  name -> title; default the name
        Returns the list of the files. An empty History plots nothing.
        """
        from pylattice.core.parallel import pplot
        comm = MPI.COMM_WORLD if comm is None else comm
        files, titles = files or {}, titles or {}
        series = [(name, self.sweeps, values) for name, values in self.values.items()]
        if self.acceptance:
            series.append(('acceptance', self.sweeps[1:], self.acceptance))
        written = []
        for name, x, y in series:
            path = files.get(name, './{}.png'.format(name))
            pplot(comm, x, y, titles.get(name, name), path)
            written.append(path)
        return written


def run(model, steps, thermalization=0, measure_every=1, progress=True, measure=True, callbacks=(), rng=None):
    """Run thermalization + steps sweeps of the model and return the History of the measurements.

    thermalization  sweeps made first, without measuring
    steps           sweeps made after that. With measure=True the observables are measured on the
                    configuration after the thermalization (sweep 0) and then after every
                    measure_every sweeps (so the last one is missed if measure_every does not
                    divide steps)
    progress        progress bar, drawn by rank 0 only
    measure         False: just make the sweeps (the History is empty)
    callbacks       called after every sweep, thermalization included (see the module docstring)
    rng             the numpy Generator of this rank, given to every sweep(rng). Default: the model's
                    `rng` if it has one, else the stream of this rank for seed 0 (pylattice.core.rng).
                    Make it with CartesianComm.rng(seed) or rank_rng(seed) to choose the seed
    """
    if steps < 0 or thermalization < 0:
        raise ValueError("steps and thermalization cannot be negative")
    if measure_every < 1:
        raise ValueError("measure_every must be at least 1")
    comm = getattr(model, 'comm', MPI.COMM_WORLD)
    if rng is None:
        rng = getattr(model, 'rng', None)
    if rng is None:
        rng = rank_rng(0, comm)
    history = History()
    callbacks = list(callbacks)
    observables = _Once(model.observables)
    accepted = attempted = 0                               # this rank, since the last measurement
    step = 0                                               # sweeps made, thermalization included
    for callback in callbacks:
        getattr(callback, 'start', lambda model: None)(model)
    bar = Progress(TextColumn('[progress.description]{task.description}'), BarColumn(), MofNCompleteColumn(),
                   TimeElapsedColumn(), TimeRemainingColumn(),
                   disable=not (progress and comm.Get_rank() == 0))
    with bar:
        task = bar.add_task('THERMALIZATION' if thermalization else 'MONTE CARLO', total=thermalization + steps)
        for _ in range(thermalization):
            model.sweep(rng)
            step += 1
            observables.reset()
            for callback in callbacks:
                callback(model, step, observables)
            bar.advance(task)
        if thermalization:
            bar.update(task, description='MONTE CARLO')
        if measure:
            history.record(0, observables())
        for n in range(1, steps + 1):
            a, t = model.sweep(rng)
            step += 1
            observables.reset()
            accepted += int(a)
            attempted += int(t)
            if measure and n % measure_every == 0:
                total_accepted, total_attempted = comm.allreduce(np.array([accepted, attempted]))
                history.record(n, observables(),
                               total_accepted / total_attempted if total_attempted else float('nan'))
                accepted = attempted = 0
            for callback in callbacks:
                callback(model, step, observables)
            bar.advance(task)
    for callback in callbacks:
        getattr(callback, 'finish', lambda model, step: None)(model, step)
    return history
