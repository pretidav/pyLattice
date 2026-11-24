"""Callbacks of the Monte Carlo driver (pylattice.core.montecarlo.run):

    Checkpointer   saves the configuration to disk every few sweeps
    RunLog         writes a line of diagnostics to a text file every few sweeps
    MeasurementFile  writes every measurement to a JSON file as soon as it is made

They are called as callback(model, step, observables) after every sweep, thermalization included;
step counts the sweeps since run() started. They run on every rank, at the same steps, because they
use collective MPI calls.
"""
import json
import math
import os
import time

import numpy as np
import psutil
from mpi4py import MPI


def _check_every(every):
    if int(every) != every or every < 1:
        raise ValueError("every must be an integer of at least 1, got {}".format(every))
    return int(every)


class Checkpointer:
    """Saves the configuration every `every` sweeps, in parallel HDF5 (pylattice.core.hdf5io).

    A checkpoint is the file  directory/prefix_<step>.hdf5  (step with 8 digits, so the names sort),
    which has the attribute `step`; read it with pylattice.core.hdf5io.Reader. It is written under a
    temporary name and renamed when it is complete, so a job killed during a write never leaves a
    truncated checkpoint.

    every      sweeps between two checkpoints
    directory  where they go (created if needed)
    prefix     start of the file names
    keep       keep only the newest `keep` checkpoints (default: all)
    final      also save after the last sweep, if it is not a checkpoint already
    lattice    function model -> the lattice to save (default: model.field)

    Saving needs h5py built with parallel HDF5 (see the README). Only the configuration is saved,
    not the state of the random number generators, so a restart is not bit-for-bit the same run.
    """

    def __init__(self, every, directory='checkpoints', prefix='config', keep=None, final=False, lattice=None):
        self.every = _check_every(every)
        if keep is not None and keep < 1:
            raise ValueError("keep must be at least 1, got {}".format(keep))
        self.directory, self.prefix, self.keep, self.final = directory, prefix, keep, final
        self.lattice = lattice if lattice is not None else self._field
        self.last = None                                  # step of the last checkpoint

    @staticmethod
    def _field(model):
        field = getattr(model, 'field', None)
        if field is None:
            raise TypeError("the model has no `field` to save: give Checkpointer(lattice=function of the model)")
        return field

    def path(self, step):
        return os.path.join(self.directory, '{}_{:08d}.hdf5'.format(self.prefix, step))

    def start(self, model):
        comm = self.lattice(model).cartesiancomm.comm
        if comm.Get_rank() == 0:
            os.makedirs(self.directory, exist_ok=True)
            for name in os.listdir(self.directory):          # left by a job that was killed while writing
                if name.startswith(self.prefix + '_') and name.endswith('.partial.hdf5'):
                    os.remove(os.path.join(self.directory, name))
        comm.Barrier()
        self.last = None

    def __call__(self, model, step, observables):
        if step % self.every == 0:
            self.save(model, step)

    def finish(self, model, step):
        if self.final and self.last != step:
            self.save(model, step)

    def save(self, model, step):
        """write the checkpoint of this step; returns its path"""
        import h5py
        from pylattice.core.hdf5io import Writer          # h5py (with MPI) is only needed to save
        lattice = self.lattice(model)
        cartesiancomm = lattice.cartesiancomm
        comm = cartesiancomm.comm
        final = self.path(step)
        partial = final[:-len('.hdf5')] + '.partial.hdf5'
        Writer(lattice=lattice, cartesiancomm=cartesiancomm).write(filename=partial)
        comm.Barrier()                                       # the file is closed on every rank
        if comm.Get_rank() == 0:
            with h5py.File(partial, 'r+') as f:
                f.attrs['step'] = step
            os.replace(partial, final)
            self._prune()
        comm.Barrier()
        self.last = step
        return final

    def _prune(self):
        if self.keep is None:
            return
        steps = []
        for name in os.listdir(self.directory):
            body = name[len(self.prefix) + 1:-len('.hdf5')]
            if name.startswith(self.prefix + '_') and name.endswith('.hdf5') and body.isdigit():
                steps.append(int(body))
        for step in sorted(steps)[:-self.keep]:
            os.remove(self.path(step))


class RunLog:
    """One line per `every` sweeps in a text file (rank 0 writes it):

        step=100 m=0.123 e=-1.91 mem_MB=312.4 cpu_pct=98.7 steps_per_s=211.30

    step          sweeps since run() started (thermalization included)
    observables   those of the model, in the order it gives them
    mem_MB        resident memory of all the ranks together
    cpu_pct       cpu use of a rank since the previous line, averaged over the ranks (100 = one core)
    steps_per_s   sweeps per second of wall time since the previous line

    The file is appended to (mode='w' starts it afresh) and flushed at every line.
    """

    def __init__(self, every, filename='run.log', mode='a'):
        self.every = _check_every(every)
        self.filename, self.mode = filename, mode
        self.file = None

    def start(self, model):
        self.comm = getattr(model, 'comm', MPI.COMM_WORLD)
        self.process = psutil.Process()
        self.process.cpu_percent(interval=None)            # the first call only starts the counter
        if self.comm.Get_rank() == 0:
            self.file = open(self.filename, self.mode)
        self.last_step = 0
        self.last_time = time.perf_counter()

    def __call__(self, model, step, observables):
        if step % self.every:
            return
        now = time.perf_counter()
        values = observables()                               # these three are collective calls
        memory = self.comm.allreduce(self.process.memory_info().rss)
        cpu = self.comm.allreduce(self.process.cpu_percent(interval=None)) / self.comm.Get_size()
        if self.comm.Get_rank() == 0:
            speed = (step - self.last_step) / (now - self.last_time)
            line = 'step={}'.format(step)
            line += ''.join(' {}={:.6g}'.format(name, value) for name, value in values.items())
            line += ' mem_MB={:.1f} cpu_pct={:.1f} steps_per_s={:.2f}'.format(memory / 2 ** 20, cpu, speed)
            self.file.write(line + '\n')
            self.file.flush()
        self.last_step = step
        self.last_time = time.perf_counter()               # the time spent here is not counted in the speed

    def finish(self, model, step):
        if self.file is not None:
            self.file.close()
            self.file = None


def _jsonable(value):
    """a number or an array as plain JSON: numpy types converted, nan and inf (not JSON) as null"""
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class MeasurementFile:
    """Writes every measurement to a JSON file as soon as it is made, not at the end of the run.

        {"measurements": [
          {"sweep": 0, "m": 0.01, "e": -0.02, "acceptance": null},
          {"sweep": 1, "m": 0.02, "e": -0.05, "acceptance": 0.46}
        ]}

    sweep and acceptance are those of the History (the sweep counts from the end of the
    thermalization; the acceptance, null for the first measurement, covers the sweeps since the
    previous one), the other entries are the observables of the model.

    The file is a valid JSON document from the start (an empty list) and after every measurement: the
    closing of the list is rewritten after each record and the file is flushed, so a job that dies
    keeps all its measurements, and another program can read the file while the run goes on. Rank 0
    writes it; an existing file is overwritten. nan and inf are written as null.

    Read it with  json.load(open('meas.json'))['measurements']  (a list of dicts).
    """
    _HEADER = b'{"measurements": ['
    _CLOSING = b'\n]}\n'

    def __init__(self, filename='meas.json'):
        self.filename = filename
        self.file = None

    def start(self, model):
        comm = getattr(model, 'comm', MPI.COMM_WORLD)
        if comm.Get_rank() == 0:
            self.file = open(self.filename, 'wb')
            self.file.write(self._HEADER + self._CLOSING)
            self.file.flush()
            self.end = len(self._HEADER)                      # where the next record goes
            self.count = 0

    def measurement(self, model, sweep, values, acceptance):
        # every rank encodes the record (the values are the same on all of them), so a value that
        # cannot be written fails everywhere at once instead of leaving the other ranks waiting for rank 0
        record = {'sweep': sweep}
        record.update({name: _jsonable(value) for name, value in values.items()})
        record['acceptance'] = _jsonable(acceptance)
        text = json.dumps(record, allow_nan=False).encode()
        if self.file is None:
            return
        data = (b',' if self.count else b'') + b'\n  ' + text
        self.file.seek(self.end)
        self.file.write(data + self._CLOSING)                 # the record and the closing, in one write
        self.file.flush()
        self.end += len(data)
        self.count += 1

    def finish(self, model, step):
        if self.file is not None:
            self.file.close()
            self.file = None
