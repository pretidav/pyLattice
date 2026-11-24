"""Measurements on configurations that are already on disk (the checkpoints of a run), without running
any Monte Carlo: the offline counterpart of the measurements made during a run.

    measure_files(files, measure, comm, filename='meas.json')

For every file it calls measure(path), which loads the configuration and returns its observables as a
dict, and writes the record to a meas.json of the same format as a run's (pylattice.core.callbacks.
MeasurementFile), with the step the checkpoint was saved at as "sweep" and no acceptance. The records
are in the order of the steps, each one flushed as soon as it is made, so the file is valid at every
moment. Print and plot it like any other: python -m pylattice.utils.plot_meas meas.json

A theory supplies measure() (see pylattice.theories.spin.ising_meas). Everything is collective: every
rank takes part in the loading and in the observables.
"""
from pylattice.core.callbacks import MeasurementFile
from pylattice.core.hdf5io import describe


class _Model:
    """what MeasurementFile asks of a model: the communicator"""

    def __init__(self, comm):
        self.comm = comm


def steps_of(files, comm):
    """the step each file was saved at (None if it does not say)"""
    return [describe(f, comm)['attrs'].get('step') for f in files]


def measure_files(files, measure, comm, filename='meas.json', log=None):
    """Measures the configuration in each file; returns the list of the records written, [(step, values)].

    files     paths of HDF5 files written by a Checkpointer; measured in the order of their steps (files
              that do not say their step come after, in the order given, and get their position as step)
    measure   function path -> dict observable name -> number, called on every rank
    comm      the MPI communicator
    filename  the measurements file, rank 0 writes it; an existing one is overwritten
    log       a function that takes a line (e.g. logger.info)
    """
    log = log if log is not None else (lambda line: None)
    files = list(files)
    if not files:
        raise ValueError("no configuration files to measure")
    steps = steps_of(files, comm)
    order = sorted(range(len(files)), key=lambda i: (steps[i] is None, steps[i] if steps[i] is not None else 0, i))
    if len({s for s in steps if s is not None}) < len([s for s in steps if s is not None]):
        log("WARNING: several files have the same step: they come from different runs?")
    writer, model, records = MeasurementFile(filename), _Model(comm), []
    writer.start(model)
    try:
        for k, i in enumerate(order):
            log("measuring {} ({}/{})".format(files[i], k + 1, len(files)))
            values = measure(files[i])
            step = int(steps[i]) if steps[i] is not None else k
            writer.measurement(model, step, values, None)
            records.append((step, values))
    finally:
        writer.finish(model, len(records))
    log("{} measurements written to {}".format(len(records), filename))
    return records
