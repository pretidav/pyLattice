"""The run of a Monte Carlo simulation from the command line, the same for every theory:

    PP = Parser(argv)                                   # the options
    CC = CartesianComm(mpigrid=PP.mpigrid)
    LL = Logger(debug=PP.debug, logfile=PP.log_filename() is not None, filename=PP.log_filename() or 'run.log')
    LL.banner(PP, CC.comm)                              # logo and boxes with the system and run information
    ... build the model of the theory (loading PP.start_from if given) ...
    run_simulation(model, PP, LL)                       # the sweeps, with their logs and files, and the summary

run_simulation adds the callbacks every run has (the measurement file, the checkpoints if asked, the run log
lines on the terminal and in the log file), runs the sweeps with pylattice.core.montecarlo, and shows a box
with the summary. The model needs what montecarlo.run needs (sweep, observables); if it has its own run_mc it is
used. A collective call: every rank makes it.
"""
import time

from pylattice.core import montecarlo
from pylattice.core.callbacks import Checkpointer, MeasurementFile, RunLog
from pylattice.core.cli import info_panels


def run_simulation(model, parser, logger, prefix=None, measurement_file=None, callbacks=()):
    """Runs parser.steps sweeps of the model, measuring after each, and returns the History.

    parser     the Parser of the run (steps, checkpoint_every, runlog_every, progress, logfile); it names the files:
               <theory>_grid<grid>_mpi<mpigrid>_b<beta> (Parser.run_name) followed by _meas.json for the measurements,
               _<sweep>.hdf5 for the checkpoints (in ./checkpoints), .log for the log
    logger     the Logger of the run (pylattice.core.cli.Logger)
    prefix     another start of the names of the checkpoints (default: the run name of the parser)
    measurement_file  another name of the measurements file (default: that of the parser)
    callbacks  more callbacks, after the standard ones
    """
    measurement_file = measurement_file or parser.meas_filename()
    prefix = prefix or parser.checkpoint_prefix()
    standard = [MeasurementFile(measurement_file)]                       # every measurement, as soon as it is made
    if parser.checkpoint_every:
        standard.append(Checkpointer(parser.checkpoint_every, directory='checkpoints', prefix=prefix, final=True))
    if parser.runlog_every:
        standard.append(RunLog(parser.runlog_every, logger=logger))      # one line per interval: terminal and log file
    callbacks = standard + list(callbacks)

    start = time.perf_counter()
    run = dict(thermalization=parser.thermalization, measure_every=parser.measure_every, callbacks=callbacks,
               progress=parser.progress)
    if hasattr(model, 'run_mc'):
        model.run_mc(steps=parser.steps, online_obs=True, **run)
        history = model.history
    else:
        history = montecarlo.run(model, parser.steps, **run)
    elapsed = time.perf_counter() - start

    acceptance = [a for a in history.acceptance if a is not None]
    sweeps = parser.thermalization + parser.steps
    logger.show(info_panels([('done', [
        ('sweeps', '{} ({} of thermalization, {} measured)'.format(sweeps, parser.thermalization, parser.steps)
         if parser.thermalization else sweeps),
        ('wall time', '{:.1f} s ({:.1f} sweeps/s)'.format(elapsed, sweeps / elapsed if elapsed else 0.0)),
        ('acceptance', '{:.3f}'.format(sum(acceptance) / len(acceptance)) if acceptance else 'n/a'),
        ('measurements', '{}  (print and plot them with: python -m pylattice.utils.plot_meas {})'.format(
            measurement_file, measurement_file)),
        ('checkpoints', 'checkpoints/{}_<sweep>.hdf5'.format(prefix) if parser.checkpoint_every else 'off'),
        ('log file', parser.log_filename() or 'off'),
    ])], exact=True)[0])
    return history
