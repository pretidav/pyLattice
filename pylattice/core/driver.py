"""The run of a Monte Carlo simulation from the command line, the same for every theory:

    PP = Parser(argv)                                   # the options
    CC = CartesianComm(mpigrid=PP.mpigrid)
    LL = Logger(debug=PP.debug, logfile=PP.logfile is not None, filename=PP.logfile or 'run.log')
    LL.banner(PP, CC.comm)                              # logo and boxes with the system and run information
    ... build the model of the theory (loading PP.start_from if given) ...
    run_simulation(model, PP, LL, prefix='ising')       # the sweeps, with their logs and files, and the summary

run_simulation adds the callbacks every run has (the measurement file, the checkpoints if asked, the run log
lines on the terminal and in the log file), runs the sweeps with pylattice.core.montecarlo, and shows a box
with the summary. The model needs what montecarlo.run needs (sweep, observables); if it has its own run_mc it is
used. A collective call: every rank makes it.
"""
import time

from pylattice.core import montecarlo
from pylattice.core.callbacks import Checkpointer, MeasurementFile, RunLog
from pylattice.core.cli import info_panels


def run_simulation(model, parser, logger, prefix, measurement_file='meas.json', callbacks=()):
    """Runs parser.steps sweeps of the model, measuring after each, and returns the History.

    parser     the Parser of the run (steps, checkpoint_every, runlog_every, progress, logfile)
    logger     the Logger of the run (pylattice.core.cli.Logger)
    prefix     the checkpoints are written as checkpoints/<prefix>_<sweep>.hdf5
    callbacks  more callbacks, after the standard ones
    """
    standard = [MeasurementFile(measurement_file)]                       # every measurement, as soon as it is made
    if parser.checkpoint_every:
        standard.append(Checkpointer(parser.checkpoint_every, directory='checkpoints', prefix=prefix, final=True))
    if parser.runlog_every:
        standard.append(RunLog(parser.runlog_every, logger=logger))      # one line per interval: terminal and log file
    callbacks = standard + list(callbacks)

    start = time.perf_counter()
    if hasattr(model, 'run_mc'):
        model.run_mc(steps=parser.steps, online_obs=True, callbacks=callbacks, progress=parser.progress)
        history = model.history
    else:
        history = montecarlo.run(model, parser.steps, callbacks=callbacks, progress=parser.progress)
    elapsed = time.perf_counter() - start

    acceptance = [a for a in history.acceptance if a is not None]
    logger.show(info_panels([('done', [
        ('sweeps', parser.steps),
        ('wall time', '{:.1f} s ({:.1f} sweeps/s)'.format(elapsed, parser.steps / elapsed if elapsed else 0.0)),
        ('acceptance', '{:.3f}'.format(sum(acceptance) / len(acceptance)) if acceptance else 'n/a'),
        ('measurements', '{}  (print and plot them with: python -m pylattice.utils.plot_meas {})'.format(
            measurement_file, measurement_file)),
        ('checkpoints', 'checkpoints/' if parser.checkpoint_every else 'off'),
        ('log file', parser.logfile or 'off'),
    ])], exact=True)[0])
    return history
