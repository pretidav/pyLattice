"""Measures Ising configurations that are stored on disk (the checkpoints of a run) and writes a
<run name>_meas_offline.json (the run name of the checkpoints: ising_grid<grid>_mpi<mpigrid>_b<beta>), as a run does, so a measurement can be made again or on another set of configurations
without running the Monte Carlo:

    mpiexec -np 4 python -m pylattice.theories.spin.ising_meas checkpoints/ising_grid10x10_mpi2x2_b1.5_*.hdf5 --mpigrid=2x2

Every file is checked first against the Ising field to fill (type, global lattice, dataset, only
spins +1 and -1) and the structure found is logged; a file that does not match stops the run, with
what was measured so far kept in the output. The lattice is that of the files; --mpigrid says how
to split it over the ranks (default: along the first direction), which need not be the layout of
the run that wrote them.

Observables, per site: m (magnetization), abs_m, m2 (m squared), e (energy of H = -sum_<ij> s_i s_j),
e2 (e squared). Averages over the configurations, susceptibility and specific heat follow from them:
pylattice.utils.plot_meas prints the mean, error and autocorrelation time of each.
"""
import argparse
import math

from mpi4py import MPI

from pylattice.core.cli import Logger, offline_name
from pylattice.core.hdf5io import describe
from pylattice.core.measure import measure_files
from pylattice.core.parallel import CartesianComm
from pylattice.theories.spin.ising import IsingField, IsingModel


def ising_observables(model):
    """the observables of the configuration in model.field (a collective call)"""
    m, e = float(model.magnetization()), float(model.energy())
    return {'m': m, 'abs_m': abs(m), 'm2': m * m, 'e': e, 'e2': e * e}


def measure_ising(path, cartesiancomm, grid, log=None, beta=1.0):
    """loads the configuration in path (checking it is an Ising one for this lattice) and measures it.
    The energy does not depend on beta; it is only what the model is built with."""
    field = IsingField(grid=grid, cartesiancomm=cartesiancomm, initialization='cold')
    field.load(path, log=log, label='check')
    return ising_observables(IsingModel(field=field, logger=None, beta=beta))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Measure Ising configurations stored in HDF5 files.',
                                     epilog='example: mpiexec -np 4 python -m pylattice.theories.spin.ising_meas '
                                            'checkpoints/ising_grid10x10_mpi2x2_b1.5_*.hdf5 --mpigrid=2x2')
    parser.add_argument('files', nargs='+', metavar='FILE.hdf5', help='configurations (checkpoints) to measure')
    parser.add_argument('--mpigrid', default=None, help='number of ranks along each direction, AxBx... '
                        '(default: all the ranks along the first direction)')
    parser.add_argument('--out', default=None, help='the measurements file (default: the common run name of the files '
                                                    'followed by _meas_offline.json)')
    parser.add_argument('--debug', action='store_true', help='log at DEBUG level, from every rank')
    parser.add_argument('--logfile', action='store_true', help='also write the log, to <run name>_meas_offline.log')
    args = parser.parse_args(argv)

    comm = MPI.COMM_WORLD
    out = args.out or offline_name(args.files) + '.json'
    log = Logger(debug=args.debug, logfile=args.logfile, filename=offline_name(args.files) + '.log').log
    try:
        attrs = describe(args.files[0], comm)['attrs']
        total = [int(g) for g in attrs['grid']]
        mpigrid = ([int(a) for a in args.mpigrid.split('x')] if args.mpigrid
                   else [comm.Get_size()] + [1] * (len(total) - 1))
        if len(mpigrid) != len(total) or math.prod(mpigrid) != comm.Get_size():
            raise ValueError("--mpigrid {} does not fit {} dimensions and {} ranks".format(
                'x'.join(map(str, mpigrid)), len(total), comm.Get_size()))
        if any(t % m for t, m in zip(total, mpigrid)):
            raise ValueError("the lattice {} cannot be split over the ranks {}".format(
                'x'.join(map(str, total)), 'x'.join(map(str, mpigrid))))
        grid = [t // m for t, m in zip(total, mpigrid)]
        cartesiancomm = CartesianComm(mpigrid=mpigrid)
        log.info("lattice {} on {} ranks ({}), local grid {}; {} files; output {}".format(
            'x'.join(map(str, total)), comm.Get_size(), 'x'.join(map(str, mpigrid)),
            'x'.join(map(str, grid)), len(args.files), out))
        measure_files(args.files, lambda path: measure_ising(path, cartesiancomm, grid, log.info),
                      comm, filename=out, log=log.info)
    except (ValueError, OSError, KeyError) as error:
        log.error(str(error))
        comm.Barrier()
        raise SystemExit(1)
    log.info('done: print and plot with: python -m pylattice.utils.plot_meas {}'.format(out))


if __name__ == '__main__':
    main()
