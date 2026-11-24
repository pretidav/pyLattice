"""Measures SU(2) gauge configurations that are stored on disk (the checkpoints of a run) and writes a meas.json, as a
run does, so the expensive measurements need not be made during the Monte Carlo:

    mpiexec -np 4 python -m pylattice.theories.gauge.su2_meas checkpoints/su2_*.hdf5 --beta=2.3 --mpigrid=2x2x1x1

Every file is checked first against the SU(2) field to fill (type, global lattice, dataset, only SU(2) matrices) and the
structure found is logged; a file that does not match stops the run, with what was measured so far kept in the output. The
lattice is that of the files; --mpigrid says how to split it over the ranks (default: along the first direction), which
need not be the layout of the run that wrote them. --beta is the coupling of the run: the action depends on it and the
checkpoints do not store it.

Observables, per configuration:
    plaq_re, plaq_im  the average plaquette (1/2) tr P (the imaginary part is zero up to rounding)
    action            the Wilson action per site, summed over the planes
    plaq_s, plaq_t    the plaquette in the spatial planes and in the planes with the last direction, the time
                      (plaq_s only for 3 or more dimensions)
    polyakov          the Polyakov loop along the last direction (1/2 tr of the product of its links around the lattice),
    polyakov_abs      averaged over the sites, and its absolute value
    W12, W22, ...     the Wilson loops R x T with R <= T <= --max-loop (R = T = 1 is the plaquette), written W<R><T>,
                      (1/2) tr averaged over the sites and the orientations
    creutz22, ...     the Creutz ratios chi(R,R) = -ln( W(R,R) W(R-1,R-1) / W(R-1,R)^2 ), for R = 2 .. --max-loop, from
                      which the string tension is read (nan for a configuration where the argument is not positive: a very
                      noisy measurement; average the loops first and take the ratio of the averages, not the reverse)
pylattice.utils.plot_meas prints the mean, error and autocorrelation time of each.
"""
import argparse
import math

from mpi4py import MPI

from pylattice.core.cli import Logger
from pylattice.core.parallel import CartesianComm
from pylattice.theories.gauge.su2 import load, plaquette_observables
from pylattice.theories.gauge.sun import SU2Field


def _check_max_loop(field, max_loop):
    extents = [field._extent(mu) for mu in range(field.Nd)]
    if not 1 <= max_loop <= 9:
        raise ValueError("the largest loop must be between 1 and 9 sites, got {}".format(max_loop))
    if max_loop > min(extents):
        raise ValueError("a {0}x{0} loop does not fit in the lattice {1}".format(max_loop, 'x'.join(map(str, extents))))


def loop_observables(field, max_loop=2):
    """The expensive observables of the SU2Field `field` (a collective call): the spatial and temporal plaquettes, the
    Polyakov loop along the last direction, the Wilson loops R x T with R <= T <= max_loop and the Creutz ratios (see the
    module docstring)."""
    _check_max_loop(field, max_loop)
    Nd = field.Nd
    t = Nd - 1                                                          # the time: the last direction
    pairs = [(mu, nu) for mu in range(Nd) for nu in range(Nd) if mu != nu]
    out = {}
    if Nd >= 3:
        out['plaq_s'] = float(field.Plaq([p for p in pairs if t not in p]).real)
    out['plaq_t'] = float(field.Plaq([p for p in pairs if t in p]).real)
    polyakov = field.Polyakov(t)
    out['polyakov'], out['polyakov_abs'] = float(polyakov.real), float(abs(polyakov))
    W = {(1, 1): field.Plaq().real}
    for T in range(1, max_loop + 1):
        for R in range(1, T + 1):
            if (R, T) != (1, 1):
                W[R, T] = field.Wilson(R, T).real
                out['W{}{}'.format(R, T)] = float(W[R, T])
    for R in range(2, max_loop + 1):
        ratio = W[R, R] * W[R - 1, R - 1] / (W[R - 1, R] ** 2) if W[R - 1, R] else float('nan')
        out['creutz{}{}'.format(R, R)] = float(-math.log(ratio)) if ratio > 0 else float('nan')
    return out


def measure_su2(path, cartesiancomm, grid, beta, max_loop=2, log=None):
    """loads the configuration in path (checking it is an SU(2) one for this lattice) and measures it: the plaquette
    observables of the run and the loop observables"""
    field = SU2Field(grid=grid, cartesiancomm=cartesiancomm, Nd=len(grid))
    load(field, path, log=log, label='check')
    return {**plaquette_observables(field, beta), **loop_observables(field, max_loop)}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Measure SU(2) gauge configurations stored in HDF5 files.',
                                     epilog='example: mpiexec -np 4 python -m pylattice.theories.gauge.su2_meas '
                                            'checkpoints/su2_*.hdf5 --beta=2.3 --mpigrid=2x2x1x1')
    parser.add_argument('files', nargs='+', metavar='FILE.hdf5', help='configurations (checkpoints) to measure')
    parser.add_argument('--beta', type=float, required=True,
                        help='the coupling of the run that made the files (for the action)')
    parser.add_argument('--max-loop', type=int, default=2, metavar='L',
                        help='measure the Wilson loops R x T with R <= T <= L, and the Creutz ratios up to L x L')
    parser.add_argument('--mpigrid', default=None, help='number of ranks along each direction, AxBx... '
                        '(default: all the ranks along the first direction)')
    parser.add_argument('--out', default='meas.json', help='the measurements file')
    parser.add_argument('--debug', action='store_true', help='log at DEBUG level, from every rank')
    parser.add_argument('--logfile', action='store_true', help='also write the log to su2_meas.log')
    args = parser.parse_args(argv)
    from pylattice.core.hdf5io import describe                        # h5py is needed only to read the checkpoints
    from pylattice.core.measure import measure_files
    if not math.isfinite(args.beta) or args.beta < 0:
        parser.error('--beta must be finite and >= 0')

    comm = MPI.COMM_WORLD
    log = Logger(debug=args.debug, logfile=args.logfile, filename='su2_meas.log').log
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
        if not 1 <= args.max_loop <= min(9, min(total)):
            raise ValueError("--max-loop must be between 1 and {} for a lattice {}".format(
                min(9, min(total)), 'x'.join(map(str, total))))
        grid = [t // m for t, m in zip(total, mpigrid)]
        cartesiancomm = CartesianComm(mpigrid=mpigrid)
        log.info("lattice {} on {} ranks ({}), local grid {}; beta {}; loops up to {}x{}; {} files; output {}".format(
            'x'.join(map(str, total)), comm.Get_size(), 'x'.join(map(str, mpigrid)), 'x'.join(map(str, grid)),
            args.beta, args.max_loop, args.max_loop, len(args.files), args.out))
        measure_files(args.files, lambda path: measure_su2(path, cartesiancomm, grid, args.beta, args.max_loop, log.info),
                      comm, filename=args.out, log=log.info)
    except (ValueError, OSError, KeyError) as error:
        log.error(str(error))
        comm.Barrier()
        raise SystemExit(1)
    log.info('done: print and plot with: python -m pylattice.utils.plot_meas {}'.format(args.out))


if __name__ == '__main__':
    main()
