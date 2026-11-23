import argparse
import logging
import math

from rich.logging import RichHandler


def _steps(text):
    try:
        steps = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("'{}' is not an integer".format(text))
    if steps < 0:
        raise argparse.ArgumentTypeError("the number of steps cannot be negative, got {}".format(steps))
    return steps


def _beta(text):
    try:
        beta = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError("'{}' is not a number".format(text))
    if not math.isfinite(beta):
        raise argparse.ArgumentTypeError("beta must be finite, got {}".format(text))
    return beta


def _seed(text):
    try:
        seed = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("'{}' is not an integer".format(text))
    if not 0 <= seed < 2**32:                         # what numpy.random.seed accepts
        raise argparse.ArgumentTypeError("the seed must be between 0 and 2**32-1, got {}".format(seed))
    return seed


class Parser():
    """Command line of the simulations. `argv` is a list of arguments (default: sys.argv[1:])."""

    def __init__(self, argv=None):
        self.args = self.parser(argv)
        self.grid = self.args['grid']
        self.mpigrid = self.args['mpigrid']
        self.seed = self.args['seed']
        self.debug = self.args['debug']
        self.logfile = self.args['logfile']
        self.beta = self.args['beta']
        self.steps = self.args['steps']

    def parser(self, argv=None):
        parser = argparse.ArgumentParser(
            description='Monte Carlo simulation on a lattice split over MPI ranks.',
            epilog='example: mpiexec -np 4 python ising_model.py --grid=20x20 --mpigrid=2x2 --beta=1.5 --steps=2000',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--grid', type=self._lattice_arg, default='4x4',
                            help='lattice size of each MPI rank, AxBx... (the global lattice is grid*mpigrid)')
        parser.add_argument('--mpigrid', type=self._lattice_arg, default='1x2',
                            help='number of MPI ranks along each direction, AxBx... '
                                 '(run with mpiexec -np <product of the sizes>)')
        parser.add_argument('--steps', type=_steps, default=100, help='number of Monte Carlo steps')
        parser.add_argument('--beta', type=_beta, default=0.5, help='inverse temperature')
        parser.add_argument('--seed', type=_seed, default=0, help='seed of the random number generators')
        parser.add_argument('--debug', action='store_true',
                            help='log at DEBUG level, from every rank')
        parser.add_argument('--logfile', action='store_true',
                            help='also write the log to a file (see Logger)')
        args = vars(parser.parse_args(argv))
        if len(args['grid']) != len(args['mpigrid']):
            parser.error('--grid has {} dimensions but --mpigrid has {}'.format(
                len(args['grid']), len(args['mpigrid'])))
        return args

    def parse_latt(self, par):
        """'2x3x4' -> [2, 3, 4]. Raises ValueError unless every size is an integer >= 1."""
        dims = [int(a) for a in str(par).split('x')]
        if any(d < 1 for d in dims):
            raise ValueError("every size must be at least 1, got '{}'".format(par))
        return dims

    def _lattice_arg(self, text):
        try:
            return self.parse_latt(text)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "'{}' is not of the form AxBx... with integer sizes of at least 1".format(text))


class _RankFilter(logging.Filter):
    """Adds the MPI rank to the records and decides which ranks speak: rank 0 says everything,
    the other ranks only warnings and errors, unless debug is on (then every rank says everything)."""

    def __init__(self, rank, size, debug):
        super().__init__()
        self.rank, self.size, self.debug = rank, size, debug

    def filter(self, record):
        record.rankprefix = '[rank {}] '.format(self.rank) if self.size > 1 else ''
        return self.rank == 0 or self.debug or record.levelno >= logging.WARNING


class Logger():
    """Logger 'pylattice' with colored output (rich) and, with logfile=True, also a plain-text file.

    debug: DEBUG level on every rank (otherwise INFO from rank 0, WARNING and above from any rank).
    console: a rich Console to write to (default: the standard rich console, which uses colors when
             the output is a terminal; set NO_COLOR=1 / FORCE_COLOR=1 to switch them off / on).
             Under mpiexec every rank's output is a terminal, so redirected output keeps its color
             codes (and mpiexec does not pass these variables to the ranks): use logfile=True for a
             plain-text copy.
    rank, size: MPI rank and number of ranks (default: asked to mpi4py, or 0 and 1 without it).
    Creating a Logger again replaces the handlers of the previous one; the root logger is not touched.
    """

    def __init__(self, debug, logfile, filename='pylattice.log', console=None, rank=None, size=None):
        self.debug = debug
        self.logfile = logfile
        self.filename = filename
        self.console = console
        self.rank, self.size = self._mpi_rank_size(rank, size)
        self.log = self.get_logger(debug=self.debug, logfile=self.logfile)

    @staticmethod
    def _mpi_rank_size(rank, size):
        if rank is None or size is None:
            try:
                from mpi4py import MPI
                rank, size = MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()
            except ImportError:
                rank, size = 0, 1
        return rank, size

    def get_logger(self, debug, logfile):
        logger = logging.getLogger('pylattice')
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        logger.propagate = False
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        rank_filter = _RankFilter(self.rank, self.size, debug)

        console_handler = RichHandler(console=self.console, show_path=False, rich_tracebacks=True,
                                   log_time_format='[%H:%M:%S]')
        console_handler.setFormatter(logging.Formatter('%(rankprefix)s%(message)s'))
        console_handler.addFilter(rank_filter)
        logger.addHandler(console_handler)

        if logfile:
            file_handler = logging.FileHandler(self.filename)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s:  %(rankprefix)s%(message)s'))
            file_handler.addFilter(rank_filter)
            logger.addHandler(file_handler)
        return logger
