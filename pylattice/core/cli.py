import argparse
import logging
import math
import os
import platform
import re
import socket
import subprocess
import sys
import time

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

WIDTH = 110                                       # the widest box, on the terminal and in the file

LOGO = r"""
__| |_________| |__
__   _________   __
  | |         | |  
  | |pyLattice| |  
__| |_________| |__
__   _________   __
  | |         | |  
"""

def _steps(text):
    try:
        steps = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("'{}' is not an integer".format(text))
    if steps < 0:
        raise argparse.ArgumentTypeError("the number of steps cannot be negative, got {}".format(steps))
    return steps


def _every(text):
    try:
        every = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("'{}' is not an integer".format(text))
    if every < 0:
        raise argparse.ArgumentTypeError("a number of sweeps cannot be negative, got {}".format(every))
    return every


def format_beta(beta):
    """beta as it appears in the names of the files: the shortest exact form, 1.5 -> '1.5', 10.0 -> '10'"""
    text = repr(float(beta))
    return text[:-2] if text.endswith('.0') else text


def offline_name(files):
    """The start of the names of the files of an offline measurement of the checkpoint files: their common run name, that is
    the file name without '_<step>.hdf5' (<theory>_grid<grid>_mpi<mpigrid>_b<beta> for the files of a run), followed by
    _meas_offline.json for the measurements and _meas_offline.log for the log; 'measurements' when the files have no common
    name. The name says 'offline' so that it does not overwrite the measurements made during the run."""
    stems = [re.sub(r'_\d+$', '', os.path.splitext(os.path.basename(f))[0]) for f in files]
    base = stems[0] if len(set(stems)) == 1 else os.path.commonprefix(stems).rstrip('_')
    return (base or 'measurements') + '_meas_offline'


def _positive(text):
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("'{}' is not an integer".format(text))
    if n < 1:
        raise argparse.ArgumentTypeError("this must be at least 1, got {}".format(n))
    return n


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
    """Command line of the simulations. `argv` is a list of arguments (default: sys.argv[1:]).

    A theory with options of its own subclasses it: add_options(parser) adds them to the argparse parser (and may
    change the defaults of the common ones with parser.set_defaults), extra_info() lists them, as [(name, value)], in
    the 'run' box of the banner. Their values are in self.args."""

    def __init__(self, argv=None, theory=None):
        self.theory = theory                       # the name of the theory in the names of the files (see run_name)
        self.args = self.parser(argv)
        self.grid = self.args['grid']
        self.mpigrid = self.args['mpigrid']
        self.seed = self.args['seed']
        self.debug = self.args['debug']
        self.logfile = self.args['logfile']
        self.progress = self.args['progress']
        self.beta = self.args['beta']
        self.steps = self.args['steps']
        self.thermalization = self.args['thermalization']
        self.measure_every = self.args['measure_every']
        self.checkpoint_every = self.args['checkpoint_every']
        self.runlog_every = self.args['runlog_every']
        self.start_from = self.args['start_from']

    def parser(self, argv=None):
        parser = argparse.ArgumentParser(
            description='Monte Carlo simulation on a lattice split over MPI ranks.',
            epilog='example: mpiexec -np 4 python -m pylattice.theories.spin.ising --grid=20x20 --mpigrid=2x2 --beta=1.5 --steps=2000',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--grid', type=self._lattice_arg, default='4x4',
                            help='lattice size of each MPI rank, AxBx... (the global lattice is grid*mpigrid)')
        parser.add_argument('--mpigrid', type=self._lattice_arg, default='1x2',
                            help='number of MPI ranks along each direction, AxBx... '
                                 '(run with mpiexec -np <product of the sizes>)')
        parser.add_argument('--steps', type=_steps, default=100,
                            help='number of Monte Carlo steps (sweeps) that are measured, after the thermalization')
        parser.add_argument('--thermalization', type=_every, default=0, metavar='N',
                            help='sweeps made first and not measured, to reach equilibrium (they count in the '
                                 'sweeps of the run log and of the checkpoints)')
        parser.add_argument('--measure-every', type=_positive, default=1, metavar='N',
                            help='measure the observables every N sweeps (the first measurement is the '
                                 'configuration after the thermalization)')
        parser.add_argument('--beta', type=_beta, default=0.5, help='inverse temperature')
        parser.add_argument('--seed', type=_seed, default=0, help='seed of the random number generators')
        parser.add_argument('--checkpoint-every', type=_every, default=0, metavar='N',
                            help='save the configuration to ./checkpoints every N sweeps, and at the end (0: never); the files are '
                                 '<theory>_grid<grid>_mpi<mpigrid>_b<beta>_<sweep>.hdf5')
        parser.add_argument('--runlog-every', type=_every, default=None, metavar='N',
                            help='log a line every N sweeps, on the terminal and in the log file: step, observables, '
                                 'memory, cpu, sweeps per second (0: never; default: about 20 lines per run)')
        parser.add_argument('--start-from', default=None, metavar='FILE.hdf5',
                            help='start from the configuration saved in this HDF5 file (a checkpoint) instead '
                                 'of a random one; its lattice must match --grid x --mpigrid, the layout '
                                 'of the ranks may differ')
        parser.add_argument('--debug', action='store_true',
                            help='log at DEBUG level, from every rank')
        parser.add_argument('--logfile', default=None, metavar='FILE',
                            help="the log file: logo, system and run summary, messages and one line per "
                                 "--runlog-every sweeps, the same as on the terminal but in plain text, appended "
                                 "to if it exists (default: <theory>_grid<grid>_mpi<mpigrid>_b<beta>.log; 'off': no file)")
        parser.add_argument('--progress', action='store_true',
                            help='also draw a progress bar (rank 0)')
        self.add_options(parser)
        args = vars(parser.parse_args(argv))
        if args['runlog_every'] is None:
            args['runlog_every'] = max(1, (args['steps'] + args['thermalization']) // 20) if args['steps'] + args['thermalization'] else 0
        if args['logfile'] is not None and args['logfile'].lower() in ('off', 'none', ''):
            args['logfile'] = False                 # None: the default name, False: no file
        if args['start_from'] is not None and not os.path.isfile(args['start_from']):
            parser.error('--start-from: no such file: {}'.format(args['start_from']))
        if len(args['grid']) != len(args['mpigrid']):
            parser.error('--grid has {} dimensions but --mpigrid has {}'.format(
                len(args['grid']), len(args['mpigrid'])))
        return args

    def run_name(self):
        """The start of the name of every file of the run, <theory>_grid<grid>_mpi<mpigrid>_b<beta>, e.g.
        ising_grid20x20_mpi2x2_b1.5: the log is run_name + '.log', the measurements run_name + '_meas.json', the checkpoints
        run_name + '_<sweep>.hdf5', so that a file says which run it comes from. The grid and the mpigrid are as given
        (the lattice of each rank, and the ranks along each direction). Without a theory the name starts with 'run'."""
        return '{}_grid{}_mpi{}_b{}'.format(self.theory or 'run', 'x'.join(map(str, self.grid)),
                                            'x'.join(map(str, self.mpigrid)), format_beta(self.beta))

    def log_filename(self):
        """the log file: --logfile if given, else run_name + '.log'; None if the log file is switched off (--logfile=off)"""
        if self.logfile is False:
            return None
        return self.logfile if self.logfile else self.run_name() + '.log'

    def meas_filename(self):
        """the measurements file of the run, run_name + '_meas.json'"""
        return self.run_name() + '_meas.json'

    def checkpoint_prefix(self):
        """the checkpoints are written as checkpoints/<checkpoint_prefix>_<sweep>.hdf5, the prefix being run_name"""
        return self.run_name()

    def add_options(self, parser):
        """the options of a theory, added to the argparse `parser` (none here)"""

    def extra_info(self):
        """the options of a theory as [(name, value), ...] for the banner (none here)"""
        return []

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
    """Logger 'pylattice' with colored output (rich) and, with logfile=True, also a plain-text file (filename).

    Besides the messages (log.info, ...), show(renderable) prints a rich object (the boxes of the banner, a line of
    the run log) on the terminal and, in plain text, in the file. Only rank 0 shows anything.

    debug: DEBUG level on every rank (otherwise INFO from rank 0, WARNING and above from any rank).
    console: a rich Console to write to (default: the standard rich console, which uses colors when
             the output is a terminal; set NO_COLOR=1 / FORCE_COLOR=1 to switch them off / on).
             Under mpiexec every rank's output is a terminal, so redirected output keeps its color
             codes (and mpiexec does not pass these variables to the ranks): use logfile=True for a
             plain-text copy.
    rank, size: MPI rank and number of ranks (default: asked to mpi4py, or 0 and 1 without it).
    Creating a Logger again replaces the handlers of the previous one; the root logger is not touched.
    """

    def __init__(self, debug, logfile, filename='run.log', console=None, rank=None, size=None):
        self.debug = debug
        self.logfile = logfile
        self.filename = filename
        self.console = console if console is not None else Console()
        self.file_console = None
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
            file_handler = logging.FileHandler(self.filename, encoding='utf-8')
            self._file_handler = file_handler
            self.file_console = Console(file=file_handler.stream, width=WIDTH, color_system=None, force_terminal=False,
                                        highlight=False, emoji=False)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s:  %(rankprefix)s%(message)s'))
            file_handler.addFilter(rank_filter)
            logger.addHandler(file_handler)
        else:
            self.file_console = None
        return logger

    def show(self, renderable):
        """Prints a rich object (Text, Panel, ...) on the terminal, in color, and in the file as plain text. Rank 0 only."""
        if self.rank != 0:
            return
        line = isinstance(renderable, Text)                 # a line is never wrapped or cut
        self.console.print(renderable, highlight=False, soft_wrap=line)
        if self.file_console is not None:
            self.file_console.print(renderable, soft_wrap=line)
            self._file_handler.flush()

    def banner(self, parser=None, comm=None):
        """The logo and the information of run_info(), one box per section. A collective call: every rank makes it."""
        sections = run_info(parser, comm)
        self.show(Text(LOGO.strip('\n'), style='cyan'))
        for panel in info_panels(sections):
            self.show(panel)


def _version(module):
    try:
        return __import__(module).__version__
    except Exception:
        return 'not installed'


def _git_revision():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        run = lambda *a: subprocess.run(('git',) + a, cwd=here, capture_output=True, text=True, timeout=5).stdout.strip()
        revision = run('rev-parse', '--short', 'HEAD')
        if not revision:
            return 'unknown'
        return revision + (' (uncommitted changes)' if run('status', '--porcelain') else '')
    except Exception:
        return 'unknown'


def run_info(parser=None, comm=None, argv=None):
    """The information on the run and the system, as a list of (section, [(name, value), ...]).

    parser  the Parser of the run: its arguments and the size of the lattice are listed
    comm    the MPI communicator (default COMM_WORLD). This is a collective call: every rank must make it,
            because the hosts of all the ranks are gathered.
    """
    from mpi4py import MPI
    comm = comm if comm is not None else MPI.COMM_WORLD
    hosts = {}
    for rank, host in enumerate(comm.allgather(socket.gethostname())):
        hosts.setdefault(host, []).append(rank)

    try:
        import psutil
        memory = '{:.1f} GB'.format(psutil.virtual_memory().total / 2 ** 30)
    except Exception:
        memory = 'unknown'
    try:
        import h5py
        h5 = '{} (HDF5 {}, parallel: {})'.format(h5py.__version__, h5py.version.hdf5_version, h5py.get_config().mpi)
    except Exception:
        h5 = 'not installed'
    mpi = MPI.Get_library_version().strip().split(',')[0].strip()      # e.g. 'Open MPI v5.0.11'

    system = [
        ('started', time.strftime('%Y-%m-%d %H:%M:%S')),
        ('hosts', '; '.join('{} (ranks {})'.format(h, _ranges(r)) for h, r in hosts.items())),
        ('os', platform.platform()),
        ('architecture', '{} ({})'.format(platform.machine(), platform.processor() or 'unknown processor')),
        ('cpu cores', '{} on this host'.format(os.cpu_count())),
        ('memory', memory + ' on this host'),
        ('python', platform.python_version()),
        ('numpy', _version('numpy')),
        ('mpi4py', '{} / {}'.format(_version('mpi4py'), mpi)),
        ('h5py', h5),
        ('pylattice', 'git ' + _git_revision()),
        ('command', ' '.join(getattr(sys, 'orig_argv', None) or ['python'] + sys.argv) if argv is None
         else ' '.join(['python'] + list(argv))),
        ('directory', os.getcwd()),
    ]
    sections = [('system', system), ('parallel', [
        ('mpi ranks', comm.Get_size()),
        ('threads', os.environ.get('OMP_NUM_THREADS', 'not set (OMP_NUM_THREADS)')),
    ])]
    if parser is not None:
        grid, mpigrid = parser.grid, parser.mpigrid
        total = [g * m for g, m in zip(grid, mpigrid)]
        sites = math.prod(total)
        sections.append(('lattice', [
            ('local grid (per rank)', 'x'.join(map(str, grid))),
            ('mpigrid', 'x'.join(map(str, mpigrid))),
            ('global lattice', '{} = {} sites'.format('x'.join(map(str, total)), sites)),
            ('sites per rank', math.prod(grid)),
        ]))
        sections.append(('run', [
            ('beta', parser.beta),
            ('steps', parser.steps),
            ('thermalization', parser.thermalization),
            ('measure every', parser.measure_every),
            ('seed', '{} (one independent stream per rank)'.format(parser.seed)),
            ('start from', parser.start_from or 'random configuration (hot start)'),
            ('checkpoint every', parser.checkpoint_every or 'off'),
            ('run log every', parser.runlog_every or 'off'),
            ('debug', parser.debug),
            ('log file', parser.log_filename() or 'off'),
            ('measurements', parser.meas_filename()),
        ] + list(parser.extra_info())))
    return sections


def _ranges(ranks):
    """[0, 1, 2, 5] -> '0-2, 5'"""
    out, start = [], 0
    for i in range(1, len(ranks) + 1):
        if i == len(ranks) or ranks[i] != ranks[i - 1] + 1:
            a, b = ranks[start], ranks[i - 1]
            out.append(str(a) if a == b else '{}-{}'.format(a, b))
            start = i
    return ', '.join(out)


def info_panels(sections, width=WIDTH, exact=False):
    """One rich Panel per section, [(title, [(name, value), ...]), ...], all of the same width: that of the longest
    line, at most `width` (exactly `width` with exact=True)."""
    key = max(len(str(name)) for _, items in sections for name, _ in items)
    natural = key + 2 + max(len(str(value)) for _, items in sections for _, value in items) + 4
    panels = []
    for title, items in sections:
        table = Table.grid(padding=(0, 2))
        table.add_column(style='dim', no_wrap=True, min_width=key)
        table.add_column(overflow='fold')
        for name, value in items:
            table.add_row(Text(str(name)), Text(str(value)))
        panels.append(Panel(table, title=Text(title, style='bold cyan'), title_align='left', border_style='cyan',
                            width=width if exact else min(natural, width), expand=True))
    return panels
