#!/usr/bin/env python
"""Runs the whole test routine of pyLattice on this machine, the same checks as the GitHub workflow
(.github/workflows/testonbuild.yml) plus the ones that need more ranks than a hosted runner has. Nothing
runs it for you: call it when you want to check that an installation (or a change) works.

    python run_tests.py              # the standard routine: single process and 2 ranks
    python run_tests.py --quick      # only the single-process tests (pytest), about 15 s
    python run_tests.py --full       # also every MPI test on 1 and 4 ranks, and the cross-layout HDF5 test
    python run_tests.py -k ising     # only the steps whose name contains "ising"
    python run_tests.py -v           # show the output of every step, not only of the failed ones
    python run_tests.py --list       # list the steps and exit

It needs pytest and, for the MPI steps, mpiexec; the steps that write checkpoints need an h5py built with
parallel HDF5 and are skipped without it (the report says so). Every step runs in a scratch directory, so no
files are left in the repository. The exit status is 1 if any step failed (a skipped step is not a failure).
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).resolve().parent
PY = sys.executable
console = Console(highlight=False)
COLOR = {'PASS': 'green', 'FAIL': 'red', 'SKIP': 'yellow'}

# ---------------------------------------------------------------------------------------------- the steps
# pytest on one process: every file that pytest collects and runs by itself
PYTEST_FILES = [
    'tests/core/test_tensors.py', 'tests/core/test_cli.py',
    'tests/core/test_discrete.py', 'tests/core/test_su2.py', 'tests/core/test_su.py',
    'tests/core/test_group_interface.py', 'tests/theories/gauge/test_gauge_sun.py',
    'tests/core/test_rng.py', 'tests/core/test_stats.py', 'tests/core/test_montecarlo.py',
    'tests/core/test_callbacks.py', 'tests/theories/spin/test_ising.py',
    'tests/theories/spin/test_ising_meas.py', 'tests/utils/test_plot_meas.py', 'tests/utils/test_run_tests.py',
]
# plain scripts: they run their tests when executed
SCRIPT_FILES = ['tests/core/test_grid.py', 'tests/core/test_grid_eo.py']
# MPI scripts: file -> (ranks of the standard routine, further ranks of --full)
MPI_FILES = {
    'tests/core/test_grid_mpi.py': ((2,), ()),                  # its rank grids are written for exactly 2 ranks
    'tests/core/test_rng.py': ((2,), (1, 4)),
    'tests/core/test_montecarlo.py': ((2,), (1, 4)),
    'tests/core/test_callbacks.py': ((2,), (1, 4)),
    'tests/theories/spin/test_ising.py': ((2,), (1, 4)),
    'tests/theories/spin/test_ising_meas.py': ((2,), (1, 4)),
}
HDF5_FILE = 'tests/core/test_hdf5io.py'                       # pytest on exactly 4 ranks (a 2x2 rank grid)
HDF5_CROSS = ('tests/core/test_hdf5io_w.py', 'tests/core/test_hdf5io_r.py')   # write on 4 ranks, read on 2


def all_files_of_the_routine():
    """every test file the routine runs (a test checks that no test file of the repository is missing)"""
    return set(PYTEST_FILES) | set(SCRIPT_FILES) | set(MPI_FILES) | {HDF5_FILE} | set(HDF5_CROSS)


class Step:
    def __init__(self, name, commands, needs_mpi=False, needs_parallel_h5=False):
        self.name, self.commands = name, commands              # commands: argument lists, run in order
        self.needs_mpi, self.needs_parallel_h5 = needs_mpi, needs_parallel_h5


def build_steps(mpiexec, quick=False, full=False, oversubscribe=False):
    def mpi(n, *args):
        return [mpiexec] + (['--oversubscribe'] if oversubscribe else []) + ['-n', str(n), PY] + list(args)

    pytest = ['-m', 'pytest', '-q', '--color=no', '-p', 'no:cacheprovider', '--rootdir', str(ROOT), '-c', str(ROOT / 'pyproject.toml')]
    steps = [Step('pytest (single process)', [[PY] + pytest + [str(ROOT / f) for f in PYTEST_FILES]])]
    if quick:
        return steps
    for f in SCRIPT_FILES:
        steps.append(Step(Path(f).name, [[PY, str(ROOT / f)]]))
    for f, (standard, extra) in MPI_FILES.items():
        for n in (standard + extra if full else standard):
            steps.append(Step('{} on {} rank{}'.format(Path(f).name, n, '' if n == 1 else 's'),
                              [mpi(n, str(ROOT / f))], needs_mpi=True))
    steps.append(Step('test_hdf5io.py on 4 ranks', [mpi(4, *pytest, str(ROOT / HDF5_FILE))],
                      needs_mpi=True, needs_parallel_h5=True))
    if full:
        w, r = HDF5_CROSS
        steps.append(Step('HDF5 cross-layout: write on 4 ranks, read on 2',
                          [mpi(4, str(ROOT / w)), mpi(2, str(ROOT / r))], needs_mpi=True, needs_parallel_h5=True))
    return steps


# ---------------------------------------------------------------------------------------------- the system
def find_mpiexec():
    for name in ('mpiexec', 'mpirun'):
        found = shutil.which(name) or shutil.which(name, path=str(Path(PY).parent))
        if found:
            return found
    return None


def parallel_h5py():
    out = subprocess.run([PY, '-c', 'import h5py; print(h5py.get_config().mpi)'], capture_output=True, text=True)
    return out.returncode == 0 and out.stdout.strip() == 'True'


def physical_cores():
    try:
        import psutil
        return psutil.cpu_count(logical=False) or os.cpu_count() or 1
    except ImportError:
        return os.cpu_count() or 1


def environment():
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT)] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    env.setdefault('MPLBACKEND', 'Agg')                        # no plot windows from the tests
    return env


def versions():
    code = ('import sys, numpy, scipy, mpi4py\n'
            'try:\n import h5py; h5 = h5py.__version__\n'
            'except ImportError:\n h5 = "not installed"\n'
            'print("python", sys.version.split()[0] + ", numpy", numpy.__version__ + ", scipy", scipy.__version__'
            ' + ", mpi4py", mpi4py.__version__ + ", h5py", h5)')
    out = subprocess.run([PY, '-c', code], capture_output=True, text=True)
    return out.stdout.strip() or (out.stderr.strip().splitlines() or ['?'])[-1]


# ---------------------------------------------------------------------------------------------- the run
def run_step(step, env, workdir, timeout):
    start, output = time.time(), []
    for command in step.commands:
        try:
            done = subprocess.run(command, cwd=workdir, env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            text = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or '')
            return 'FAIL', time.time() - start, ''.join(output) + text + '\nTIMEOUT after {} s'.format(timeout)
        output.append(done.stdout + done.stderr)
        if done.returncode != 0:
            return 'FAIL', time.time() - start, ''.join(output)
    return 'PASS', time.time() - start, ''.join(output)


def summary_of(text):
    """what to show next to PASS: the last line of a pytest run, or how many tests a script ran"""
    lines = [l for l in text.splitlines() if l.strip()]
    pytest_line = [l for l in lines if ' passed' in l and ' in ' in l]
    if pytest_line:
        return re.sub(r' in [\d.]+s.*$', '', pytest_line[-1].strip('= ').strip())      # the time is shown anyway
    ok, skipped = sum('[+]' in l for l in lines), sum('[-]' in l for l in lines)
    return ('{} tests'.format(ok) + (', {} skipped'.format(skipped) if skipped else '')) if ok or skipped else ''


def main(argv=None):
    p = argparse.ArgumentParser(description='Run the pyLattice tests on this machine.')
    p.add_argument('--quick', action='store_true', help='only the single-process tests')
    p.add_argument('--full', action='store_true', help='MPI tests on 1, 2 and 4 ranks and the cross-layout HDF5 test')
    p.add_argument('-k', metavar='TEXT', help='only the steps whose name contains TEXT')
    p.add_argument('-v', '--verbose', action='store_true', help='print the output of every step')
    p.add_argument('--list', action='store_true', help='list the steps and exit')
    p.add_argument('--timeout', type=int, default=900, help='seconds allowed to each step (default 900)')
    p.add_argument('--oversubscribe', action='store_true',
                   help='pass --oversubscribe to mpiexec (automatic when there are fewer cores than ranks)')
    args = p.parse_args(argv)

    mpiexec, cores = find_mpiexec(), physical_cores()
    oversubscribe = args.oversubscribe or cores < 4                  # test_hdf5io.py needs 4 ranks
    steps = build_steps(mpiexec or 'mpiexec', quick=args.quick, full=args.full, oversubscribe=oversubscribe)
    if args.k:
        steps = [s for s in steps if args.k.lower() in s.name.lower()]
    if args.list:
        console.print('\n'.join(escape(s.name) for s in steps))
        return 0
    if not steps:
        console.print('[red]no step matches -k {!r}[/]'.format(args.k))
        return 1

    h5 = parallel_h5py()
    try:
        from pylattice.core.cli import LOGO
        console.print('[cyan]{}[/]'.format(escape(LOGO.strip('\n'))))
    except ImportError:                                        # pylattice itself is not importable: the steps will say why
        pass
    info = Table.grid(padding=(0, 2))
    info.add_row('[dim]packages[/]', escape(versions()))
    info.add_row('[dim]mpiexec[/]', escape(mpiexec) if mpiexec else '[red]NOT FOUND[/]')
    info.add_row('[dim]cores[/]', '{} physical{}'.format(cores, '  (mpiexec --oversubscribe)' if oversubscribe else ''))
    info.add_row('[dim]parallel h5py[/]', '[green]yes[/]' if h5 else '[yellow]no: the steps that write checkpoints are skipped[/]')
    console.print(Panel(info, title='[bold]pyLattice tests[/]', border_style='cyan', expand=False))

    env, results = environment(), []
    with tempfile.TemporaryDirectory(prefix='pylattice_tests_') as scratch:
        for step in steps:
            if step.needs_mpi and mpiexec is None:
                status, seconds, text, note = 'SKIP', 0.0, '', 'mpiexec not found'
            elif step.needs_parallel_h5 and not h5:
                status, seconds, text, note = 'SKIP', 0.0, '', 'no parallel h5py'
            else:
                with console.status('[cyan]running[/] {}'.format(escape(step.name))):         # a spinner on a terminal
                    status, seconds, text = run_step(step, env, tempfile.mkdtemp(dir=scratch), args.timeout)
                note = summary_of(text) if status == 'PASS' else ''
            color = COLOR[status]
            console.print(' [bold {c}]{s:<4}[/]  {n:<48} [dim]{t:>6.1f}s[/]  [{c}]{note}[/]'.format(
                c=color, s=status, n=escape(step.name), t=seconds, note=escape(note)))
            if text and (status == 'FAIL' or args.verbose):
                shown = text if args.verbose else '\n'.join(text.splitlines()[-60:])
                console.print(Panel(escape(shown), title=escape(step.name), border_style=color, style='dim',
                                    subtitle='last 60 lines' if status == 'FAIL' and not args.verbose else None))
            results.append((status, step.name))

    failed = [name for status, name in results if status == 'FAIL']
    count = lambda k: sum(s == k for s, _ in results)
    console.print('\n[bold green]{} passed[/], [bold {}]{} failed[/], [bold yellow]{} skipped[/]'.format(
        count('PASS'), 'red' if failed else 'green', count('FAIL'), count('SKIP')))
    for name in failed:
        console.print('[red]FAILED[/] {}'.format(escape(name)))
    if not failed:
        console.print('[green]all the tests that ran passed[/]')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
