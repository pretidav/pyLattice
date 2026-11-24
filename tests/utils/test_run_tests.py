"""Checks of run_tests.py, the script that runs the whole test routine locally: that it lists every test file of the
repository (a new test file that nobody added to it would silently never run there), and how it builds its steps.
Run with:  pytest tests/utils/test_run_tests.py
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('run_tests', ROOT / 'run_tests.py')
run_tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_tests)


def test_every_test_file_of_the_repository_is_in_the_routine():
    on_disk = {str(p.relative_to(ROOT)) for p in (ROOT / 'tests').rglob('test_*.py')}
    listed = run_tests.all_files_of_the_routine()
    assert on_disk - listed == set(), 'not run by run_tests.py: {}'.format(sorted(on_disk - listed))
    assert listed - on_disk == set(), 'listed but not there: {}'.format(sorted(listed - on_disk))


def test_the_steps_of_each_mode():
    names = lambda **kw: [s.name for s in run_tests.build_steps('mpiexec', **kw)]
    assert names(quick=True) == ['pytest (single process)']
    standard, full = names(), names(full=True)
    assert set(standard) < set(full) and len(standard) > 5
    assert 'test_ising.py on 2 ranks' in standard and 'test_ising.py on 1 rank' not in standard
    assert {'test_ising.py on 1 rank', 'test_ising.py on 4 ranks'} <= set(full)
    assert 'HDF5 cross-layout: write on 4 ranks, read on 2' in full and 'HDF5 cross-layout: write on 4 ranks, read on 2' not in standard


def test_mpi_commands_use_mpiexec_with_the_ranks_and_oversubscribe_only_when_asked():
    steps = {s.name: s for s in run_tests.build_steps('/x/mpiexec', full=True)}
    cmd = steps['test_ising.py on 2 ranks'].commands[0]
    assert cmd[:3] == ['/x/mpiexec', '-n', '2'] and cmd[3] == run_tests.PY
    assert steps['test_ising.py on 2 ranks'].needs_mpi and not steps['test_grid.py'].needs_mpi
    over = {s.name: s for s in run_tests.build_steps('/x/mpiexec', oversubscribe=True)}
    assert over['test_ising.py on 2 ranks'].commands[0][:4] == ['/x/mpiexec', '--oversubscribe', '-n', '2']
    cross = steps['HDF5 cross-layout: write on 4 ranks, read on 2']
    assert len(cross.commands) == 2 and cross.needs_parallel_h5
    assert steps['test_hdf5io.py on 4 ranks'].needs_parallel_h5


def test_the_summary_shown_next_to_a_pass():
    assert run_tests.summary_of('....\n571 passed in 12.24s\n') == '571 passed'
    assert run_tests.summary_of('[+] a passed on 2 rank(s)\n[+] b passed on 2 rank(s)\n[-] c skipped: x\n') == '2 tests, 1 skipped'
    assert run_tests.summary_of('') == ''
