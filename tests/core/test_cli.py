"""Tests of pylattice.core.cli: the command line Parser and the rich Logger.

Run with:  pytest tests/core/test_cli.py
"""
import io
import logging
import os
import re
import subprocess
import sys

import pytest
from rich.console import Console

from pylattice.core.cli import Parser, Logger

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =========================================================================
# Parser
# =========================================================================
def test_defaults():
    P = Parser([])
    assert P.grid == [4, 4] and P.mpigrid == [1, 2]
    assert P.steps == 100 and P.beta == 0.5 and P.seed == 0
    assert P.debug is False and P.logfile is None and P.progress is False            # None: the log file has its default name
    assert type(P.steps) is int and type(P.beta) is float and type(P.seed) is int
    assert P.checkpoint_every == 0 and P.runlog_every == 5                  # checkpoints off; about 20 log lines per run
    assert (P.thermalization, P.measure_every) == (0, 1)
    assert P.args == {'grid': [4, 4], 'mpigrid': [1, 2], 'steps': 100, 'thermalization': 0, 'measure_every': 1, 'beta': 0.5, 'seed': 0,
                      'checkpoint_every': 0, 'runlog_every': 5, 'start_from': None, 'debug': False, 'logfile': None,
                      'progress': False}


def test_readme_command_line():
    P = Parser(['--grid=20x20', '--mpigrid=2x2', '--beta=1.5', '--steps=2000'])
    assert (P.grid, P.mpigrid, P.beta, P.steps) == ([20, 20], [2, 2], 1.5, 2000)


def test_all_options_and_both_syntaxes():
    P = Parser(['--grid', '8x8x8x4', '--mpigrid', '1x1x2x2', '--steps', '5', '--beta', '-0.25',
                '--seed', '42', '--debug', '--logfile', 'my.log', '--progress'])
    assert P.grid == [8, 8, 8, 4] and P.mpigrid == [1, 1, 2, 2]
    assert P.steps == 5 and P.beta == -0.25 and P.seed == 42 and P.debug and P.logfile == 'my.log' and P.progress
    Q = Parser(['--grid=8x8x8x4', '--mpigrid=1x1x2x2', '--steps=5', '--beta=-0.25', '--seed=42'])
    assert (Q.grid, Q.mpigrid, Q.steps, Q.beta, Q.seed) == (P.grid, P.mpigrid, P.steps, P.beta, P.seed)
    assert not Q.debug and Q.logfile is None and not Q.progress


def test_one_dimensional_and_boundary_values():
    P = Parser(['--grid=16', '--mpigrid=4', '--steps=0', '--seed=4294967295', '--beta=0'])
    assert P.grid == [16] and P.mpigrid == [4] and P.steps == 0 and P.seed == 2**32 - 1 and P.beta == 0.0


def test_parse_latt():
    P = Parser([])
    assert P.parse_latt('2x3x4') == [2, 3, 4]
    assert P.parse_latt('7') == [7]
    assert P.parse_latt(5) == [5]
    for bad in ('ax4', '4x', 'x4', '', '4.5x4', '0x4', '-2x4', '4X4', '4,4'):
        with pytest.raises(ValueError):
            P.parse_latt(bad)


@pytest.mark.parametrize('argv,option', [
    (['--grid=ax4'], '--grid'), (['--grid=4x'], '--grid'), (['--grid=0x4'], '--grid'),
    (['--grid=-2x4'], '--grid'), (['--grid=4.5x4'], '--grid'), (['--grid='], '--grid'),
    (['--mpigrid=2xb'], '--mpigrid'), (['--mpigrid=1x0'], '--mpigrid'),
    (['--steps=-1'], '--steps'), (['--steps=ten'], '--steps'), (['--steps=2.5'], '--steps'),
    (['--beta=hot'], '--beta'), (['--beta=nan'], '--beta'), (['--beta=inf'], '--beta'),
    (['--seed=-1'], '--seed'), (['--seed=4294967296'], '--seed'), (['--seed=x'], '--seed'),
    (['--steps'], '--steps'), (['--debug=yes'], '--debug'),
    (['--checkpoint-every=-1'], '--checkpoint-every'), (['--checkpoint-every=x'], '--checkpoint-every'),
    (['--runlog-every=-5'], '--runlog-every'), (['--runlog-every=2.5'], '--runlog-every'),
])
def test_invalid_values_give_a_clean_error(argv, option, capsys):
    with pytest.raises(SystemExit) as exc:
        Parser(argv)
    assert exc.value.code == 2                           # argparse usage error, not a traceback
    err = capsys.readouterr().err
    assert option in err and 'Traceback' not in err


def test_thermalization_and_measurement_interval():
    P = Parser(['--thermalization=50', '--measure-every', '5', '--steps=200'])
    assert (P.thermalization, P.measure_every, P.steps) == (50, 5, 200)
    assert type(P.thermalization) is int and type(P.measure_every) is int
    assert Parser(['--steps=100', '--thermalization=100']).runlog_every == 10          # about 20 lines over all the sweeps
    assert Parser(['--steps=0', '--thermalization=40']).runlog_every == 2
    for argv, option in ((['--thermalization=-1'], '--thermalization'), (['--thermalization=x'], '--thermalization'),
                         (['--measure-every=0'], '--measure-every'), (['--measure-every=2.5'], '--measure-every')):
        with pytest.raises(SystemExit) as exc:
            Parser(argv)
        assert exc.value.code == 2


def test_the_banner_lists_the_thermalization_and_the_measurement_interval():
    from pylattice.core.cli import run_info
    rows = dict(dict(run_info(Parser(['--grid=4x4', '--mpigrid=1x1', '--thermalization=7', '--measure-every=3'])))['run'])
    assert rows['thermalization'] == 7 and rows['measure every'] == 3


def test_the_files_of_a_run_are_named_after_the_theory_the_lattice_the_ranks_and_beta():
    P = Parser(['--grid=20x20', '--mpigrid=2x2', '--beta=1.5'], theory='ising')
    assert P.run_name() == 'ising_grid20x20_mpi2x2_b1.5' and P.checkpoint_prefix() == P.run_name()
    assert P.log_filename() == 'ising_grid20x20_mpi2x2_b1.5.log' and P.meas_filename() == 'ising_grid20x20_mpi2x2_b1.5_meas.json'
    assert Parser(['--grid=4x4x8x8', '--mpigrid=2x2x1x1', '--beta=2.3'], theory='su2').run_name() == 'su2_grid4x4x8x8_mpi2x2x1x1_b2.3'
    assert Parser(['--grid=6', '--mpigrid=4', '--beta=10']).run_name() == 'run_grid6_mpi4_b10'      # no theory: 'run'; beta 10.0 as 10
    assert Parser(['--beta=0.4406868']).run_name().endswith('_b0.4406868') and Parser(['--beta=-0.25']).run_name().endswith('_b-0.25')
    assert Parser(['--beta=0']).run_name().endswith('_b0') and Parser([], theory='x').theory == 'x'
    assert Parser(['--logfile=mine.log'], theory='ising').log_filename() == 'mine.log'              # an explicit name is kept
    assert Parser(['--logfile=off'], theory='ising').log_filename() is None
    from pylattice.core.cli import format_beta
    assert [format_beta(b) for b in (1.5, 10, 10.0, 0.0, 5.7, 1e-05, 123456.0)] == ['1.5', '10', '10', '0', '5.7', '1e-05', '123456']


def test_the_offline_measurements_are_named_after_the_run_of_the_checkpoints():
    from pylattice.core.cli import offline_name
    run = 'su3_grid4x4x8x8_mpi2x2x1x1_b5.7'
    files = ['checkpoints/{}_{:08d}.hdf5'.format(run, k) for k in (10, 20, 300)]
    assert offline_name(files) == run + '_meas_offline'                                         # not the name of the online file
    assert offline_name(files[:1]) == run + '_meas_offline' and offline_name(['/abs/dir/' + files[0]]) == run + '_meas_offline'
    assert offline_name(['ising_grid8x8_mpi2x1_b0.4_00000005.hdf5', 'ising_grid8x8_mpi2x1_b0.4_00000007.hdf5']) \
        == 'ising_grid8x8_mpi2x1_b0.4_meas_offline'
    mixed = ['a_grid4_mpi1_b1_00000001.hdf5', 'a_grid4_mpi1_b2_00000001.hdf5']                   # two runs: what they share
    assert offline_name(mixed) == 'a_grid4_mpi1_b_meas_offline'
    assert offline_name(['x_1.hdf5', 'y_2.hdf5']) == 'measurements_meas_offline' and offline_name(['conf.hdf5']) == 'conf_meas_offline'


def test_the_run_log_interval_defaults_to_about_twenty_lines_and_the_logfile_can_be_switched_off():
    assert Parser(['--steps=2000']).runlog_every == 100 and Parser(['--steps=7']).runlog_every == 1
    assert Parser(['--steps=0']).runlog_every == 0 and Parser(['--steps=2000', '--runlog-every=0']).runlog_every == 0
    assert Parser(['--logfile=off']).log_filename() is None and Parser(['--logfile=OFF']).log_filename() is None
    assert Parser(['--logfile=off']).logfile is False                                   # off, not just the default name
    assert Parser(['--logfile=a/b.log']).logfile == 'a/b.log' and Parser(['--logfile=a/b.log']).log_filename() == 'a/b.log'


def test_checkpoint_and_runlog_options():
    P = Parser(['--checkpoint-every=500', '--runlog-every', '10'])
    assert (P.checkpoint_every, P.runlog_every) == (500, 10)
    assert type(P.checkpoint_every) is int and P.args['checkpoint_every'] == 500
    assert Parser(['--checkpoint-every=0']).checkpoint_every == 0


def test_grid_and_mpigrid_must_have_the_same_dimensions(capsys):
    with pytest.raises(SystemExit) as exc:
        Parser(['--grid=4x4x4', '--mpigrid=2x2'])
    assert exc.value.code == 2
    assert '--grid has 3 dimensions but --mpigrid has 2' in capsys.readouterr().err
    with pytest.raises(SystemExit):                      # the default mpigrid is 2-dimensional
        Parser(['--grid=4'])


def test_unknown_option_and_stray_argument(capsys):
    for argv in (['--nope'], ['stray'], ['--grid=4x4', '--grid2=3']):
        with pytest.raises(SystemExit) as exc:
            Parser(argv)
        assert exc.value.code == 2
    capsys.readouterr()


def test_help_describes_every_option(capsys):
    with pytest.raises(SystemExit) as exc:
        Parser(['--help'])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for option in ('--grid', '--mpigrid', '--steps', '--beta', '--seed', '--debug', '--logfile', '--progress',
                   '--checkpoint-every', '--runlog-every'):
        assert option in out
    assert 'XxYxZx' not in out                           # the old copy-pasted help texts are gone
    assert 'inverse temperature' in out and 'default: 4x4' in out and 'mpiexec' in out


def test_parser_reads_sys_argv_by_default(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['prog', '--grid=6x6', '--mpigrid=1x1', '--steps=3'])
    P = Parser()
    assert (P.grid, P.mpigrid, P.steps) == ([6, 6], [1, 1], 3)


def test_ising_model_command_line_is_wired():
    r = subprocess.run([sys.executable, '-m', 'pylattice.theories.spin.ising', '--help'], cwd=ROOT, capture_output=True,
                       text=True, env=dict(os.environ, PYTHONPATH=ROOT), timeout=120)
    assert r.returncode == 0 and '--beta' in r.stdout and 'inverse temperature' in r.stdout
    # (a bad command line stopping with status 2 is tested in process: test_grid_and_mpigrid_must_have_the_same_dimensions
    # and test_ising.test_command_line_driver; one launch of the module is enough to check the entry point)


# =========================================================================
# Logger
# =========================================================================
@pytest.fixture(autouse=True)
def clean_logger():
    yield
    lg = logging.getLogger('pylattice')
    for h in list(lg.handlers):
        lg.removeHandler(h)
        h.close()


def console(color=True):
    return Console(file=io.StringIO(), force_terminal=color, color_system='standard' if color else None,
                   width=120)


def raw(c):
    return c.file.getvalue()


def text(c):
    # the output without its ANSI color codes (rich also colors parts of the message, e.g. numbers)
    return re.sub(r'\x1b\[[0-9;]*m', '', raw(c))


def test_info_level_and_colors():
    c = console()
    log = Logger(debug=False, logfile=False, console=c, rank=0, size=1).log
    log.debug('hidden debug')
    log.info('visible info')
    log.warning('visible warning')
    log.error('visible error')
    out = raw(c)
    assert 'hidden debug' not in out
    for msg in ('visible info', 'visible warning', 'visible error'):
        assert msg in text(c)
    assert '\x1b[' in out                                # ANSI colors
    # each level name is preceded by its own color code
    codes = dict((name, code) for code, name in re.findall(r'(\x1b\[[0-9;]*m)(INFO|WARNING|ERROR)', out))
    assert set(codes) == {'INFO', 'WARNING', 'ERROR'} and len(set(codes.values())) == 3, codes


def test_no_colors_when_not_a_terminal():
    c = console(color=False)
    log = Logger(debug=False, logfile=False, console=c, rank=0, size=1).log
    log.info('plain message')
    out = text(c)
    assert 'plain message' in out and 'INFO' in out and '\x1b[' not in out


def test_debug_level():
    c = console()
    log = Logger(debug=True, logfile=False, console=c, rank=0, size=1).log
    log.debug('now visible')
    assert 'now visible' in text(c) and 'DEBUG' in text(c)
    assert log.level == logging.DEBUG
    assert Logger(debug=False, logfile=False, console=console(), rank=0, size=1).log.level == logging.INFO


def test_logger_object_is_usable_by_the_model():
    L = Logger(debug=False, logfile=False, console=console(), rank=0, size=1)
    assert isinstance(L.log, logging.Logger) and L.log.name == 'pylattice'
    assert (L.debug, L.logfile) == (False, False)
    assert L.get_logger(debug=False, logfile=False) is L.log


def test_ranks_rank0_speaks_others_only_warn():
    c0, c1 = console(), console()
    Logger(debug=False, logfile=False, console=c0, rank=0, size=4).log.info('from zero')
    assert 'from zero' in text(c0) and '[rank 0]' in text(c0)
    log1 = Logger(debug=False, logfile=False, console=c1, rank=1, size=4).log
    log1.info('chatter')
    log1.warning('problem')
    out = text(c1)
    assert 'chatter' not in out
    assert 'problem' in out and '[rank 1]' in out


def test_ranks_debug_makes_every_rank_speak():
    c = console()
    log = Logger(debug=True, logfile=False, console=c, rank=3, size=4).log
    log.info('info from three')
    log.debug('debug from three')
    out = text(c)
    assert 'info from three' in out and 'debug from three' in out and out.count('[rank 3]') == 2


def test_single_rank_has_no_rank_prefix():
    c = console()
    Logger(debug=False, logfile=False, console=c, rank=0, size=1).log.info('alone')
    assert 'alone' in text(c) and 'rank' not in text(c)


def test_rank_is_asked_to_mpi_when_not_given():
    c = console()
    L = Logger(debug=False, logfile=False, console=c)    # a single process: rank 0 of 1
    assert (L.rank, L.size) == (0, 1)
    L.log.info('hello')
    assert 'hello' in text(c)


def test_logfile_is_plain_text_and_console_still_logs(tmp_path):
    c = console()
    path = str(tmp_path / 'run.log')
    log = Logger(debug=False, logfile=True, filename=path, console=c, rank=0, size=1).log
    log.info('to both')
    log.warning('careful')
    log.debug('not at INFO')
    for h in log.handlers:
        h.flush()
    content = open(path).read()
    assert 'to both' in content and 'careful' in content and 'not at INFO' not in content
    assert 'INFO' in content and 'WARNING' in content and '\x1b[' not in content
    assert 'to both' in text(c) and 'careful' in text(c)
    for line in content.splitlines():                    # time, level, message
        assert line[:4].isdigit() and ':  ' in line


def test_logfile_default_name_and_no_file_without_the_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Logger(debug=False, logfile=False, console=console(), rank=0, size=1).log.info('x')
    assert os.listdir(str(tmp_path)) == []
    log = Logger(debug=False, logfile=True, console=console(), rank=0, size=1).log
    log.info('x')
    for h in log.handlers:
        h.flush()
    assert os.listdir(str(tmp_path)) == ['run.log']


def test_logfile_filtered_like_the_console(tmp_path):
    path = str(tmp_path / 'r2.log')
    log = Logger(debug=False, logfile=True, filename=path, console=console(), rank=2, size=4).log
    log.info('quiet')
    log.error('loud')
    for h in log.handlers:
        h.flush()
    content = open(path).read()
    assert 'quiet' not in content and 'loud' in content and '[rank 2]' in content


def test_creating_a_logger_again_replaces_the_handlers_and_configures_again():
    c1, c2 = console(), console()
    Logger(debug=False, logfile=False, console=c1, rank=0, size=1)
    L2 = Logger(debug=True, logfile=False, console=c2, rank=0, size=1)    # basicConfig would ignore this
    assert len(L2.log.handlers) == 1
    L2.log.debug('only once')
    assert text(c1) == '' and text(c2).count('only once') == 1
    L3 = Logger(debug=False, logfile=True, filename=os.devnull, console=console(), rank=0, size=1)
    assert len(L3.log.handlers) == 2
    assert len(Logger(debug=False, logfile=False, console=console(), rank=0, size=1).log.handlers) == 1


def test_root_logger_is_left_alone():
    root = logging.getLogger()
    before = (list(root.handlers), root.level)
    log = Logger(debug=True, logfile=False, console=console(), rank=0, size=1).log
    assert (list(root.handlers), root.level) == before
    assert log.propagate is False


def test_exception_is_logged_with_a_traceback():
    c = console(color=False)
    log = Logger(debug=False, logfile=False, console=c, rank=0, size=1).log
    try:
        1 / 0
    except ZeroDivisionError:
        log.exception('it failed')
    out = text(c)
    assert 'it failed' in out and 'ZeroDivisionError' in out


def test_brackets_in_messages_are_not_rich_markup():
    c = console(color=False)
    Logger(debug=False, logfile=False, console=c, rank=0, size=1).log.info('grid [4, 4] [bold]kept[/bold]')
    assert 'grid [4, 4] [bold]kept[/bold]' in text(c)


# ------------------------------------------------------------------ logo and run information
def test_banner_has_the_logo_and_a_box_for_each_section_on_the_terminal_and_in_the_file(tmp_path):
    from pylattice.core.cli import LOGO, run_info
    p = Parser(['--grid=4x6', '--mpigrid=1x1', '--beta=0.7', '--steps=33', '--seed=5'])
    c = console()
    lg = Logger(debug=False, logfile=True, filename=str(tmp_path / 'run.log'), console=c, rank=0, size=1)
    lg.banner(p)
    for h in lg.log.handlers:
        h.flush()
    on_file = (tmp_path / 'run.log').read_text(encoding='utf-8')
    for shown in (on_file, text(c)):
        assert 'pyLattice' in shown and LOGO.splitlines()[2].rstrip() in shown
        for title in ('system', 'parallel', 'lattice', 'run'):
            assert '\u256d\u2500 {} \u2500'.format(title) in shown, title                    # a box with its title
        for needle in ('os ', 'architecture', 'mpi ranks', 'python', 'numpy', 'h5py', 'local grid',
                       '4x6', 'global lattice', '24 sites', '0.7', '33', 'seed', 'run_grid4x6_mpi1x1_b0.7.log',
                       'run_grid4x6_mpi1x1_b0.7_meas.json'):                                  # the names of the files of the run
            assert needle in shown, needle
        tops = [l for l in shown.splitlines() if l.startswith('\u256d')]
        bottoms = [l for l in shown.splitlines() if l.startswith('\u2570')]
        assert len(tops) == len(bottoms) == 4 and len({len(l) for l in tops + bottoms}) == 1      # equal boxes
    assert '\x1b[' not in on_file and '\x1b[' in raw(c)                                         # color on the terminal only
    sections = dict(run_info(p))
    assert dict(sections['lattice'])['global lattice'].startswith('4x6')
    assert dict(sections['run'])['log file'] == 'run_grid4x6_mpi1x1_b0.7.log' and dict(sections['run'])['run log every'] == 1
    assert dict(sections['run'])['measurements'] == 'run_grid4x6_mpi1x1_b0.7_meas.json'


def test_show_prints_in_color_on_the_terminal_and_in_plain_text_in_the_file_and_only_on_rank_zero(tmp_path):
    from rich.text import Text
    c0, c1 = console(), console()
    path = str(tmp_path / 'run.log')
    lg = Logger(debug=False, logfile=True, filename=path, console=c0, rank=0, size=2)
    lg.show(Text.assemble(('step=', 'dim'), ('10', 'bold')))
    lg.log.info('a message')
    Logger(debug=False, logfile=True, filename=str(tmp_path / 'r1.log'), console=c1, rank=1, size=2).show(Text('silent'))
    content = open(path, encoding='utf-8').read()
    assert content.splitlines()[0] == 'step=10' and 'a message' in content and '\x1b[' not in content   # one line, plain
    assert 'step=10' in text(c0) and '\x1b[' in raw(c0)
    assert text(c1) == '' and 'silent' not in open(str(tmp_path / 'r1.log'), encoding='utf-8').read()


def test_the_log_file_is_appended_to_and_long_lines_are_not_wrapped(tmp_path):
    from rich.text import Text
    path = str(tmp_path / 'run.log')
    for k in range(2):
        lg = Logger(debug=False, logfile=True, filename=path, console=console(), rank=0, size=1)
        lg.show(Text('x' * 300 + str(k)))
    lines = open(path, encoding='utf-8').read().splitlines()
    assert [len(l) for l in lines] == [301, 301]
