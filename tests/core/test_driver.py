"""run_simulation (pylattice.core.driver): the run of any model from the command line, with the files, the log and the
summary every theory gets. A toy model, not Ising. One process. Run with:  pytest tests/core/test_driver.py
"""
import io
import json
import os
import re

import pytest
from rich.console import Console

from pylattice.core.cli import Logger, Parser
from pylattice.core.driver import run_simulation


class Toy:
    """what montecarlo.run needs of a model: sweep(rng) -> (accepted, attempted) and observables()"""
    def __init__(self):
        self.sweeps = 0

    def sweep(self, rng):
        self.sweeps += 1
        return 1, 2

    def observables(self):
        return {'x': float(self.sweeps)}


def plain(console):
    return re.sub(r'\x1b\[[0-9;]*m', '', console.file.getvalue())


def run(tmp_path, monkeypatch, *argv):
    monkeypatch.chdir(tmp_path)
    parser = Parser(list(argv))
    term = Console(file=io.StringIO(), force_terminal=True, color_system='standard', width=120)
    logger = Logger(debug=False, logfile=parser.logfile is not None, filename=parser.logfile or 'run.log', console=term,
                    rank=0, size=1)
    model = Toy()
    history = run_simulation(model, parser, logger, prefix='toy')
    for handler in logger.log.handlers:
        handler.flush()
    return model, history, term, parser


def test_a_run_gives_the_measurement_file_the_run_log_lines_and_the_summary(tmp_path, monkeypatch):
    model, history, term, parser = run(tmp_path, monkeypatch, '--grid=2x2', '--mpigrid=1x1', '--steps=10', '--runlog-every=5')
    assert model.sweeps == 10 and len(history) == 11 and history.acceptance[1] == 0.5
    records = json.load(open('meas.json'))['measurements']
    assert [r['sweep'] for r in records] == list(range(11)) and records[3]['x'] == 3.0
    shown = plain(term)
    on_file = open('run.log', encoding='utf-8').read()
    for text in (shown, on_file):
        assert [l.split()[0] for l in text.splitlines() if l.startswith('step=')] == ['step=5', 'step=10']
        assert '╭─ done' in text and '0.500' in text and 'meas.json' in text and 'sweeps' in text
    assert 'MONTE CARLO' not in shown and sorted(os.listdir('.')) == ['meas.json', 'run.log']        # no bar, no checkpoints


def test_no_log_lines_without_runlog_every_and_no_file_with_logfile_off(tmp_path, monkeypatch):
    model, history, term, parser = run(tmp_path, monkeypatch, '--steps=6', '--runlog-every=0', '--logfile=off', '--progress')
    assert 'step=' not in plain(term)
    assert os.listdir('.') == ['meas.json']
    assert 'off' in plain(term).split('log file')[1]


def test_the_default_gives_about_twenty_log_lines(tmp_path, monkeypatch):
    model, history, term, parser = run(tmp_path, monkeypatch, '--steps=100')
    lines = [l for l in plain(term).splitlines() if l.startswith('step=')]
    assert len(lines) == 20 and parser.runlog_every == 5


@pytest.mark.parametrize('steps', [0, 1])
def test_zero_and_one_sweep_do_not_break_the_summary(tmp_path, monkeypatch, steps):
    model, history, term, parser = run(tmp_path, monkeypatch, '--steps={}'.format(steps))
    assert '╭─ done' in plain(term) and model.sweeps == steps
