"""Tests of pylattice.utils.plot_meas, the offline printing and plotting of meas.json.
Run with:  pytest tests/utils/test_plot_meas.py
"""
import json

import numpy as np
import pytest

from pylattice.utils import plot_meas


def write(path, n=400, seed=0, nulls=()):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal(n) * 0.1 + 0.5
    records = [{'sweep': i, 'm': (None if i in nulls else float(m[i])), 'e': -1.5 + 0.01 * i / n,
                'acceptance': None if i == 0 else 0.4 + 0.1 * float(rng.random())} for i in range(n)]
    with open(path, 'w') as f:
        json.dump({'measurements': records}, f)
    return m


def test_load_gives_a_history_with_nan_for_null(tmp_path):
    p = tmp_path / 'meas.json'
    m = write(p, n=50, nulls=(3,))
    h = plot_meas.load(p)
    assert len(h) == 50 and h.sweeps == list(range(50)) and len(h.acceptance) == 49
    assert np.isnan(h.values['m'][3]) and h.values['m'][4] == pytest.approx(m[4])


def test_summary_has_mean_error_and_tau_for_every_quantity(tmp_path):
    p = tmp_path / 'meas.json'
    m = write(p, n=2000)
    rows = plot_meas.summary(plot_meas.load(p))
    assert set(rows) == {'m', 'e', 'acceptance'}
    assert rows['m']['n'] == 2000 and rows['m']['mean'] == pytest.approx(m.mean())
    assert rows['m']['error'] == pytest.approx(0.1 / np.sqrt(2000), rel=0.5)          # independent points
    assert 0.3 < rows['m']['tau'] < 0.8
    assert rows['acceptance']['n'] == 1999


def test_summary_leaves_out_what_is_not_finite_and_survives_a_constant_series(tmp_path):
    p = tmp_path / 'meas.json'
    m = write(p, n=300, nulls=(5, 6))
    rows = plot_meas.summary(plot_meas.load(p))
    assert rows['m']['n'] == 300 and rows['m']['finite'] == 298
    assert rows['m']['mean'] == pytest.approx(np.delete(m, [5, 6]).mean())
    h = plot_meas.History()
    for i in range(100):
        h.record(i, {'x': 1.0})
    r = plot_meas.summary(h)['x']
    assert r['mean'] == 1.0 and r['tau'] is None                                       # no scatter: no autocorrelation time
    assert 'not finite' in plot_meas.format_summary(rows)


def test_skip_leaves_out_the_thermalization(tmp_path):
    p = tmp_path / 'meas.json'
    write(p, n=100)
    h = plot_meas.skip_first(plot_meas.load(p), 30)
    assert h.sweeps[0] == 30 and len(h) == 70 and len(h.acceptance) == 69 and len(h.values['m']) == 70


def test_command_prints_the_table_and_writes_the_figures(tmp_path, capsys):
    p = tmp_path / 'meas.json'
    write(p, n=500)
    out = tmp_path / 'figs'
    plot_meas.main([str(p), '--skip', '100', '--outdir', str(out)])
    text = capsys.readouterr().out
    assert '400 measurements, sweeps 100 to 499 (100 skipped)' in text and 'tau_int' in text
    for name in ('m', 'e', 'acceptance'):
        assert name in text
        assert (out / (name + '.png')).read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'
    other = tmp_path / 'other'
    plot_meas.main([str(p), '--no-plot', '--outdir', str(other)])
    assert not other.exists()


def test_reads_a_file_of_a_run_that_is_still_going(tmp_path, capsys):
    p = tmp_path / 'meas.json'
    p.write_text('{"measurements": [\n  {"sweep": 0, "m": 0.1, "acceptance": null},\n  {"sweep": 1, "m": 0.2, "acceptance": 0.5}\n]}\n')
    plot_meas.main([str(p), '--no-plot'])
    assert '2 measurements' in capsys.readouterr().out


def test_errors_are_clear(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        plot_meas.main([str(tmp_path / 'missing.json')])
    assert e.value.code == 1 and 'cannot read' in capsys.readouterr().err
    bad = tmp_path / 'bad.json'
    bad.write_text('{"measurements": [')                                               # cut by a crash
    with pytest.raises(SystemExit):
        plot_meas.main([str(bad)])
    p = tmp_path / 'meas.json'
    write(p, n=10)
    with pytest.raises(SystemExit) as e:
        plot_meas.main([str(p), '--skip', '1000'])
    assert 'after skipping 1000 sweeps (it has 10)' in capsys.readouterr().err
    p.write_text('{"measurements": []}')
    with pytest.raises(SystemExit):
        plot_meas.main([str(p)])


def test_a_file_without_acceptance_can_be_skipped_and_summarised(tmp_path):
    # what the offline measurements of stored configurations (pylattice.core.measure) write
    p = tmp_path / 'meas.json'
    records = [{'sweep': 10 * i, 'm': 0.1 * (i % 5), 'acceptance': None} for i in range(60)]
    p.write_text(json.dumps({'measurements': records}))
    h = plot_meas.skip_first(plot_meas.load(p), 100)
    assert len(h) == 50 and h.acceptance == []
    assert set(plot_meas.summary(h)) == {'m'}
