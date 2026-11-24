"""Prints and plots the measurements of a run, offline: reads the meas.json that the run wrote
(pylattice.core.callbacks.MeasurementFile) and needs no MPI processes, no lattice, and not even
the run to be over (the file is a valid JSON document after every measurement).

    python -m pylattice.utils.plot_meas meas.json
    python -m pylattice.utils.plot_meas meas.json --skip 200 --outdir figures
    python -m pylattice.utils.plot_meas meas.json --no-plot

It prints, for every observable and for the acceptance, the number of points, the mean and its
error (batch means, pylattice.core.stats.mean_error) and the integrated autocorrelation time with
its error (Gamma method, pylattice.core.stats.uwerr); and writes one figure per quantity against
the sweep, <outdir>/<name>.png, in the style of plot_style.
"""
import argparse
import json
import os
import sys

import numpy as np

from pylattice.core import stats
from pylattice.core.montecarlo import History


def load(path):
    """meas.json -> History. null (a nan or inf of the run) becomes nan."""
    with open(path) as f:
        records = json.load(f)['measurements']
    history = History()
    for r in records:
        values = {k: (np.nan if v is None else v) for k, v in r.items() if k not in ('sweep', 'acceptance')}
        history.record(r['sweep'], values, r['acceptance'])
    return history


def skip_first(history, skip):
    """a History without the measurements of the first `skip` sweeps (the thermalization of the run)"""
    out = History()
    for i, sweep in enumerate(history.sweeps):
        if sweep >= skip:
            # the acceptance of a measurement covers the sweeps since the previous one: the first kept has none
            out.record(sweep, {k: v[i] for k, v in history.values.items()},
                       history.acceptance[i - 1] if (i > 0 and len(out) and i - 1 < len(history.acceptance)) else None)
    return out


def summary(history):
    """name -> dict(n, mean, error, tau, dtau) for every observable and the acceptance. tau and dtau are
    None when the series is too short or constant for the Gamma method."""
    series = {name: np.array(v, dtype=float) for name, v in history.values.items()}
    if history.acceptance:
        series['acceptance'] = np.array(history.acceptance, dtype=float)
    out = {}
    for name, x in series.items():
        good = x[np.isfinite(x)]
        row = dict(n=len(x), finite=len(good), mean=np.nan, error=np.nan, tau=None, dtau=None)
        if len(good):
            row['mean'] = float(good.mean())
        if len(good) >= 40:
            row['error'] = float(stats.mean_error(good)[1])
        try:
            _, _, row['tau'], row['dtau'] = stats.uwerr(good)
        except ValueError:
            pass
        out[name] = row
    return out


def format_summary(rows):
    lines = ['{:<12} {:>7} {:>14} {:>12} {:>16}'.format('', 'points', 'mean', 'error', 'tau_int')]
    for name, r in rows.items():
        tau = '{:.2f} +- {:.2f}'.format(r['tau'], r['dtau']) if r['tau'] is not None else '-'
        lines.append('{:<12} {:>7} {:>14.6g} {:>12.2g} {:>16}'.format(name, r['n'], r['mean'], r['error'], tau))
        if r['finite'] != r['n']:
            lines.append('{:<12} {} of the {} values are not finite (nan/inf in the run) and are left out'.format(
                '', r['n'] - r['finite'], r['n']))
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Print and plot the measurements (meas.json) of a run.')
    parser.add_argument('file', nargs='?', default='meas.json', help='the measurements file')
    parser.add_argument('--skip', type=int, default=0, metavar='SWEEPS',
                        help='leave out the measurements of the first SWEEPS sweeps (thermalization)')
    parser.add_argument('--outdir', default='.', help='where the figures go')
    parser.add_argument('--no-plot', action='store_true', help='only print the table')
    args = parser.parse_args(argv)

    try:
        history = load(args.file)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, '{}: cannot read the measurements of {}: {}\n'.format(parser.prog, args.file, error))
    if args.skip < 0:
        parser.error('--skip cannot be negative')
    total = len(history)
    history = skip_first(history, args.skip)
    if len(history) == 0:
        parser.exit(1, '{}: no measurements in {}{}\n'.format(
            parser.prog, args.file, ' after skipping {} sweeps (it has {})'.format(args.skip, total) if total else ''))

    print('{}: {} measurements, sweeps {} to {}{}'.format(
        args.file, len(history), history.sweeps[0], history.sweeps[-1],
        ' ({} skipped)'.format(total - len(history)) if total != len(history) else ''))
    print(format_summary(summary(history)))

    if not args.no_plot:
        from mpi4py import MPI
        os.makedirs(args.outdir, exist_ok=True)
        names = list(history.values) + (['acceptance'] if history.acceptance else [])
        history.plot(MPI.COMM_SELF, files={n: os.path.join(args.outdir, n + '.png') for n in names})


if __name__ == '__main__':
    main()
