"""Compare strong scaling of (A*B).reducesum() on a 40x40x40x40 LatticeRealMatrix (N=2).

Old: Intel Core i5-8365U (4 cores / 8 threads), numbers from plot.py (per-site loops in the lattice classes).
M4 Pro before: Apple M4 Pro (10 performance + 4 efficiency cores), same per-site loops.
M4 Pro now: the same machine with the vectorized lattice arithmetic of core/grid.py.
All run with performance_evaluation.py, best of 3 repeats.
"Naive" = plain per-site np.dot loop on the whole grid, no MPI.
Style: plot_style (pip install -r requirements.txt).
Usage: python plot_compare.py   (writes perf_compare.png next to this file)
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
import plot_style

# the style's bold font has no bold fallback on macOS (Monaco): hide that one matplotlib message
logging.getLogger('matplotlib.font_manager').addFilter(
    lambda record: not record.getMessage().startswith('findfont: Failed to find font weight'))

old_x = [1, 2, 4]
old_t = [4.035101890563965, 2.142939567565918, 1.300379753112793]
old_naive = 4.168240785598755

new_x = [1, 2, 4, 8, 10]
new_t = [1.4035320281982422, 0.715540885925293, 0.3644578456878662,
         0.18857717514038086, 0.1403048038482666]
new_naive = 1.2318642139434814

vec_t = [0.12886595726013184, 0.06572103500366211, 0.035180091857910156,
         0.029230117797851562, 0.022132158279418945]          # vectorized lattice arithmetic, 2026-11-24
vec_naive = 1.1818819046020508

OLD, NEW = plot_style.GREY, plot_style.ACCENT          # old machine in grey, the new one in the accent colour

with plot_style.context():
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.4))

    a = ax[0]
    a.plot(old_x, old_t, 'v-', color=OLD, label='MPI, INTEL I5-8365U')
    a.axhline(old_naive, ls=plot_style.DASHED, color=OLD, label='NAIVE, INTEL I5-8365U')
    a.plot(new_x, new_t, 'o--', color=NEW, alpha=0.45, label='MPI, APPLE M4 PRO, LOOPS')
    a.plot(new_x, vec_t, 'o-', color=NEW, label='MPI, APPLE M4 PRO, VECTORIZED')
    a.axhline(vec_naive, ls=plot_style.DASHED, color=NEW, label='NAIVE, APPLE M4 PRO')
    a.set_yscale('log')
    a.set_xlabel('PROCESSES')
    a.set_ylabel('TIME [S]')
    a.set_xticks(new_x)
    a.set_title('TIME')
    plot_style.style_axes(a)
    plot_style.legend(a, 'upper center', fontsize=7, ncol=2, bbox_to_anchor=(0.5, -0.2))

    b = ax[1]
    b.plot([1, 10], [1, 10], ls=plot_style.DOTTED, color=plot_style.INK, label='IDEAL')
    b.plot(old_x, [old_t[0] / t for t in old_t], 'v-', color=OLD, label='INTEL I5-8365U')
    b.plot(new_x, [new_t[0] / t for t in new_t], 'o--', color=NEW, alpha=0.45, label='APPLE M4 PRO, LOOPS')
    b.plot(new_x, [vec_t[0] / t for t in vec_t], 'o-', color=NEW, label='APPLE M4 PRO, VECTORIZED')
    b.set_xlabel('PROCESSES')
    b.set_ylabel('SPEEDUP VS 1 PROCESS')
    b.set_xticks(new_x)
    b.set_title('SPEEDUP')
    plot_style.style_axes(b)
    plot_style.legend(b, 'upper left', fontsize=8)

    fig.suptitle('LATTICEREALMATRIX 40X40X40X40, N=2: (A*B).REDUCESUM()', fontweight='bold')
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'perf_compare.png')
    plot_style.savefig(fig, out)
print(out)
