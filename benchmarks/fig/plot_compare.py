"""Compare strong scaling of (A*B).reducesum() on a 40x40x40x40 LatticeRealMatrix (N=2).

Old: Intel Core i5-8365U (4 cores / 8 threads), numbers from plot.py.
New: Apple M4 Pro (10 performance + 4 efficiency cores), run with
     performance_evaluation.py, best of 3 repeats.
"Naive" = plain per-site np.dot loop on the whole grid, no MPI.
Usage: python plot_compare.py   (writes perf_compare.png next to this file)
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

old_x = [1, 2, 4]
old_t = [4.035101890563965, 2.142939567565918, 1.300379753112793]
old_naive = 4.168240785598755

new_x = [1, 2, 4, 8, 10]
new_t = [1.4035320281982422, 0.715540885925293, 0.3644578456878662,
         0.18857717514038086, 0.1403048038482666]
new_naive = 1.2318642139434814

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

a = ax[0]
a.plot(old_x, old_t, 'v-', color='tab:green', alpha=0.7, label='MPI, Intel i5-8365U')
a.axhline(old_naive, ls='--', color='tab:green', alpha=0.7, label='Naive, Intel i5-8365U')
a.plot(new_x, new_t, 'o-', color='tab:blue', label='MPI, Apple M4 Pro')
a.axhline(new_naive, ls='--', color='tab:blue', label='Naive, Apple M4 Pro')
a.set_xlabel('processes')
a.set_ylabel('time [s]')
a.set_xticks(new_x)
a.set_title('Time')
a.grid(alpha=0.4)
a.legend(fontsize=8)

b = ax[1]
b.plot(old_x, [old_t[0] / t for t in old_t], 'v-', color='tab:green', alpha=0.7, label='Intel i5-8365U')
b.plot(new_x, [new_t[0] / t for t in new_t], 'o-', color='tab:blue', label='Apple M4 Pro')
b.plot([1, 10], [1, 10], ':', color='gray', label='ideal')
b.set_xlabel('processes')
b.set_ylabel('speedup vs 1 process')
b.set_xticks(new_x)
b.set_title('Speedup')
b.grid(alpha=0.4)
b.legend(fontsize=8)

fig.suptitle('LatticeRealMatrix 40x40x40x40, N=2: (A*B).reducesum()')
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'perf_compare.png')
fig.savefig(out, dpi=130)
print(out)
