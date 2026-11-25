[![Tests core](https://github.com/pretidav/pyLattice/actions/workflows/tests-core.yml/badge.svg?branch=on)](https://github.com/pretidav/pyLattice/actions/workflows/tests-core.yml) [![Tests gauge](https://github.com/pretidav/pyLattice/actions/workflows/tests-gauge.yml/badge.svg?branch=on)](https://github.com/pretidav/pyLattice/actions/workflows/tests-gauge.yml) [![Tests spin](https://github.com/pretidav/pyLattice/actions/workflows/tests-spin.yml/badge.svg?branch=on)](https://github.com/pretidav/pyLattice/actions/workflows/tests-spin.yml)

# pyLattice
<p align="center">
  <img width="150" height="150" src="docs/mylogo.png">
</p>

A lattice (gauge) theory simulation library in Python, parallel with MPI. A lattice is split over MPI ranks,
a theory provides the Monte Carlo sweep, and the library provides the rest: checkpoints and exact restarts,
logs, measurements written as the run goes, and offline analysis.

**Documentation:** [docs/notes.pdf](docs/notes.pdf) (source: [docs/notes.tex](docs/notes.tex)) describes the design, the
parallelization, the file formats, the algorithms and the tests.

Implemented: the Ising model (spin theories on sites, with values in the group Z2) and the Wilson-action gauge theories
SU(N) (SU(2) included) and SO(N), with heat bath, Metropolis and overrelaxation updates.

**Contents:** [Installation](#installation) · [Quick start](#quick-start) · [Running a simulation](#running-a-simulation) ·
[Working with the results](#working-with-the-results) · [Repository layout](#repository-layout) · [Tests](#tests) ·
[Performance](#performance) · [Validation](#validation) · [Citation](#citation)

---

## Installation

pyLattice needs MPI and an `h5py` built with **parallel HDF5** (the standard PyPI `h5py` wheels are not).
The simplest way is a conda-forge environment, where MPI, HDF5, `mpi4py` and `h5py` are built against each other.

### macOS (Apple Silicon) and Linux, with conda-forge

~~~
conda create -n pylattice -c conda-forge --override-channels \
  python=3.11 numpy scipy matplotlib psutil pytest rich openmpi mpi4py "h5py=*=mpi_openmpi*"
conda activate pylattice
pip install "plot_style @ git+https://github.com/pretidav/plot_style.git"   # style of the figures, not on conda
pip install -e . --no-deps                                                  # optional: import pylattice from anywhere
~~~

- `--override-channels` uses only conda-forge and ignores the channels of your conda config, so packages from
  different channels are not mixed.
- Without `pip install -e .`, run everything from the repository root (or `export PYTHONPATH=.`).
- `pip install -r requirements.txt` alone is not enough: it has no `h5py`, and the PyPI one has no parallel HDF5.
  Without parallel `h5py` everything works except checkpoints, restart and offline measurements (their tests are
  skipped).

Check it:

~~~
python -c "import h5py; print(h5py.get_config().mpi)"   # must print True
mpiexec -n 2 python tests/core/test_grid_mpi.py
~~~

MPI uses physical cores only: `python -c "import psutil; print(psutil.cpu_count(logical=False))"`. With more ranks
than cores add `--oversubscribe` to `mpiexec`.

### Linux without conda

Install a C++ compiler and `make` (`sudo apt-get install g++ make`), then OpenMPI following
[this guide](https://edu.itp.phys.ethz.ch/hs12/programming_techniques/openmpi.pdf), then `mpi4py` and a parallel
`h5py` (`conda install -c conda-forge mpi4py openmpi "h5py>=2.9=mpi*"`).

---

## Quick start

~~~
mpiexec -np 4 python -m pylattice.theories.spin.ising --grid=20x20 --mpigrid=2x2 --beta=1.5 --steps=2000
~~~

`--grid` is the lattice of **each rank** and `--mpigrid` the number of ranks along each direction, so this is a
40x40 lattice on 4 ranks (every local size must be even). The run prints the pyLattice logo and a summary of
the system and of the run (OS, architecture, MPI, `h5py`, lattice, parameters), one line every few sweeps, and writes
`ising_grid20x20_mpi2x2_b1.5_meas.json` and `ising_grid20x20_mpi2x2_b1.5.log`. It plots nothing: plotting is done offline, see [below](#working-with-the-results).

SU(N) gauge theories with the Wilson action run the same way, with the Cabibbo-Marinari algorithm for any N
(`--nc`, default 3; SU(2) is `--nc=2`), on a 4-dimensional lattice by default:

~~~
mpiexec -np 4 python -m pylattice.theories.gauge.sun --nc=3 --grid=4x4x8x8 --mpigrid=2x2x1x1 --steps=500 \
    --thermalization=100 --overrelax=4
mpiexec -np 4 python -m pylattice.theories.gauge.sun --nc=2 --grid=4x4x8x8 --mpigrid=2x2x1x1 --steps=500
~~~

Besides the common options (including `--thermalization` and `--measure-every`) they have:

| option | default | meaning |
|---|---|---|
| `--nc N` | `3` | the number of colors: the gauge group is SU(N) |
| `--beta B` | `2.3` for SU(2), `5.7` for SU(3) | the inverse coupling; for N >= 4 it must be given |
| `--algorithm` | `heatbath` | the update of every sweep: `heatbath` (always accepted: the exact heat bath for SU(2), the Cabibbo-Marinari one, SU(2) subgroup by subgroup, for SU(N)) or `metropolis` (the acceptance is measured) |
| `--overrelax N` | `0` | overrelaxation sweeps after it, in every sweep: they leave the action exactly unchanged and move the links far (3 to 5 is usual with the heat bath) |
| `--eps`, `--hits` | `0.4`, `1` | Metropolis only: size of the proposal in (0, 1], chosen for an acceptance of 0.4 to 0.6, and proposals per link and visit |
| `--init cold\|hot` | `cold` | the starting configuration (unless `--start-from`) |

SO(N) gauge theories (real orthogonal links with determinant 1; SO(2) = U(1)) run the same way with
`pylattice.theories.gauge.son`, with the same options (`--nc`, default 3) and the files named `so<N>_...`:

~~~
mpiexec -np 4 python -m pylattice.theories.gauge.son --nc=3 --beta=6.0 --grid=4x4x8x8 --mpigrid=2x2x1x1 --steps=500 \
    --thermalization=100 --overrelax=4
~~~

The action is the same, S = beta sum (1 - (1/N) tr P) with beta = 2N/g^2 (the convention of the SO(N) literature), and the
heat bath is the Cabibbo-Marinari one over the N(N-1)/2 rotation planes (SO(2) subgroups): in a plane the conditional weight
of a link is a von Mises distribution, drawn exactly; overrelaxation reflects the angle of each plane. `--beta` is required
except for SO(2) (default 1.0). The plaquette is real, so `plaq_im` is exactly 0. Offline measurements:
`python -m pylattice.theories.gauge.son_meas`, as for SU(N). Note that SO(3) is the adjoint-representation SU(2) theory,
whose 4-dimensional lattice has a bulk transition at weak coupling.

During the run they measure only the cheap observables: `plaq_re` and `plaq_im` (the average plaquette (1/N) tr P; the
imaginary part is zero up to rounding and checks the links), `action` per site, and the acceptance. The expensive ones are
measured offline on the stored configurations (`--checkpoint-every`, files `su<N>_grid<grid>_mpi<mpigrid>_b<beta>_<sweep>.hdf5`), see
[Measure stored configurations](#measure-stored-configurations). The first measurement of `su<N>_..._meas.json` is the
configuration after the thermalization.

---

## Running a simulation

### Command line options

| option | default | meaning |
|---|---|---|
| `--grid AxBx...` | `4x4` | lattice size of each rank (the global lattice is `grid` x `mpigrid`) |
| `--mpigrid AxBx...` | `1x2` | number of ranks along each direction (run with `mpiexec -np <product>`) |
| `--steps N` | `100` | Monte Carlo sweeps |
| `--beta B` | `0.5` | inverse temperature |
| `--seed S` | `0` | seed; every rank draws from its own independent stream of `S` |
| `--checkpoint-every N` | `0` (off) | save the configuration every N sweeps, and at the end |
| `--runlog-every N` | about 20 lines per run | log a line of diagnostics every N sweeps, on the terminal and in the log file (`0`: none) |
| `--start-from FILE.hdf5` | none | start from a saved configuration instead of a random one |
| `--debug` | off | log at DEBUG level, from every rank |
| `--logfile FILE` | `<run name>.log` | the log file, appended to if it exists (`off`: no file) |
| `--progress` | off | also draw a progress bar |

~~~
mpiexec -np 4 python -m pylattice.theories.spin.ising --grid=20x20 --mpigrid=2x2 --beta=1.5 --steps=2000 \
    --checkpoint-every=500 --runlog-every=100
~~~

The terminal shows the logo, a colored box for each section of the system and run information (system, parallel,
lattice, run), one line per `--runlog-every` sweeps, and a closing summary box. The log file holds the same, in plain
text.

### What a run writes

Every file starts with the **run name**, `<theory>_grid<grid>_mpi<mpigrid>_b<beta>`, so that it says what it is and where it
comes from: the theory (`ising`, `su2`, `su3`, `so3`, ...), the lattice of each rank as given to `--grid`, the ranks as given to
`--mpigrid`, and the inverse coupling. For the run above it is `ising_grid20x20_mpi2x2_b1.5`:

| file | when | content |
|---|---|---|
| `ising_grid20x20_mpi2x2_b1.5_meas.json` | always | every measurement, flushed as soon as it is made (see below) |
| `checkpoints/ising_grid20x20_mpi2x2_b1.5_<sweep>.hdf5` | `--checkpoint-every` | configuration (parallel HDF5) and the random state of every rank |
| `ising_grid20x20_mpi2x2_b1.5.log` | always (`--logfile`) | the log: logo, boxes with the system and run information, messages, one line per interval (below) and the summary |

A name given with `--logfile` is used as it is. The offline tools and `plot_meas` follow the same convention (below).

- The line per interval, e.g. `step=100 m=0.12 e=-1.91 mem_MB=312.4 cpu_pct=98.7 steps_per_s=211.30`, has the sweep (thermalization included), the observables, the resident memory of
  all the ranks together, the mean cpu use per rank (100 = one core fully busy) and the sweeps per second.
- The measurements file is valid JSON at every moment, so it can be read while the run goes on and it survives a job that
  dies. An existing one is overwritten.

  ~~~
  {"measurements": [
    {"sweep": 0, "m": 0.026, "e": 0.01, "acceptance": null},
    {"sweep": 1, "m": 0.056, "e": -0.957, "acceptance": 0.581}
  ]}
  ~~~

  Read it with `json.load(open('ising_grid20x20_mpi2x2_b1.5_meas.json'))['measurements']` (or `pandas.DataFrame(...)`). The sweep counts from
  the end of the thermalization; `acceptance` is that of the sweeps since the previous measurement. `m` and `e` are
  per site.

### Restart

~~~
mpiexec -np 4 python -m pylattice.theories.spin.ising --grid=20x20 --mpigrid=2x2 --beta=1.5 --steps=1500 \
    --start-from=checkpoints/ising_grid20x20_mpi2x2_b1.5_00000500.hdf5
~~~

The log shows what is on disk (type, lattice, dataset shape and dtype, who wrote it and when) and what is being
filled, and the run stops with a message if they differ (wrong type, wrong lattice, values that are not spins).

- The lattice must be the same (`--grid` x `--mpigrid`).
- Checkpoints hold the random state of every rank. On the **same rank layout** the restarted run continues exactly,
  bit for bit, as if it had not stopped. On another layout the configuration is loaded but the random numbers
  start afresh (the log warns).
- Checkpoints keep the precision of the field (int8 for the Ising spins, one byte each; double for SU(2) links and every other lattice by default): nothing is
  rounded on the way to the disk.
- The sweep counter starts afresh, and the new run writes its own files (the same names as the first run if `--grid`,
  `--mpigrid` and `--beta` are the same: run it in another directory, or the measurements file is overwritten).

---

## Working with the results

### Print and plot the measurements

~~~
python -m pylattice.utils.plot_meas ising_grid20x20_mpi2x2_b1.5_meas.json --skip 200 --outdir figures      # --no-plot: only the table
python -m pylattice.utils.plot_meas                                                 # the only *_meas*.json of this directory
~~~

It needs no MPI and works while the run is still going. For every observable and for the acceptance it prints the
number of points, the mean, its error (batch means) and the integrated autocorrelation time (Wolff's Gamma method),
and it writes one figure per quantity, named after the run (`figures/ising_grid20x20_mpi2x2_b1.5_m.png`, ...). `--skip` leaves out the
thermalization.

### Measure stored configurations

Measurements can be made afterwards on the checkpoints, without running any Monte Carlo:

~~~
mpiexec -np 4 python -m pylattice.theories.spin.ising_meas checkpoints/ising_grid20x20_mpi2x2_b1.5_*.hdf5 --mpigrid=2x2
~~~

Each file is first checked to be an Ising configuration of the right lattice (the structure found is logged). The
output is `ising_grid20x20_mpi2x2_b1.5_meas_offline.json` (the run name of the checkpoints, and `offline`, so that it does not overwrite the
measurements of the run; `--out` chooses another name) in the same format (sweep = the step of the checkpoint, no acceptance)
with `m`, `abs_m`, `m2`, `e` and `e2` per site, in the order of the steps. With `--logfile` the log goes to `ising_grid20x20_mpi2x2_b1.5_meas_offline.log`. The rank layout may differ from the run's. Other theories
plug in with their own `measure(path)` through `pylattice.core.measure.measure_files`.

For SU(N) (and, with `pylattice.theories.gauge.son_meas`, for SO(N)) the expensive observables are measured this way:

~~~
mpiexec -np 4 python -m pylattice.theories.gauge.sun_meas checkpoints/su3_grid4x4x8x8_mpi2x2x1x1_b5.7_*.hdf5 --beta=5.7 --mpigrid=2x2x1x1 --max-loop=3
~~~

`--beta` is the coupling of the run (the action depends on it and the checkpoints do not store it); the number of colors is
read from the files. Each file is checked to be an SU(N) configuration of the right lattice. The output has, per configuration: `plaq_re`, `plaq_im`, `action`;
`plaq_s` and `plaq_t` (the plaquette in the spatial planes and in the planes with the last direction, the time);
`polyakov` and `polyakov_abs` (the Polyakov loop along the last direction, so put the time last in `--grid`; complex for
N >= 3, where `polyakov` is its real part); the Wilson loops `W<R><T>` (R x T, R <= T <= `--max-loop`, the real part
averaged over the sites and orientations; `W11` is the plaquette); and
the Creutz ratios `creutz22`, `creutz33`, ... = -ln(W(R,R) W(R-1,R-1) / W(R-1,R)^2), nan for a configuration where
the argument is not positive. Average the loops over the configurations first (`plot_meas` prints the mean, error and
autocorrelation time of each) and take the ratio of the averages: that is the usual estimate.

---

## Repository layout

~~~
pylattice/
  core/            tensors, lattice fields on an MPI grid, parallel HDF5 I/O, command line, random streams,
                   statistics, Monte Carlo driver (the theories provide the sweep), callbacks
                   (checkpoints, run log, measurements file), offline measurements
    groups/        the groups the variables take values in, behind one interface (base.py: identity,
                   product, inverse, random, project, contains):
                     discrete.py  Z2 (the Ising spin), Z_q (clock, Potts-like), exact heat bath
                     lie.py       interface of the Lie groups (algebra, exp, log, near-identity)
                     su2.py       SU(2): algebra, exponential, quaternions (Creutz), heat bath of a link
                     su.py        SU(N): generators (Gell-Mann), f and d symbols, exp / log, Haar-random and
                                  near-identity elements, projection, and the SU(2) blocks of a
                                  Cabibbo-Marinari update
                     so.py        SO(N): generators, exp / log, Haar-random elements, and the SO(2) rotation
                                  planes of a Cabibbo-Marinari update (von Mises heat bath, overrelaxation)
                     gauge_observables.py  plaquettes, staples, loops, action and gauge transformation of a
                                  field of links, for any gauge group
  theories/
    spin/          variables on sites: Ising (simulation and offline measurements)
    gauge/         variables on links: SU(N) and SO(N) gauge theories
                     field.py     SUNField (and its SU(2) case SU2Field), SONField: the links of each group, with the
                                  observables of gauge_observables
                     sun.py       the Wilson-action theory (any N, SU(2) included): Cabibbo-Marinari heat bath,
                                  Metropolis and overrelaxation sweeps, and the command line
                     sun_meas.py  the loops and Creutz ratios, measured on stored configurations
                     son.py       the same for SO(N) (the sweeps with the rotation-plane steps)
                     son_meas.py  offline measurements of SO(N) configurations
  utils/           plot_meas: print and plot a measurements file
tests/             mirrors the package (tests/core, tests/theories, tests/utils)
benchmarks/        performance evaluation, parallel HDF5 demo
docs/              notes.tex / notes.pdf: the documentation
~~~

A theory is a class with `sweep(rng) -> (accepted, attempted)` and `observables() -> dict`; the driver
(`pylattice.core.montecarlo.run`) handles thermalization, measurements, progress and the callbacks. All the
randomness of a sweep comes from the generator it is given, which is what makes runs reproducible.

---

## Tests

Nothing runs the tests for you after an installation. To check that everything works on your machine, run
the script yourself (from the repository root, in the environment you installed):

~~~
python run_tests.py              # the standard routine: single process and 2 ranks (about 30 s)
python run_tests.py --quick      # only the single-process tests (about 15 s)
python run_tests.py --full       # also every MPI test on 1 and 4 ranks, and the cross-layout HDF5 test
python run_tests.py -k ising     # only the steps whose name contains "ising"
python run_tests.py --chunk gauge   # only one chunk of the tests: core, gauge or spin
python run_tests.py -v           # show the output of every step, not only of the failed ones
~~~

It prints the system it found (packages, `mpiexec`, cores, parallel `h5py`), one coloured line per step
(`PASS`, `FAIL` with the end of its output, or `SKIP` with the reason), and a summary; the exit status is 1 if
anything failed, so it can be used in your own scripts. The steps run in a scratch directory and leave no files
behind. They are the same checks as the GitHub workflows plus the ones that need 4 ranks. Without a parallel `h5py` the tests that write checkpoints (checkpoints, restart, offline
measurements, the HDF5 tests) are skipped and say so; the GitHub workflow has no parallel `h5py` either.

A new test file has to be added to the lists at the top of `run_tests.py`: a test checks that none is missing.

### Continuous integration

The tests are split into three chunks, each with its own GitHub workflow and badge (`.github/workflows/tests-<chunk>.yml`):

| chunk | tests |
|---|---|
| `core` | tensors, lattices and shifts, MPI, command line, random streams, statistics, Monte Carlo driver, checkpoints |
| `gauge` | the groups (Z2, SU(2), SU(N), SO(N)) and the SU(N) and SO(N) gauge theories: fields, sweeps, offline measurements |
| `spin` | the Ising model, its offline measurements, printing and plotting of measurements |

A push runs a chunk only if a file that the chunk covers changed: the source files its tests import, the tests, and the
configuration of the run. A change in the gauge theory runs `gauge` only; a change in the shared code (`pylattice/core/`)
runs the chunks that use it; documentation runs nothing. A badge shows the last run of its chunk, so it can be older than
the latest commit. Each workflow runs `python run_tests.py --chunk <chunk>`. `tests/utils/test_ci_paths.py` fails if a filter
misses a file that its chunk imports, and `tests/utils/test_run_tests.py` if a test file is in no chunk; the tests of
the repository itself run in every chunk. Which chunk a test belongs to is decided in `run_tests.py` (`chunk_of`).

To run a test file by hand (needs `pip install -e . --no-deps`, or `export PYTHONPATH=.`):

~~~
pytest tests/core/test_su2.py -q
mpiexec -n 2 python tests/theories/spin/test_ising.py          # the MPI scripts: 1, 2 or 4 ranks
mpiexec -n 4 python tests/core/test_hdf5io.py                  # needs exactly 4 ranks
mpiexec -n 2 python tests/core/test_grid_mpi.py                # needs exactly 2 ranks
~~~

---

## Performance

Strong scaling at fixed global size, best of 3 runs. "Naive" is a plain per-site `np.dot` loop without MPI.
The lattice classes act on all the sites at once (numpy arrays), which is what the "vectorized" curve shows; the
"loops" curves are the same operation as it was written before, one site at a time.

~~~
Grid 40x40x40x40

MatrixField = LatticeRealMatrix(lattice=Grid,N=2)
MatrixField2 = LatticeRealMatrix(lattice=Grid,N=2)
(MatrixField*MatrixField2).reducesum()

@ Intel(R) Core(TM) i5-8365U CPU @ 1.60GHz
  (4 cores, 2 threads per core)
@ Apple M4 Pro (10 performance + 4 efficiency cores)
~~~

On the Apple M4 Pro, vectorized (seconds, and speedup over one process):

| processes | 1 | 2 | 4 | 8 | 10 |
|---|---|---|---|---|---|
| time | 0.129 | 0.066 | 0.035 | 0.029 | 0.022 |
| speedup | 1.0 | 2.0 | 3.7 | 4.4 | 5.8 |

One vectorized process is about 11 times faster than the loops, and 9 times faster than the naive loop (1.18 s). The
scaling is near ideal up to 4 processes; after that each rank has too little work (about 0.03 s) and the fixed
costs of the call and of the reduction show. Threads do not help: the arithmetic is numpy ufuncs and batched 2x2
products, which do not use a threaded BLAS, so scale with MPI ranks (`OMP_NUM_THREADS` has no effect).

~~~
mpiexec -np 4 python benchmarks/performance_evaluation.py --mpigrid=2x2x1x1
~~~

![Performance comparison: Intel i5-8365U vs Apple M4 Pro](benchmarks/fig/perf_compare.png)

### Parallel HDF5 demo

~~~
mpiexec -n 4 python benchmarks/hdf5_demo.py
h5dump -n parallel_test.hdf5          # structure of the file
h5dump -D /test parallel_test.hdf5    # the data
~~~

---

## Validation

The average plaquette (1/N) Re tr U_p of the Wilson action, S = β Σ (1 − (1/N) Re tr U_p) with β = 2N/g², is compared with the
published values of Athenodorou and Teper, *SU(N) gauge theories in 3+1 dimensions: glueball spectrum, string tensions and topology*,
JHEP 12 (2021) 082, [arXiv:2106.00364](https://arxiv.org/abs/2106.00364) (Tables 1-4 of its appendix). It is the observable `plaq_re`
of this code, with the same β convention. More in
the [documentation](docs/notes.pdf).

| N | β | lattice (paper) | paper | lattice (this work) | this work |
|---|---|---|---|---|---|
| 2 | 2.60 | 30⁴ | 0.6700089(5) | 8⁴ | 0.670096(88) |
| 3 | 5.8941 | 14³×16 | 0.5810697(18) | 12⁴ | 0.581053(72) |
| 4 | 10.85 | 14³×20 | 0.5664268(15) | 12⁴ | 0.566385(47) |
| 5 | 17.43 | 14³×20 | 0.5685281(10) | 12⁴ | 0.568564(35) |

![Average plaquette of the paper and of this code as a function of β](docs/plaquette_vs_beta.png)

![Plaquette along the Monte Carlo for each N; the band is the value of the paper (mean ± error)](docs/plaquette_history.png)

### SO(N)

The same comparison for SO(N), in **2+1 dimensions**, with the values of Lau and Teper, *SO(N) gauge theories in 2+1 dimensions*,
JHEP 10 (2017) 022, [arXiv:1701.06941](https://arxiv.org/abs/1701.06941) (Tables 12-16 of its appendix; same action and β = 2N/g²).
Each β is above the bulk-transition interval of the paper (SO(4) [9.1, 10.2], SO(6) [18.0, 21.3], SO(8) [31, 35]).
The paper quotes the plaquette without error.

| N | β | lattice (paper) | paper | lattice (this work) | this work |
|---|---|---|---|---|---|
| 4 | 11.0 | 20²×28 | 0.80135 | 20²×28 | 0.801341(31) |
| 6 | 28.0 | 20²×28 | 0.80677 | 20²×28 | 0.806442(34) |
| 8 | 51.0 | 16²×24 | 0.80206 | 16²×24 | 0.801619(39) |

![Average plaquette of the paper and of this code as a function of β, SO(N)](docs/plaquette_vs_beta_so.png)

![Plaquette along the Monte Carlo for each SO(N); the line is the value of the paper](docs/plaquette_history_so.png)

---

## Citation

The error analysis (`pylattice.core.stats`) includes the code of
[py-uwerr](https://github.com/pretidav/py-uwerr) (Wolff's Gamma method, MIT license, see the header of the file).
If you publish results obtained with it, cite U. Wolff, *Monte Carlo errors with less errors*,
Comput. Phys. Commun. **156**, 143 (2004), [hep-lat/0306017](https://arxiv.org/abs/hep-lat/0306017).
