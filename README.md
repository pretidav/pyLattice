[![Run Python Tests](https://github.com/pretidav/pyLattice/actions/workflows/testonbuild.yml/badge.svg?branch=su2)](https://github.com/pretidav/pyLattice/actions/workflows/testonbuild.yml)

# pyLattice
<p align="center">
  <img width="150" height="150" src="docs/mylogo.png">
</p>

A lattice (gauge) theory simulation library in Python, parallel with MPI. A lattice is split over MPI ranks,
a theory provides the Monte Carlo sweep, and the library provides the rest: checkpoints and exact restarts,
logs, measurements written as the run goes, and offline analysis.

Implemented: the Ising model (spin theories on sites, with values in the group Z2) and SU(2): the group (algebra,
quaternionic representation of Creutz, link heat bath, SU(N) from SU(2) blocks) and a field of links; the gauge
Monte Carlo itself (Wilson action, staples) is not written yet.

**Contents:** [Installation](#installation) · [Quick start](#quick-start) · [Running a simulation](#running-a-simulation) ·
[Working with the results](#working-with-the-results) · [Repository layout](#repository-layout) · [Tests](#tests) ·
[Performance](#performance) · [Citation](#citation)

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
the system and of the run (OS, architecture, MPI, `h5py`, lattice, parameters), shows a progress bar, and writes
`meas.json`. It plots nothing: plotting is done offline, see [below](#working-with-the-results).

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
| `--logfile FILE` | `run.log` | the log file, appended to if it exists (`off`: no file) |
| `--progress` | off | also draw a progress bar |

~~~
mpiexec -np 4 python -m pylattice.theories.spin.ising --grid=20x20 --mpigrid=2x2 --beta=1.5 --steps=2000 \
    --checkpoint-every=500 --runlog-every=100
~~~

The terminal shows the logo, a colored box for each section of the system and run information (system, parallel,
lattice, run), one line per `--runlog-every` sweeps, and a closing summary box. The log file holds the same, in plain
text.

### What a run writes

| file | when | content |
|---|---|---|
| `meas.json` | always | every measurement, flushed as soon as it is made (see below) |
| `checkpoints/ising_<sweep>.hdf5` | `--checkpoint-every` | configuration (parallel HDF5) and the random state of every rank |
| `run.log` | always (`--logfile`) | the log: logo, boxes with the system and run information, messages, one line per interval (below) and the summary |

- The line per interval, e.g. `step=100 m=0.12 e=-1.91 mem_MB=312.4 cpu_pct=98.7 steps_per_s=211.30`, has the sweep (thermalization included), the observables, the resident memory of
  all the ranks together, the mean cpu use per rank (100 = one core fully busy) and the sweeps per second.
- `meas.json` is valid JSON at every moment, so it can be read while the run goes on and it survives a job that
  dies. An existing one is overwritten.

  ~~~
  {"measurements": [
    {"sweep": 0, "m": 0.026, "e": 0.01, "acceptance": null},
    {"sweep": 1, "m": 0.056, "e": -0.957, "acceptance": 0.581}
  ]}
  ~~~

  Read it with `json.load(open('meas.json'))['measurements']` (or `pandas.DataFrame(...)`). The sweep counts from
  the end of the thermalization; `acceptance` is that of the sweeps since the previous measurement. `m` and `e` are
  per site.

### Restart

~~~
mpiexec -np 4 python -m pylattice.theories.spin.ising --grid=20x20 --mpigrid=2x2 --beta=1.5 --steps=1500 \
    --start-from=checkpoints/ising_00000500.hdf5
~~~

The log shows what is on disk (type, lattice, dataset shape and dtype, who wrote it and when) and what is being
filled, and the run stops with a message if they differ (wrong type, wrong lattice, values that are not spins).

- The lattice must be the same (`--grid` x `--mpigrid`).
- Checkpoints hold the random state of every rank. On the **same rank layout** the restarted run continues exactly,
  bit for bit, as if it had not stopped. On another layout the configuration is loaded but the random numbers
  start afresh (the log warns).
- Checkpoints keep the precision of the field (int8 for the Ising spins, one byte each; double for SU(2) links and every other lattice by default): nothing is
  rounded on the way to the disk.
- The sweep counter and `meas.json` start afresh.

---

## Working with the results

### Print and plot the measurements

~~~
python -m pylattice.utils.plot_meas meas.json --skip 200 --outdir figures      # --no-plot: only the table
~~~

It needs no MPI and works while the run is still going. For every observable and for the acceptance it prints the
number of points, the mean, its error (batch means) and the integrated autocorrelation time (Wolff's Gamma method),
and it writes one figure per quantity (`figures/m.png`, ...). `--skip` leaves out the thermalization.

### Measure stored configurations

Measurements can be made afterwards on the checkpoints, without running any Monte Carlo:

~~~
mpiexec -np 4 python -m pylattice.theories.spin.ising_meas checkpoints/ising_*.hdf5 --mpigrid=2x2 --out meas_offline.json
~~~

Each file is first checked to be an Ising configuration of the right lattice (the structure found is logged). The
output is a `meas.json` in the same format (sweep = the step of the checkpoint, no acceptance) with `m`, `abs_m`,
`m2`, `e` and `e2` per site, in the order of the steps. The rank layout may differ from the run's. Other theories
plug in with their own `measure(path)` through `pylattice.core.measure.measure_files`.

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
                     su.py        SU(N) from its SU(2) subgroups (Cabibbo-Marinari)
  theories/
    spin/          variables on sites: Ising (simulation and offline measurements)
    gauge/         variables on links: SU(2) gauge field
  utils/           plot_meas: print and plot a meas.json
tests/             mirrors the package (tests/core, tests/theories, tests/utils)
benchmarks/        performance evaluation, parallel HDF5 demo
docs/              notes
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
python run_tests.py -v           # show the output of every step, not only of the failed ones
~~~

It prints the system it found (packages, `mpiexec`, cores, parallel `h5py`), one coloured line per step
(`PASS`, `FAIL` with the end of its output, or `SKIP` with the reason), and a summary; the exit status is 1 if
anything failed, so it can be used in your own scripts. The steps run in a scratch directory and leave no files
behind. They are the same checks as the GitHub workflow (`.github/workflows/testonbuild.yml`) plus the ones that
need 4 ranks. Without a parallel `h5py` the tests that write checkpoints (checkpoints, restart, offline
measurements, the HDF5 tests) are skipped and say so; the GitHub workflow has no parallel `h5py` either.

A new test file has to be added to the lists at the top of `run_tests.py`: a test checks that none is missing.

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

~~~
Grid 40x40x40x40

MatrixField = LatticeRealMatrix(lattice=Grid,N=2)
MatrixField2 = LatticeRealMatrix(lattice=Grid,N=2)
(MatrixField*MatrixField2).reducesum()

@ Intel(R) Core(TM) i5-8365U CPU @ 1.60GHz
  (4 cores, 2 threads per core)
@ Apple M4 Pro (10 performance + 4 efficiency cores)
~~~

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

## Citation

The error analysis (`pylattice.core.stats`) includes the code of
[py-uwerr](https://github.com/pretidav/py-uwerr) (Wolff's Gamma method, MIT license, see the header of the file).
If you publish results obtained with it, cite U. Wolff, *Monte Carlo errors with less errors*,
Comput. Phys. Commun. **156**, 143 (2004), [hep-lat/0306017](https://arxiv.org/abs/hep-lat/0306017).
