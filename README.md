[![Run Python Tests](https://github.com/pretidav/pyLattice/actions/workflows/testonbuild.yml/badge.svg?branch=refactor-SU2)](https://github.com/pretidav/pyLattice/actions/workflows/testonbuild.yml)

# pyLattice
<p align="center">
  <img width="150" height="150" src="docs/mylogo.png">
</p>
A lattice (gauge) theory simulation library in python.  

## Repository layout

~~~
pylattice/
  core/            tensors, lattice fields on an MPI grid, parallel HDF5 I/O, command line,
                   Monte Carlo driver (the theories provide the sweep)
    groups/        groups and algebras (SU(2) for now)
  theories/
    spin/          variables on sites: Ising
    gauge/         variables on links: SU(2) gauge field
tests/             mirrors the package (tests/core, tests/theories)
benchmarks/        performance evaluation, parallel HDF5 demo
docs/              notes
~~~

Run everything from the repository root, or install the package once with `pip install -e . --no-deps`
(after the environment below), which makes `pylattice` importable from any directory.

The error analysis (`pylattice.core.stats`) includes the code of
[py-uwerr](https://github.com/pretidav/py-uwerr) (Wolff's Gamma method, MIT license, see the header of the file).
If you publish results obtained with it, cite U. Wolff, *Monte Carlo errors with less errors*,
Comput. Phys. Commun. **156**, 143 (2004), [hep-lat/0306017](https://arxiv.org/abs/hep-lat/0306017).

# Ising Model 

~~~
mpiexec -np 4 python -m pylattice.theories.spin.ising --grid=20x20 --mpigrid=2x2 --beta=1.5 --steps=2000
~~~

Every run writes its measurements to `meas.json`, each one flushed to the file as soon as it is
computed (not at the end), so the file can be read while the run goes on and survives a job that dies:

~~~
{"measurements": [
  {"sweep": 0, "m": 0.026, "e": 0.01, "acceptance": null},
  {"sweep": 1, "m": 0.056, "e": -0.957, "acceptance": 0.581}
]}
~~~

It is valid JSON at every moment (`json.load(open('meas.json'))['measurements']`, or
`pandas.DataFrame(...)`); the sweep counts from the end of the thermalization, and `acceptance` is
that of the sweeps since the previous measurement. An existing `meas.json` is overwritten.

The run itself plots nothing: it produces configurations and `meas.json`. Look at the measurements offline,
while the run goes on or after it:

~~~
python -m pylattice.utils.plot_meas meas.json --skip 200 --outdir figures   # --no-plot: only the table
~~~

Measurements can also be made afterwards on stored configurations (the checkpoints), without running
any Monte Carlo; each file is first checked to be an Ising configuration of the right lattice, and the
structure found is logged:

~~~
mpiexec -np 4 python -m pylattice.theories.spin.ising_meas checkpoints/ising_*.hdf5 --mpigrid=2x2 --out meas_offline.json
~~~

It writes the same kind of `meas.json` (sweep = the step of the checkpoint, no acceptance) with `m`,
`abs_m`, `m2`, `e`, `e2` per site, in the order of the steps; the layout of the ranks may differ from the
run's. Other theories plug into `pylattice.core.measure.measure_files` with their own `measure(path)`.

`plot_meas` prints, for every observable and the acceptance, the mean, its error and the integrated
autocorrelation time, and writes one figure per quantity (`figures/m.png`, ...).

Two options, both off by default, record the run (they need `--steps` sweeps to be longer than the interval):

~~~
mpiexec -np 4 python -m pylattice.theories.spin.ising --grid=20x20 --mpigrid=2x2 --beta=1.5 --steps=2000 \
    --checkpoint-every=500 --runlog-every=100
~~~

- `--checkpoint-every N` saves the configuration (parallel HDF5, read it with `pylattice.core.hdf5io.Reader`)
  to `checkpoints/ising_<sweep>.hdf5` every N sweeps and at the end.
- `--start-from FILE.hdf5` starts from a saved configuration (e.g. `checkpoints/ising_00000500.hdf5`)
  instead of a random one. The log shows what is on disk (type, lattice, dataset shape and dtype, who
  wrote it and when) and what is being filled, and the run stops with a message if they differ. The
  lattice must be the same (`--grid` x `--mpigrid`). Checkpoints also hold the random state of every rank:
  restarting on the same rank layout continues the run exactly (bit for bit, as if it had not stopped);
  on another layout the configuration is loaded but the random numbers start afresh (the log warns).
  The sweep counter and `meas.json` start afresh.
- `--seed S` fixes the run: every rank draws from its own independent stream of `S`
  (`pylattice.core.rng`, numpy `SeedSequence`), so a run is reproducible for a given seed and rank layout.
- `--runlog-every N` appends a line to `run.log` every N sweeps:
  `step=100 m=0.12 e=-1.91 mem_MB=312.4 cpu_pct=98.7 steps_per_s=211.30`
  (sweeps since the start, the observables, resident memory of all the ranks, mean cpu use per rank,
  sweeps per second).

## Performance Test 

~~~
Grid 40x40x40x40

MatrixField = LatticeRealMatrix(lattice=Grid,N=2)
MatrixField2 = LatticeRealMatrix(lattice=Grid,N=2)
(MatrixField*MatrixField2).reducesum()

@ Intel(R) Core(TM) i5-8365U CPU @ 1.60GHz
  (4 cores, 2 threads per core)
@ Apple M4 Pro (10 performance + 4 efficiency cores)
~~~

Strong scaling at fixed global size, best of 3 runs. "Naive" is a plain per-site `np.dot` loop without MPI.

~~~
mpiexec -np 4 python benchmarks/performance_evaluation.py --mpigrid=2x2x1x1
~~~

![Performance comparison: Intel i5-8365U vs Apple M4 Pro](benchmarks/fig/perf_compare.png)


## macOS installation (Apple Silicon, conda-forge)

Tested on macOS with an Apple Silicon (arm64) Mac. Everything comes from conda-forge, so MPI, HDF5, `mpi4py` and `h5py` are built against each other and no `apt-get` or manual OpenMPI build is needed.

~~~
conda create -n pylattice -c conda-forge --override-channels \
  python=3.11 numpy scipy matplotlib psutil pytest rich openmpi mpi4py "h5py=*=mpi_openmpi*"
conda activate pylattice
pip install "plot_style @ git+https://github.com/pretidav/plot_style.git"   # style of the figures, not on conda
~~~

- `--override-channels` makes conda use only conda-forge and ignore the channels in your conda config. This avoids mixing packages from different channels.
- `plot_style` is not on conda-forge: it is installed from GitHub with `pip`, also by `pip install -r requirements.txt`.
- `pip install -r requirements.txt` alone is not enough: the standard `h5py` wheels from PyPI have no parallel HDF5 (`mpio`) support, and `scipy` and `h5py` are not listed in `requirements.txt`.

Check the installation:

~~~
python -c "import h5py; print(h5py.get_config().mpi)"   # must print True
export PYTHONPATH=.          # run from the repository root (not needed after pip install -e . --no-deps)
pytest tests/core/test_tensors.py
python tests/core/test_grid.py
mpiexec -n 2 python tests/core/test_grid_mpi.py
mpiexec -n 4 python tests/core/test_hdf5io.py
pytest tests/core/test_rng.py tests/core/test_stats.py
mpiexec -n 2 python tests/core/test_rng.py              # one independent random stream per rank
pytest tests/core/test_montecarlo.py tests/core/test_callbacks.py
mpiexec -n 2 python tests/core/test_callbacks.py        # checkpoints and run log (the checkpoint tests need parallel h5py)
mpiexec -n 2 python tests/core/test_montecarlo.py   # the driver, also on a split job
pytest tests/theories/spin/test_ising.py
mpiexec -n 2 python tests/theories/spin/test_ising.py      # same checks on a lattice split over 2 (or 1, 4) ranks
~~~

`test_hdf5io.py` uses a 2x2 process grid, so it needs exactly 4 processes. On a Mac with fewer than 4 physical cores, add `--oversubscribe` to `mpiexec`.

## MPI installation (Linux)

### install OpenMPI 
if c++ compiler is not present: 
~~~
sudo apt-get install g++
~~~

if make is not present: 
~~~
sudo apt install make
~~~
then follow: 
~~~
https://edu.itp.phys.ethz.ch/hs12/programming_techniques/openmpi.pdf
~~~

### install mpi4py - openmpi 
~~~
conda install -c conda-forge mpi4py openmpi
~~~

### note 
mpi takes advantage of physical CPUs only. 
This number can be checked with 
~~~
import psutil 
print(psutil.cpu_count(logical=False))
~~~

## HDF5 installation 

~~~
conda install -c conda-forge "h5py>=2.9=mpi*"
~~~

Run test script 

~~~
mpiexec -n 4 python benchmarks/hdf5_demo.py
~~~

check hdf5 file output structure with 

~~~
h5dump -n parallel_test.hdf5
~~~

and inspect it with 

~~~
h5dump -D /test parallel_test.hdf5
~~~
