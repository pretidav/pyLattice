[![Run Python Tests](https://github.com/pretidav/pyLattice/actions/workflows/testonbuild.yml/badge.svg)](https://github.com/pretidav/pyLattice/actions/workflows/testonbuild.yml)

# pyLattice
A lattice (gauge) theory simulation library in python.  

<p align="center">
  <img width="460" height="300" src=(https://github.com/pretidav/pyLattice/raw/feature/parallelIO/fig/mylogo.png">
</p>

# Ising Model 

~~~
mpiexec -np 4 python ising_model.py --grid=20x20 --mpigrid=2x2 --beta=1.5 --steps=2000
~~~

## Performance Test 

~~~
Grid 40x40x40x40

MatrixField = LatticeRealMatrix(lattice=Grid,N=2)
MatrixField2 = LatticeRealMatrix(lattice=Grid,N=2)
(MatrixField*MatrixField2).reducesum()

@ Intel(R) Core(TM) i5-8365U CPU @ 1.60GHz

Thread(s) per core:  2
Core(s) per socket:  4
~~~

![alt text](https://github.com/pretidav/pyLattice/raw/main/fig/perf.png)


## MPI installation 

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
cd IO/
mpiexec -n 4 python demo.py
~~~

check hdf5 file output structure with 

~~~
h5dump -n parallel_test.hdf5
~~~

and inspect it with 

~~~
h5dump -D /test parallel_test.hdf5
~~~
