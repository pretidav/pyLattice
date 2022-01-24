# pyLattice
Lattice Gauge Theory simulation library: 

[![Run Python Tests](https://github.com/pretidav/pyLattice/actions/workflows/testonbuild.yml/badge.svg)](https://github.com/pretidav/pyLattice/actions/workflows/testonbuild.yml)


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