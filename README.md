# pyLattice
Lattice Gauge Theory simulation library: 

[![Run Python Tests](https://github.com/pretidav/pyLattice/actions/workflows/testonbuild.yml/badge.svg)](https://github.com/pretidav/pyLattice/actions/workflows/testonbuild.yml)

# Performance Test 

~~~
Grid 40x40x40x40

MatrixField = LatticeRealMatrix(lattice=Grid,Nd=2,N=3)
MatrixField2 = LatticeRealMatrix(lattice=Grid,Nd=2,N=3)
(MatrixField*MatrixField2).reducesum()

@ Intel(R) Core(TM) i5-8365U CPU @ 1.60GHz

Thread(s) per core:  2
Core(s) per socket:  4
~~~

![alt text](https://github.com/pretidav/pyLattice/raw/main/fig/perf.png)