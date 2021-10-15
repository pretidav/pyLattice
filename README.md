# pyLattice
Lattice Gauge Theory simulation library: 

[![Run Python Tests](https://github.com/pretidav/pyLattice/actions/workflows/testonbuild.yml/badge.svg)](https://github.com/pretidav/pyLattice/actions/workflows/testonbuild.yml)
[![codecov](https://codecov.io/gh/pretidav/pyLattice/branch/parallel/graph/badge.svg?token=KWS8S8BH3T)](https://codecov.io/gh/pretidav/pyLattice)
[![CodeQL](https://github.com/pretidav/pyLattice/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/pretidav/pyLattice/actions/workflows/codeql-analysis.yml)

MatrixField = LatticeVectorRealMatrix(lattice=Grid,Nd=2,N=3)
MatrixField2 = LatticeVectorRealMatrix(lattice=Grid,Nd=2,N=3)
Grid 60x60x60x60
MatrixField*MatrixField2

Intel(R) Core(TM) i5-8365U CPU @ 1.60GHz
Thread(s) per core:  2
Core(s) per socket:  4

![alt text](https://github.com/pretidav/pyLattice/raw/parallel/fig/perf.png)