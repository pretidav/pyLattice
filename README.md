# pyLattice
Lattice Gauge Theory simulation library: 

![](https://github.com/pretidav/pyLattice/actions/workflows/testonbuild.yml/badge.svg)

[![codecov](https://codecov.io/gh/pretidav/pyLattice/branch/parallel/graph/badge.svg?token=KWS8S8BH3T)](https://codecov.io/gh/pretidav/pyLattice)

MatrixField = LatticeVectorRealMatrix(lattice=Grid,Nd=2,N=3)
MatrixField2 = LatticeVectorRealMatrix(lattice=Grid,Nd=2,N=3)
Grid 60x60x60x60
MatrixField*MatrixField2

Intel(R) Core(TM) i5-8365U CPU @ 1.60GHz
Thread(s) per core:  2
Core(s) per socket:  4

Queue 
1x1x1x1
--- 77.97438859939575 seconds ---
2x1x1x1
--- 46.04473257064819 seconds ---
2x2x1x1
--- 31.80805730819702 seconds ---
2x2x2x1
--- 38.320847511291504 seconds ---
2x2x2x2
--- 39.833889961242676 seconds ---
4x4x4x4
--- 39.63519358634949 seconds ---
Process
1x1x1x1
--- 87.73139786720276 seconds ---
2x1x1x1
--- 57.25099849700928 seconds ---
2x2x1x1
--- 43.5087628364563 seconds ---
2x2x2x1
--- 38.92410230636597 seconds ---
Thread
1x1x1x1
--- 79.55425310134888 seconds ---
2x1x1x1
--- 192.892076253891 seconds ---
2x2x1x1
--- 233.64544987678528 seconds ---
2x2x2x1
--- 285.8275969028473 seconds ---
Naive
1x1x1x1
--- 79.00875163078308 seconds ---