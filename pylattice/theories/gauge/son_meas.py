"""Measures SO(N) gauge configurations that are stored on disk (the checkpoints of a run of pylattice.theories.gauge.son) and writes
<run name>_meas_offline.json (the run name of the checkpoints: so<N>_grid<grid>_mpi<mpigrid>_b<beta>), as a run does:

    mpiexec -np 4 python -m pylattice.theories.gauge.son_meas checkpoints/so3_grid4x4x8x8_mpi2x2x1x1_b6_*.hdf5 --beta=6 --mpigrid=2x2x1x1

The observables and the options are those of pylattice.theories.gauge.sun_meas (plaquette, spatial and temporal plaquettes, Polyakov
loop, Wilson loops, Creutz ratios), computed on the SO(N) field; a file that does not hold SO(N) matrices (orthogonal, determinant 1) on
the lattice of the others stops the run. All the traces are real, so plaq_im is exactly 0 and the Polyakov loop is real.
"""
from pylattice.theories.gauge.sun_meas import main as run


def main(argv=None):
    run(argv, group='so', module='son_meas')


if __name__ == '__main__':
    main()
