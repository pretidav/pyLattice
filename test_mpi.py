from utils.inputargs import Parser
from mpi.parallel import CartesianComm, pprint
from lattice.grid import LatticeReal, LatticeComplex
import numpy as np



def test_LatticeReal(grid,comm):
    tol = 10e-6
    Lattice = LatticeReal(grid=grid,cartesiancomm=comm)
    Lattice.fill_value(n=np.array([1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],dtype='float32'))
    assert(np.sum(Lattice.value - np.array([1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9]))<tol)
    assert( np.sum(Lattice.tensor_idx - np.array([[0,1,2],[3,4,5],[6,7,8]]))<tol )
    Lattice.movebackward(mu=0)
    assert( np.sum(Lattice.tensor_idx - np.array([[3,4,5],[6,7,8],[0,1,2]]))<tol)
    assert( np.sum(Lattice.value - np.array([1.4,1.5,1.6,1.7,1.8,1.9,1.1,1.2,1.3]))<tol)
    Lattice.moveforward(mu=0)
    assert( np.sum(Lattice.value - np.array([1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9]))<tol)
    assert( np.sum(Lattice.tensor_idx - np.array([[0,1,2],[3,4,5],[6,7,8]]))<tol)
    assert( (Lattice.reducesum() - 13.5*np.prod(comm.mpigrid))<tol)
    assert( (Lattice.average() - 1.5)<tol)
    Lattice.fill_value(2)
    Lattice2 = LatticeReal(grid=grid,cartesiancomm=comm)
    Lattice2.fill_value(3)
    assert( np.sum(((Lattice + Lattice2).value)-np.array([5., 5., 5., 5., 5., 5., 5., 5., 5.]))<tol)    
    assert( np.sum(((Lattice2 - Lattice).value)-np.array([1., 1., 1., 1., 1., 1., 1., 1., 1.]))<tol)
    assert( np.sum(((Lattice*Lattice2).value)-np.array([6., 6., 6., 6., 6., 6., 6., 6., 6.]))<tol)
    pprint(comm=comm.comm,msg='[+] TEST grid={} mpigrid={}: LatticeReal passed'.format('x'.join([str(a) for a in grid]),'x'.join([str(a) for a in CC.mpigrid])))

def test_LatticeComplex(grid,comm):
    tol = 10e-6
    Lattice = LatticeComplex(grid=grid,cartesiancomm=comm)
    Lattice.fill_value(n=np.array([1.1,1.2j,1.3,1.4j,1.5,1.6j,1.7,1.8j,1.9],dtype='complex64'))
    assert(np.sum(Lattice.value - np.array([1.1,1.2j,1.3,1.4j,1.5,1.6j,1.7,1.8j,1.9]))<tol)
    assert( np.sum(Lattice.tensor_idx - np.array([[0,1,2],[3,4,5],[6,7,8]]))<tol )
    Lattice.movebackward(mu=0)
    print( Lattice.value)
    
    assert( np.sum(Lattice.tensor_idx - np.array([[3,4,5],[6,7,8],[0,1,2]]))<tol)
    assert( np.sum(Lattice.value - np.array([1.4j,1.5,1.6j,1.7,1.8j,1.9,1.1,1.2j,1.3]))<tol)
    Lattice.moveforward(mu=0)
    assert( np.sum(Lattice.value - np.array([1.1,1.2j,1.3,1.4j,1.5,1.6j,1.7,1.8j,1.9]))<tol)
    assert( np.sum(Lattice.tensor_idx - np.array([[0,1,2],[3,4,5],[6,7,8]]))<tol)

    ComplexField = LatticeComplex(grid=grid,cartesiancomm=comm)
    ComplexField.fill_value(1j)
    ComplexField2 = LatticeComplex(grid=grid, cartesiancomm=comm)
    ComplexField2.fill_value(3)
    assert(np.abs(np.sum(ComplexField.value-np.array([1j]*3*3)))<tol)
    assert(np.abs(np.sum((ComplexField*ComplexField2).value-3j ))<tol)
    assert(np.abs(np.sum((ComplexField+ComplexField2).value-np.array([3+1j]*3*3)))<tol)
    assert(np.abs(np.sum((ComplexField-ComplexField2).value-np.array([-3+1j]*3*3)))<tol)
    assert( (ComplexField.reducesum() - 9j*np.prod(comm.mpigrid))<tol)
    assert( (ComplexField.average() - 1j)<tol)
    pprint(comm=comm.comm,msg='[+] TEST grid={} mpigrid={}: LatticeComplex passed'.format('x'.join([str(a) for a in grid]),'x'.join([str(a) for a in CC.mpigrid])))

if __name__ == '__main__':
    PP = Parser()    
    CC = CartesianComm(mpigrid=PP.mpigrid)
    test_LatticeReal(grid=PP.grid,comm=CC)
    test_LatticeComplex(grid=PP.grid,comm=CC)