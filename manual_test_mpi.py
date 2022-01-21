from utils.inputargs import Parser
from mpi.parallel import CartesianComm, pprint
from lattice.grid import LatticeReal, LatticeComplex
import numpy as np



def test_LatticeReal(grid,comm):
    tol = 10e-6
    Lattice = LatticeReal(grid=grid,cartesiancomm=comm)
    val = np.array([1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],dtype='float32') + comm.rank
    Lattice.fill_value(n=val)
    test1= Lattice.value - np.array([1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9]) - comm.rank
    assert( (np.abs(test1)<tol).all())
    Lattice.movebackward(mu=0)
    Lattice.moveforward(mu=0)
    assert( (np.abs(test1)<tol).all())
    Lattice.fill_value(n=np.array([1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],dtype='float32'))
    assert( (np.abs(Lattice.reducesum() - 13.5*np.prod(comm.mpigrid))<tol) ) 
    assert( (np.abs(Lattice.average() - 1.5)<tol) ) 
    Lattice.fill_value(2)
    Lattice2 = LatticeReal(grid=grid,cartesiancomm=comm)
    Lattice2.fill_value(3)
    assert( (np.abs(((Lattice + Lattice2).value)-np.array([5., 5., 5., 5., 5., 5., 5., 5., 5.]))<tol).all())    
    assert( (np.abs(((Lattice2 - Lattice).value)-np.array([1., 1., 1., 1., 1., 1., 1., 1., 1.]))<tol).all())
    assert( (np.abs(((Lattice*Lattice2).value)-np.array([6., 6., 6., 6., 6., 6., 6., 6., 6.]))<tol).all())
    pprint(comm=comm.comm,msg='[+] TEST grid={} mpigrid={}: LatticeReal passed'.format('x'.join([str(a) for a in grid]),'x'.join([str(a) for a in CC.mpigrid])))

def test_LatticeComplex(grid,comm):
    tol = 10e-6
    Lattice = LatticeComplex(grid=grid,cartesiancomm=comm)
    val = np.array([1.1,1.2j,1.3,1.4j,1.5,1.6j,1.7,1.8j,1.9],dtype='complex64') + comm.rank
    Lattice.fill_value(n=val)
    test1= Lattice.value - np.array([1.1,1.2j,1.3,1.4j,1.5,1.6j,1.7,1.8j,1.9]) - comm.rank
    assert( (np.abs(test1)<tol).all())
    Lattice.movebackward(mu=0)
    Lattice.moveforward(mu=0)
    assert( (np.abs(test1)<tol).all())
    ComplexField = LatticeComplex(grid=grid,cartesiancomm=comm)
    ComplexField.fill_value(1j)
    ComplexField2 = LatticeComplex(grid=grid, cartesiancomm=comm)
    ComplexField2.fill_value(3)
    assert( (np.abs(ComplexField.value-np.array([1j]*3*3))<tol).all())
    assert( (np.abs((ComplexField*ComplexField2).value-3j)<tol).all())
    assert( (np.abs((ComplexField+ComplexField2).value-np.array([3+1j]*3*3))<tol).all())
    assert( (np.abs((ComplexField-ComplexField2).value-np.array([-3+1j]*3*3))<tol).all())
    assert( np.abs(ComplexField.reducesum() - 9j*np.prod(comm.mpigrid))<tol)
    assert( np.abs(ComplexField.average() - 1j)<tol)
    pprint(comm=comm.comm,msg='[+] TEST grid={} mpigrid={}: LatticeComplex passed'.format('x'.join([str(a) for a in grid]),'x'.join([str(a) for a in CC.mpigrid])))

if __name__ == '__main__':
    PP = Parser()    
    CC = CartesianComm(mpigrid=PP.mpigrid)
    test_LatticeReal(grid=PP.grid,comm=CC)
    test_LatticeComplex(grid=PP.grid,comm=CC)