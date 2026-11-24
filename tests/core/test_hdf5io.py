from pylattice.core.hdf5io import Writer, Reader
from pylattice.core.parallel import CartesianComm
from pylattice.core.grid import LatticeComplex, LatticeReal, LatticeComplexMatrix, LatticeRealMatrix, LatticeVectorReal, LatticeVectorComplex, LatticeVectorRealMatrix, LatticeVectorComplexMatrix
import numpy as np
import pytest

tol = 10e-6
mpigrid = [2, 2]
grid = [3,3]
CC = CartesianComm(mpigrid=mpigrid)

def test_latticereal():
    Lattice = LatticeReal(grid=grid, cartesiancomm=CC)
    Lattice.fill_value(n=np.array(
        [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9], dtype='float32')+CC.rank)
    
    W = Writer(lattice=Lattice,cartesiancomm=CC)
    R = Reader(cartesiancomm=CC)
    W.write(filename='./test_file')
    Lattice2 = R.read(filename='./test_file')
    
    assert(( np.abs(Lattice.value-Lattice2.value)<tol).all()) 

def test_latticecomplex():
    Lattice = LatticeComplex(grid=grid, cartesiancomm=CC)
    Lattice.fill_value(n=np.array(
        [1.1, 1.2j, 1.3, 1.4j, 1.5, 1.6j, 1.7, 1.8j, 1.9], dtype='complex64')+CC.rank)
    
    W = Writer(lattice=Lattice,cartesiancomm=CC)
    R = Reader(cartesiancomm=CC)
    W.write(filename='./test_file')
    Lattice2 = R.read(filename='./test_file')

    assert(( np.abs(Lattice.value-Lattice2.value)<tol).all()) 

def test_latticerealmatrix():
    Lattice = LatticeRealMatrix(grid=grid, cartesiancomm=CC, N=2)
    test = np.array([[[1, 2], [3, 4]],     [[5, 6], [7, 8]],     [[9, 10], [11, 12]],
                     [[13, 14], [15, 16]], [[17, 18], [
                         19, 20]], [[21, 22], [23, 24]],
                     [[25, 26], [27, 28]], [[29, 30], [31, 32]], [[33, 34], [35, 36]]], dtype='float32')
    Lattice.fill_value(n=test)
    W = Writer(lattice=Lattice,cartesiancomm=CC)
    R = Reader(cartesiancomm=CC)
    W.write(filename='./test_file')
    Lattice2 = R.read(filename='./test_file')

    assert(( np.abs(Lattice.value-Lattice2.value)<tol).all()) 

def test_latticecomplexmatrix():
    Lattice = LatticeComplexMatrix(grid=grid, cartesiancomm=CC, N=2)
    test = np.array([[[1, 2j], [3, 4]],     [[5, 6j], [7, 8]],     [[9, 10j], [11, 12]],
                     [[13, 14j], [15j, 16]], [[17, 18j], [
                         19, 20j]], [[21, 22j], [23, 24]],
                     [[25, 26j], [27, 28]], [[29, 30j], [31, 32]], [[33, 34j], [35, 36]]], dtype='complex64')
    Lattice.fill_value(n=test)
    W = Writer(lattice=Lattice,cartesiancomm=CC)
    R = Reader(cartesiancomm=CC)
    W.write(filename='./test_file')
    Lattice2 = R.read(filename='./test_file')

    assert(( np.abs(Lattice.value-Lattice2.value)<tol).all()) 

def test_latticerealvector():
    Lattice = LatticeVectorReal(grid=grid, cartesiancomm=CC, Nd=3)
    test = np.array([1, 2, 3], dtype='float32')
    Lattice.fill_value(n=test)
    W = Writer(lattice=Lattice,cartesiancomm=CC)
    R = Reader(cartesiancomm=CC)
    W.write(filename='./test_file')
    Lattice2 = R.read(filename='./test_file')

    assert(( np.abs(Lattice.value-Lattice2.value)<tol).all()) 


def test_latticecomplexvector():
    Lattice = LatticeVectorComplex(grid=grid, cartesiancomm=CC, Nd=3)
    test = np.array([1, 2j, 3], dtype='complex64')
    Lattice.fill_value(n=test)
    W = Writer(lattice=Lattice,cartesiancomm=CC)
    R = Reader(cartesiancomm=CC)
    W.write(filename='./test_file')
    Lattice2 = R.read(filename='./test_file')

    assert(( np.abs(Lattice.value-Lattice2.value)<tol).all()) 


def test_latticerealvectormatrix():
    Lattice = LatticeVectorRealMatrix(grid=grid, cartesiancomm=CC, N=2, Nd=2)
    test = np.reshape(
        np.array([int(i) for i in range(3*3*2*2*2)], dtype='float32')+CC.rank, (9, 2, 2, 2))

    Lattice.fill_value(n=test)
    W = Writer(lattice=Lattice,cartesiancomm=CC)
    R = Reader(cartesiancomm=CC)
    W.write(filename='./test_file')
    Lattice2 = R.read(filename='./test_file')

    assert(( np.abs(Lattice.value-Lattice2.value)<tol).all()) 


def test_latticecomplexvectormatrix():
    Lattice = LatticeVectorComplexMatrix(
        grid=grid, cartesiancomm=CC, N=2, Nd=2)
    test = np.reshape(np.array([complex(i) for i in range(
        3*3*2*2*2)], dtype='complex64')+CC.rank, (9, 2, 2, 2))

    Lattice.fill_value(n=test)
    W = Writer(lattice=Lattice,cartesiancomm=CC)
    R = Reader(cartesiancomm=CC)
    W.write(filename='./test_file')
    Lattice2 = R.read(filename='./test_file')

    assert(( np.abs(Lattice.value-Lattice2.value)<tol).all()) 


class MyField(LatticeReal):
    pass


def test_latticesubclass():
    # a subclass (an IsingField, an SU2Field) is stored as the lattice class it derives from
    import h5py
    Lattice = MyField(grid=grid, cartesiancomm=CC)
    Lattice.fill_value(n=np.array(
        [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9], dtype='float32')+CC.rank)
    W = Writer(lattice=Lattice,cartesiancomm=CC)
    R = Reader(cartesiancomm=CC)
    W.write(filename='./test_file')
    CC.comm.Barrier()
    with h5py.File('./test_file.hdf5', 'r') as f:
        assert f.attrs['type'] == 'LatticeReal'
    Lattice2 = R.read(filename='./test_file')
    assert type(Lattice2) is LatticeReal
    assert Lattice2.value.shape == Lattice.value.shape and (np.abs(Lattice.value-Lattice2.value)<tol).all()
    class NotALattice:
        grid = np.array(grid)
    class FakeFile:
        attrs = {}
    with pytest.raises(TypeError):                              # something that is not a lattice
        Writer(lattice=NotALattice(), cartesiancomm=CC).write_header(file=FakeFile())

def full_precision(shape, seed, cplx):
    x = np.random.default_rng(seed).standard_normal(shape)
    return (x + 1j * np.random.default_rng(seed + 1).standard_normal(shape)) if cplx else x


def test_double_precision_is_kept_on_disk_and_read_back_exactly():
    import h5py
    for cls, kwargs, cplx in ((LatticeComplex, {}, True), (LatticeReal, {}, False),
                              (LatticeVectorComplexMatrix, dict(Nd=2, N=2), True)):
        Lattice = cls(grid=grid, cartesiancomm=CC, **kwargs)
        Lattice.value = full_precision(Lattice.value.shape, 10 + CC.rank, cplx).astype(complex if cplx else float)
        Writer(lattice=Lattice, cartesiancomm=CC).write(filename='./test_file_double')
        CC.comm.Barrier()
        with h5py.File('./test_file_double.hdf5', 'r') as f:
            assert f['value'].dtype == Lattice.value.dtype                        # not rounded to float32 / complex64
        Lattice2 = Reader(cartesiancomm=CC).read(filename='./test_file_double')
        assert Lattice2.value.dtype == Lattice.value.dtype
        assert np.array_equal(Lattice.value, Lattice2.value)                     # bit for bit
        CC.comm.Barrier()


def test_single_precision_lattices_still_give_single_precision_files():
    import h5py
    for cls, name in ((LatticeReal, 'float32'), (LatticeComplex, 'complex64')):
        Writer(lattice=cls(grid=grid, cartesiancomm=CC), cartesiancomm=CC).write(filename='./test_file_single')
        CC.comm.Barrier()
        with h5py.File('./test_file_single.hdf5', 'r') as f:
            assert f['value'].dtype == np.dtype(name)
        CC.comm.Barrier()


def test_a_lattice_of_integers_cannot_be_written():
    Lattice = LatticeReal(grid=grid, cartesiancomm=CC)
    Lattice.value = Lattice.value.astype(int)
    with pytest.raises(TypeError, match='cannot write'):
        Writer(lattice=Lattice, cartesiancomm=CC).write(filename='./test_file_int')


def test_load_into_keeps_double_precision_and_warns_about_a_file_rounded_when_written():
    from pylattice.core.hdf5io import load_into
    Lattice = LatticeComplex(grid=grid, cartesiancomm=CC)
    Lattice.value = full_precision(Lattice.value.shape, 20 + CC.rank, True).astype(complex)
    Writer(lattice=Lattice, cartesiancomm=CC).write(filename='./test_file_double')
    CC.comm.Barrier()
    target, lines = LatticeComplex(grid=grid, cartesiancomm=CC), []
    target.value = np.zeros(target.value.shape, dtype=complex)
    load_into(target, './test_file_double.hdf5', log=lines.append)
    assert np.array_equal(target.value, Lattice.value)
    assert not any('WARNING' in l for l in lines) and any('complex128' in l for l in lines)
    single = LatticeComplex(grid=grid, cartesiancomm=CC)                           # a file written in single precision
    single.value = Lattice.value.astype('complex64')
    Writer(lattice=single, cartesiancomm=CC).write(filename='./test_file_single')
    CC.comm.Barrier()
    lines.clear()
    load_into(target, './test_file_single.hdf5', log=lines.append)
    assert any('WARNING' in l and 'rounded' in l for l in lines)
    assert np.array_equal(target.value, single.value.astype(complex))              # what is on disk, exactly
    real = LatticeReal(grid=grid, cartesiancomm=CC)                                # real numbers in a complex field: refused
    with pytest.raises(ValueError):
        load_into(real, './test_file_double.hdf5')


if __name__=='__main__': 
    test_latticereal()
    test_latticecomplex()
    test_latticerealmatrix()
    test_latticecomplexmatrix()
    test_latticerealvector()
    test_latticecomplexvector()
    test_latticerealvectormatrix()
    test_latticecomplexvectormatrix()
    test_latticesubclass()
    test_double_precision_is_kept_on_disk_and_read_back_exactly()
    test_single_precision_lattices_still_give_single_precision_files()
    test_a_lattice_of_integers_cannot_be_written()
    test_load_into_keeps_double_precision_and_warns_about_a_file_rounded_when_written()
