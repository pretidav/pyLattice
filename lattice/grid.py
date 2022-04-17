from mpi.parallel import *
from linalg.tensors import *
import numpy as np


class LatticeBase():
    def __init__(self, grid):
        if isinstance(grid, np.ndarray):
            self.grid = grid
        else:
            self.grid = np.array(grid)
        self.dimensions = len(grid)
        self.flat_idx = self.get_flat_idx()
        self.tensor_idx = self.get_tensor_idx(idx=self.flat_idx) 

    def get_EO_idx(self): 
        E_idx = np.ndarray.flatten(np.ndarray.astype(np.indices(self.grid).sum(axis=0)%2,'bool'))
        O_idx = np.logical_not(E_idx)
        return E_idx, O_idx
       
    def moveforward(self, mu, step=1):
        self.tensor_idx = np.roll(self.tensor_idx, shift=step, axis=mu)
        self.update_flat_idx()

    def movebackward(self, mu, step=1):
        self.tensor_idx = np.roll(self.tensor_idx, shift=-step, axis=mu)
        self.update_flat_idx()

    def get_idx(self, x):

        # idx = x[-1]
        # for d in reversed(range(len(self.grid)-1)):
        #    idx *= self.grid[d]
        #    idx += x[d]
        idx = np.ravel_multi_index(x,tuple(self.grid),order='F')  # FIXME need test ordering
        return idx

    def get_tensor_idx(self, idx: np.ndarray):
        out = np.reshape(idx, self.grid)
        return out

    def get_flat_idx(self):
        return np.array([i for i in range(np.prod(self.grid))])

    def update_flat_idx(self):
        self.flat_idx = np.ndarray.flatten(self.tensor_idx)

    def pick_last_slice(self, tensor, mu):
        return np.take(tensor, indices=-1, axis=mu) #FIXME need test ordering

    def pick_first_slice(self, tensor, mu):
        return np.take(tensor, indices=0, axis=mu) #FIXME need test ordering


class LatticeMPI(LatticeBase):
    def __init__(self, grid, cartesiancomm):
        super().__init__(grid)
        self.cartesiancomm = cartesiancomm

    def moveforward(self, mu, step=1, value=None, dtype='float32'):
        dummy_tensor_idx = np.reshape(
            np.array([i for i in range(np.prod(self.grid))]), self.grid)
        out = value
        snd_idx = np.ndarray.flatten(
            self.pick_last_slice(tensor=dummy_tensor_idx, mu=mu))
        snd_halo = np.ndarray.flatten(out[snd_idx])
        new_shape = list(self.grid)
        new_shape.extend(out.shape[1:])
        out = np.reshape(np.roll(np.reshape(out, new_shape),
                         shift=step, axis=mu), out.shape)
        rcv_halo = self.cartesiancomm.forwardshift(
            mu, snd_buf=snd_halo, dtype=dtype)
        rcv_idx = np.ndarray.flatten(
            self.pick_first_slice(tensor=dummy_tensor_idx, mu=mu))
        rcv_shape = [len(snd_idx)]+list(out.shape[1:])
        out[rcv_idx] = np.reshape(rcv_halo, rcv_shape)
        return out

    def movebackward(self, mu, step=1, value=None, dtype='float32'):
        dummy_tensor_idx = np.reshape(
            np.array([i for i in range(np.prod(self.grid))]), self.grid)
        out = value
        snd_idx = np.ndarray.flatten(
            self.pick_first_slice(tensor=dummy_tensor_idx, mu=mu))
        snd_halo = np.ndarray.flatten(out[snd_idx])
        new_shape = list(self.grid)
        new_shape.extend(out.shape[1:])
        out = np.reshape(np.roll(np.reshape(out, new_shape),
                         shift=-step, axis=mu), out.shape)
        rcv_halo = self.cartesiancomm.backwardshift(
            mu, snd_buf=snd_halo, dtype=dtype)
        rcv_idx = np.ndarray.flatten(
            self.pick_last_slice(tensor=dummy_tensor_idx, mu=mu))
        rcv_shape = [len(snd_idx)]+list(out.shape[1:])
        out[rcv_idx] = np.reshape(rcv_halo, rcv_shape)
        return out

    def ReduceSum(self, value, dtype='float32'):
        out = np.array(0, dtype=dtype)
        snd_buf = np.array(np.sum(value), dtype=dtype)
        self.cartesiancomm.comm.Allreduce([snd_buf, self.cartesiancomm.mpitypes[dtype]], [
                                          out, self.cartesiancomm.mpitypes[dtype]], op=MPI.SUM)
        return out

    def Average(self, value, dtype='float32'):
        return self.ReduceSum(value, dtype=dtype)/(np.prod(self.grid)*np.prod(self.cartesiancomm.mpigrid))


class LatticeReal(LatticeMPI):
    def __init__(self, grid, cartesiancomm):
        super().__init__(grid=grid, cartesiancomm=cartesiancomm)
        self.value = np.zeros(shape=(np.prod(self.grid)), dtype='float32')

    def fill_value(self, n=0):
        if isinstance(n, Real):
            self.value[:] = n.value
        elif isinstance(n, (float, int)):
            self.value[:] = n
        elif isinstance(n, np.ndarray):
            self.value = n

    def peek_EO_lattices(self):
        E_idx, O_idx = self.get_EO_idx()
        E_latt = LatticeReal(grid=np.ndarray.astype(self.grid/2,dtype='i'),cartesiancomm=self.cartesiancomm)
        O_latt = LatticeReal(grid=np.ndarray.astype(self.grid/2,dtype='i'),cartesiancomm=self.cartesiancomm)
        E_latt.value = self.value[E_idx]
        O_latt.value = self.value[O_idx]
        return E_latt, O_latt

    def poke_EO_lattices(self,E_lattice=None,O_lattice=None): 
        E_idx, O_idx = self.get_EO_idx()
        if E_lattice!=None:
            self.value[E_idx] = E_lattice.value 
        if O_lattice!=None:
            self.value[O_idx] = O_lattice.value

    def __getitem__(self, idx: int):
        return self.value[idx, :]

    def moveforward(self, mu, step=1):
        self.value = super().moveforward(mu=mu, step=1, value=self.value)

    def movebackward(self, mu, step=1):
        self.value = super().movebackward(mu=mu, step=1, value=self.value)

    def reducesum(self):
        return super().ReduceSum(value=self.value, dtype='float32')

    def average(self):
        return super().Average(value=self.value, dtype='float32')

    def __add__(self, rhs):
        out = LatticeReal(grid=self.grid, cartesiancomm=self.cartesiancomm)
        if isinstance(rhs, LatticeReal):
            assert(self.value.shape == rhs.value.shape)
            out.value = self.value + rhs.value
        elif isinstance(rhs, Real):
            out.value = self.value + rhs.value
        elif isinstance(rhs, (int, float)):
            out.value = self.value + rhs
        return out

    def __sub__(self, rhs):
        out = LatticeReal(grid=self.grid, cartesiancomm=self.cartesiancomm)
        if isinstance(rhs, LatticeReal):
            assert(self.value.shape == rhs.value.shape)
            out.value = self.value - rhs.value
        elif isinstance(rhs, Real):
            out.value = self.value - rhs.value
        elif isinstance(rhs, (int, float)):
            out.value = self.value - rhs
        return out

    def __mul__(self, rhs):
        out = LatticeReal(grid=self.grid, cartesiancomm=self.cartesiancomm)
        if isinstance(rhs, LatticeReal):
            assert(self.value.shape == rhs.value.shape)
            out.value = self.value * rhs.value
        elif isinstance(rhs, Real):
            out.value = self.value * rhs.value
        elif isinstance(rhs, (int, float)):
            out.value = self.value * rhs
        return out


class LatticeComplex(LatticeMPI):
    def __init__(self, grid, cartesiancomm):
        super().__init__(grid=grid, cartesiancomm=cartesiancomm)
        self.value = np.zeros(shape=(np.prod(self.grid)), dtype='complex64')

    def fill_value(self, n=0):
        if isinstance(n, (Real, Complex)):
            self.value[:] = n.value
        elif isinstance(n, (complex, float, int)):
            self.value[:] = n
        elif isinstance(n, np.ndarray):
            self.value = n

    def __getitem__(self, idx: int):
        return self.value[idx, :]

    def moveforward(self, mu, step=1):
        self.value = super().moveforward(mu=mu, step=1, value=self.value, dtype='complex64')

    def movebackward(self, mu, step=1):
        self.value = super().movebackward(
            mu=mu, step=1, value=self.value, dtype='complex64')

    def reducesum(self):
        return super().ReduceSum(value=self.value, dtype='complex64')

    def average(self):
        return super().Average(value=self.value, dtype='complex64')

    def __add__(self, rhs):
        out = LatticeComplex(grid=self.grid, cartesiancomm=self.cartesiancomm)
        if isinstance(rhs, (LatticeReal, LatticeComplex)):
            assert(self.value.shape == rhs.value.shape)
            out.value = self.value + rhs.value
        elif isinstance(rhs, (Complex, Real)):
            out.value = self.value + rhs.value
        elif isinstance(rhs, (int, float, complex)):
            out.value = self.value + rhs
        return out

    def __sub__(self, rhs):
        out = LatticeComplex(grid=self.grid, cartesiancomm=self.cartesiancomm)
        if isinstance(rhs, (LatticeComplex, LatticeReal)):
            assert(self.value.shape == rhs.value.shape)
            out.value = self.value - rhs.value
        elif isinstance(rhs, (Complex, Real)):
            out.value = self.value - rhs.value
        elif isinstance(rhs, (int, float, complex)):
            out.value = self.value - rhs
        return out

    def __mul__(self, rhs):
        out = LatticeComplex(grid=self.grid, cartesiancomm=self.cartesiancomm)
        if isinstance(rhs, (LatticeComplex, LatticeReal)):
            assert(self.value.shape == rhs.value.shape)
            out.value = self.value * rhs.value
        elif isinstance(rhs, (Complex, Real)):
            out.value = self.value * rhs.value
        elif isinstance(rhs, (int, float, complex)):
            out.value = self.value * rhs
        return out


class LatticeRealMatrix(LatticeMPI):
    def __init__(self, grid, cartesiancomm, N: int):
        super().__init__(grid=grid, cartesiancomm=cartesiancomm)
        self.N = N
        self.value = np.zeros(shape=(np.prod(self.grid), N, N), dtype='float32')

    def fill_value(self, n: RealMatrix):
        if isinstance(n, RealMatrix):
            self.value[:] = n.value
        if isinstance(n, np.ndarray):
            self.value = n

    def moveforward(self, mu, step=1):
        self.value = super().moveforward(mu=mu, step=1, value=self.value, dtype='float32')

    def movebackward(self, mu, step=1):
        self.value = super().movebackward(mu=mu, step=1, value=self.value, dtype='float32')

    def __getitem__(self, idx: int):
        return self.value[idx, :]

    def reducesum(self):
        return super().ReduceSum(value=self.value, dtype='float32')

    def average(self):
        return super().Average(value=self.value, dtype='float32')

    def transpose(self):
        out = LatticeRealMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, N=self.N)
        for i in range(np.prod(self.grid)):
            out.value[i] = np.transpose(self.value[i, :, :])
        return out

    def trace(self):
        out = LatticeReal(grid=self.grid, cartesiancomm=self.cartesiancomm)
        for i in range(np.prod(self.grid)):
            out.value[i] = np.trace(self.value[i, :, :])
        return out

    def det(self):
        out = LatticeReal(grid=self.grid, cartesiancomm=self.cartesiancomm)
        for i in range(np.prod(self.grid)):
            out.value[i] = np.linalg.det(self.value[i, :, :])
        return out

    def inv(self):
        out = LatticeRealMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, N=self.N)
        for i in range(np.prod(self.grid)):
            out.value[i] = np.linalg.inv(self.value[i, :, :])
        return out

    def __add__(self, rhs):
        out = LatticeRealMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, N=self.N)
        if isinstance(rhs, LatticeRealMatrix):
            assert(self.value.shape == rhs.value.shape)
            out.value = self.value + rhs.value
        elif isinstance(rhs, RealMatrix):
            assert(self.value[0].shape == rhs.value.shape)
            out.value = self.value + rhs.value
        return out

    def __sub__(self, rhs):
        out = LatticeRealMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, N=self.N)
        if isinstance(rhs, LatticeRealMatrix):
            assert(self.value.shape == rhs.value.shape)
            out.value = self.value - rhs.value
        elif isinstance(rhs, RealMatrix):
            assert(self.value[0].shape == rhs.value.shape)
            out.value = self.value - rhs.value
        return out

    def __mul__(self, rhs):
        out = LatticeRealMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, N=self.N)
        if isinstance(rhs, LatticeRealMatrix):
            assert(self.value.shape == rhs.value.shape)
            for i in range(np.prod(self.grid)):
                out.value[i] = np.dot(self.value[i], rhs.value[i])
        elif isinstance(rhs, RealMatrix):
            assert(self.value[0].shape == rhs.value.shape)
            for i in range(np.prod(self.grid)):
                out.value[i] = np.dot(self.value[i], rhs.value)
        elif isinstance(rhs, Real):
            out.value = self.value * rhs.value
        return out


class LatticeComplexMatrix(LatticeMPI):
    def __init__(self, grid, cartesiancomm, N: int):
        super().__init__(grid=grid, cartesiancomm=cartesiancomm)
        self.N = N
        self.value = np.zeros(shape=(np.prod(self.grid), N, N), dtype='complex64')

    def fill_value(self, n: ComplexMatrix):
        if isinstance(n, ComplexMatrix):
            self.value[:] = n.value
        elif isinstance(n, np.ndarray):
            self.value = n

    def moveforward(self, mu, step=1):
        self.value = super().moveforward(mu=mu, step=1, value=self.value, dtype='complex64')

    def movebackward(self, mu, step=1):
        self.value = super().movebackward(
            mu=mu, step=1, value=self.value, dtype='complex64')

    def __getitem__(self, idx: int):
        return self.value[idx, :]

    def reducesum(self):
        return super().ReduceSum(value=self.value, dtype='complex64')

    def average(self):
        return super().Average(value=self.value, dtype='complex64')

    def transpose(self):
        out = LatticeComplexMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, N=self.N)
        for i in range(np.prod(self.grid)):
            out.value[i] = np.transpose(self.value[i, :, :])
        return out

    def trace(self):
        out = LatticeComplex(grid=self.grid, cartesiancomm=self.cartesiancomm)
        for i in range(np.prod(self.grid)):
            out.value[i] = np.trace(self.value[i, :, :])
        return out

    def det(self):
        out = LatticeComplex(grid=self.grid, cartesiancomm=self.cartesiancomm)
        for i in range(np.prod(self.grid)):
            out.value[i] = np.linalg.det(self.value[i, :, :])
        return out

    def inv(self):
        out = LatticeComplexMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, N=self.N)
        for i in range(np.prod(self.grid)):
            out.value[i] = np.linalg.inv(self.value[i, :, :])
        return out

    def conj(self):
        out = LatticeComplexMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, N=self.N)
        for i in range(np.prod(self.grid)):
            out.value[i] = np.conj(self.value[i, :, :])
        return out

    def adj(self):
        tmp = LatticeComplexMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, N=self.N)
        tmp = self.conj()
        return tmp.transpose()

    def re(self):
        out = LatticeRealMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, N=self.N)
        out.value = np.real(self.value)
        return out

    def im(self):
        out = LatticeRealMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, N=self.N)
        out.value = np.imag(self.value)
        return out

    def __add__(self, rhs):
        out = LatticeComplexMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, N=self.N)
        if isinstance(rhs, (LatticeComplexMatrix, LatticeRealMatrix)):
            assert(self.value.shape == rhs.value.shape)
            out.value = self.value + rhs.value
        elif isinstance(rhs, (ComplexMatrix, RealMatrix)):
            assert(self.value[0].shape == rhs.value.shape)
            out.value = self.value + rhs.value
        return out

    def __sub__(self, rhs):
        out = LatticeComplexMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, N=self.N)
        if isinstance(rhs, (LatticeRealMatrix, LatticeComplexMatrix)):
            assert(self.value.shape == rhs.value.shape)
            out.value = self.value - rhs.value
        elif isinstance(rhs, (RealMatrix, ComplexMatrix)):
            assert(self.value[0].shape == rhs.value.shape)
            out.value = self.value - rhs.value
        return out

    def __mul__(self, rhs):
        out = LatticeComplexMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, N=self.N)
        if isinstance(rhs, (LatticeRealMatrix, LatticeComplexMatrix)):
            assert(self.value.shape == rhs.value.shape)
            for i in range(np.prod(self.grid)):
                out.value[i] = np.dot(self.value[i], rhs.value[i])
        elif isinstance(rhs, (RealMatrix, ComplexMatrix)):
            assert(self.value[0].shape == rhs.value.shape)
            for i in range(np.prod(self.grid)):
                out.value[i] = np.dot(self.value[i], rhs.value)
        elif isinstance(rhs, (Real, Complex)):
            out.value = self.value * rhs.value
        elif isinstance(rhs, (float, int, complex)):
            out.value = self.value * rhs
        return out


class LatticeVectorReal(LatticeMPI):
    def __init__(self, grid, cartesiancomm, Nd: int):
        super().__init__(grid=grid, cartesiancomm=cartesiancomm)
        self.Nd = Nd
        self.value = np.zeros(shape=(np.prod(self.grid), Nd), dtype='float32')

    def fill_value(self, n: VectorReal):
        if isinstance(n, VectorReal):
            self.value[:] = n.value
        elif isinstance(n, np.ndarray):
            self.value[:] = n

    def moveforward(self, mu, step=1):
        self.value = super().moveforward(mu=mu, step=1, value=self.value, dtype='float32')

    def movebackward(self, mu, step=1):
        self.value = super().movebackward(mu=mu, step=1, value=self.value, dtype='float32')

    def reducesum(self):
        return super().ReduceSum(value=self.value, dtype='float32')

    def average(self):
        return super().Average(value=self.value, dtype='float32')

    def __getitem__(self, idx: int):
        return self.value[idx, :]

    def peek_index(self,mu): 
        out = LatticeReal(grid=self.grid,cartesiancomm=self.cartesiancomm)
        out.value = self.value[:,mu]
        return out
    
    def poke_index(self,mu,obj: LatticeReal): 
        self.value[:,mu]=obj.value

    def transpose(self):
        out = LatticeVectorReal(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd)
        for i in range(np.prod(self.grid)):
            out.value[i] = np.transpose(self.value[i, :])
        return out

    def __add__(self, rhs):
        out = LatticeVectorReal(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd)
        if isinstance(rhs, LatticeVectorReal):
            assert(self.value.shape == rhs.value.shape)
            out.value = self.value + rhs.value
        return out

    def __sub__(self, rhs):
        out = LatticeVectorReal(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd)
        if isinstance(rhs, LatticeVectorReal):
            assert(self.value.shape == rhs.value.shape)
            out.value = self.value - rhs.value
        return out

    def __mul__(self, rhs):
        if isinstance(rhs, LatticeVectorReal):
            out = LatticeReal(grid=self.grid, cartesiancomm=self.cartesiancomm)
            assert(self.value.shape == rhs.value.shape)
            for i in range(out.length):
                out.value[i] = np.dot(self.value[i], rhs.value[i])
        elif isinstance(rhs, LatticeReal):
            out = LatticeVectorReal(
                grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd)
            for i in range(out.length):
                out.value[i] = self.value[i]*rhs.value[i]
        elif isinstance(rhs, Real):
            out = LatticeVectorReal(
                grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd)
            out.value = self.value * rhs.value
        return out


class LatticeVectorComplex(LatticeMPI):
    def __init__(self, grid, cartesiancomm, Nd: int):
        super().__init__(grid=grid, cartesiancomm=cartesiancomm)
        self.Nd = Nd
        self.value = np.zeros(shape=(np.prod(self.grid), Nd), dtype='complex64')

    def fill_value(self, n: VectorComplex):
        if isinstance(n, VectorComplex):
            self.value[:] = n.value[:]
        elif isinstance(n, np.ndarray):
            self.value[:] = n

    def moveforward(self, mu, step=1):
        self.value = super().moveforward(mu=mu, step=1, value=self.value, dtype='complex64')

    def movebackward(self, mu, step=1):
        self.value = super().movebackward(
            mu=mu, step=1, value=self.value, dtype='complex64')

    def reducesum(self):
        return super().ReduceSum(value=self.value, dtype='complex64')

    def average(self):
        return super().Average(value=self.value, dtype='complex64')

    def __getitem__(self, idx: int):
        return self.value[idx, :]

    def transpose(self):
        out = LatticeVectorComplex(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd)
        for i in range(out.lattice):
            out.value[i] = np.transpose(self.value[i, :])
        return out

    def __add__(self, rhs):
        out = LatticeVectorComplex(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd)
        if isinstance(rhs, (LatticeVectorReal, LatticeVectorComplex)):
            assert(self.value.shape == rhs.value.shape)
            out.value = self.value + rhs.value
        return out

    def __sub__(self, rhs):
        out = LatticeVectorComplex(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd)
        if isinstance(rhs, (LatticeVectorReal, LatticeVectorComplex)):
            assert(self.value.shape == rhs.value.shape)
            out.value = self.value - rhs.value
        return out

    def __mul__(self, rhs):
        if isinstance(rhs, LatticeVectorReal):
            out = LatticeReal(lattice=self.lattice)
            assert(self.value.shape == rhs.value.shape)
            for i in range(out.length):
                out.value[i] = np.dot(self.value[i], rhs.value[i])
        elif isinstance(rhs, (LatticeReal, LatticeComplex)):
            out = LatticeVectorComplex(
                grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd)
            for i in range(out.length):
                out.value[i] = self.value[i]*rhs.value[i]
        elif isinstance(rhs, (Complex, Real)):
            out = LatticeVectorComplex(
                grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd)
            out.value = self.value * rhs.value
        return out

    def conj(self):
        out = LatticeVectorComplex(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd)
        for i in range(np.prod(self.grid)):
            out.value = np.conj(self.value)
        return out

    def re(self):
        out = LatticeVectorReal(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd)
        out.value = np.real(self.value)
        return out

    def im(self):
        out = LatticeVectorReal(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd)
        out.value = np.imag(self.value)
        return out


class LatticeVectorRealMatrix(LatticeMPI):
    def __init__(self, grid, cartesiancomm, Nd: int, N: int):
        super().__init__(grid=grid, cartesiancomm=cartesiancomm)
        self.Nd = Nd
        self.N = N
        self.value = np.zeros(shape=(np.prod(self.grid), Nd, N, N), dtype='float32')

    def fill_value(self, n: VectorRealMatrix):
        if isinstance(n, VectorRealMatrix):
            self.value[:] = n.value
        elif isinstance(n, np.ndarray):
            self.value = n

    def moveforward(self, mu, step=1):
        self.value = super().moveforward(mu=mu, step=1, value=self.value, dtype='float32')

    def movebackward(self, mu, step=1):
        self.value = super().movebackward(mu=mu, step=1, value=self.value, dtype='float32')

    def reducesum(self):
        return super().ReduceSum(value=self.value, dtype='float32')

    def average(self):
        return super().Average(value=self.value, dtype='float32')

    def __getitem__(self, idx: int):
        return self.value[idx, :]

    def transpose(self):
        out = LatticeVectorRealMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd, N=self.N)
        for i in range(np.prod(self.grid)):
            for n in range(self.Nd):
                out.value[i, n] = np.transpose(self.value[i, n, :, :])
        return out

    def trace(self):
        out = LatticeVectorReal(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd)
        for i in range(np.prod(self.grid)):
            for n in range(self.Nd):
                out.value[i, n] = np.trace(self.value[i, n, :, :])
        return out

    def det(self):
        out = LatticeVectorReal(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd)
        for i in range(np.prod(self.grid)):
            for n in range(self.Nd):
                out.value[i, n] = np.linalg.det(self.value[i, n, :, :])
        return out

    def inv(self):
        out = LatticeVectorRealMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd, N=self.N)
        for i in range(np.prod(self.grid)):
            for n in range(self.Nd):
                out.value[i, n] = np.linalg.inv(self.value[i, n, :, :])
        return out

    def __add__(self, rhs):
        out = LatticeVectorRealMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd, N=self.N)
        if isinstance(rhs, LatticeVectorRealMatrix):
            assert(self.value.shape == rhs.value.shape)
            out.value = self.value + rhs.value
        elif isinstance(rhs, LatticeRealMatrix):
            assert(self.value[0, 0].shape == rhs.value[0].shape)
            for i in range(np.prod(self.grid)):
                for n in range(self.Nd):
                    out.value[i, n] = self.value[i, n] + rhs.value[i]
        elif isinstance(rhs, (Real, RealMatrix)):
            assert(self.value[0, 0].shape == rhs.value[0].shape)
            for i in range(np.prod(self.grid)):
                for n in range(self.Nd):
                    out.value[i, n] = self.value[i, n] + rhs.value
        elif isinstance(rhs, (float, int)):
            for i in range(np.prod(self.grid)):
                for n in range(self.Nd):
                    out.value[i, n] = self.value[i, n] + rhs
        return out

    def __sub__(self, rhs):
        out = LatticeVectorRealMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd, N=self.N)
        if isinstance(rhs, LatticeVectorRealMatrix):
            assert(self.value.shape == rhs.value.shape)
            out.value = self.value - rhs.value
        elif isinstance(rhs, LatticeRealMatrix):
            assert(self.value[0, 0].shape == rhs.value[0].shape)
            for i in range(np.prod(self.grid)):
                for n in range(self.Nd):
                    out.value[i, n] = self.value[i, n] - rhs.value[i]
        elif isinstance(rhs, (Real, RealMatrix)):
            assert(self.value[0, 0].shape == rhs.value[0].shape)
            for i in range(np.prod(self.grid)):
                for n in range(self.Nd):
                    out.value[i, n] = self.value[i, n] - rhs.value
        elif isinstance(rhs, (float, int)):
            for i in range(np.prod(self.grid)):
                for n in range(self.Nd):
                    out.value[i, n] = self.value[i, n] - rhs
        return out

    def __mul__(self, rhs):
        out = LatticeVectorRealMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd, N=self.N)
        if isinstance(rhs, LatticeVectorRealMatrix):
            assert(self.value.shape == rhs.value.shape)
            for i in range(np.prod(self.grid)):
                for n in range(self.Nd):
                    out.value[i, n] = np.dot(self.value[i, n], rhs.value[i, n])
        elif isinstance(rhs, LatticeRealMatrix):
            for i in range(np.prod(self.grid)):
                for n in range(self.Nd):
                    out.value[i, n] = np.dot(self.value[i, n], rhs.value[i])
        elif isinstance(rhs, RealMatrix):
            for i in range(np.prod(self.grid)):
                for n in range(self.Nd):
                    out.value[i, n] = np.dot(self.value[i, n], rhs.value)
        elif isinstance(rhs, Real):
            for i in range(np.prod(self.grid)):
                for n in range(self.Nd):
                    out.value[i, n] = self.value[i, n] * rhs.value
        elif isinstance(rhs, (float, int)):
            assert(self.value.shape == rhs.value.shape)
            for i in range(np.prod(self.grid)):
                for n in range(self.Nd):
                    out.value[i, n] = self.value[i, n] * rhs
        return out


class LatticeVectorComplexMatrix(LatticeMPI):
    def __init__(self, grid, cartesiancomm, Nd: int, N: int):
        super().__init__(grid=grid, cartesiancomm=cartesiancomm)
        self.Nd = Nd
        self.N = N
        self.value = np.zeros(shape=(np.prod(self.grid), Nd, N, N), dtype='complex64')

    def fill_value(self, n: VectorComplexMatrix):
        if isinstance(n, VectorComplexMatrix):
            self.value[:] = n.value
        elif isinstance(n, np.ndarray):
            self.value = n

    def moveforward(self, mu, step=1):
        self.value = super().moveforward(mu=mu, step=1, value=self.value, dtype='complex64')

    def movebackward(self, mu, step=1):
        self.value = super().movebackward(
            mu=mu, step=1, value=self.value, dtype='complex64')

    def reducesum(self):
        return super().ReduceSum(value=self.value, dtype='complex64')

    def average(self):
        return super().Average(value=self.value, dtype='complex64')

    def __getitem__(self, idx: int):
        return self.value[idx, :]

    def transpose(self):
        out = LatticeVectorComplexMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd, N=self.N)
        for i in range(np.prod(self.grid)):
            for n in range(self.Nd):
                out.value[i, n] = np.transpose(self.value[i, n, :, :])
        return out

    def trace(self):
        out = LatticeVectorComplex(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd)
        for i in range(np.prod(self.grid)):
            for n in range(self.Nd):
                out.value[i, n] = np.trace(self.value[i, n, :, :])
        return out

    def det(self):
        out = LatticeVectorComplex(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd)
        for i in range(np.prod(self.grid)):
            for n in range(self.Nd):
                out.value[i, n] = np.linalg.det(self.value[i, n, :, :])
        return out

    def inv(self):
        out = LatticeVectorComplexMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd, N=self.N)
        for i in range(np.prod(self.grid)):
            for n in range(self.Nd):
                out.value[i, n] = np.linalg.inv(self.value[i, n, :, :])
        return out

    def conj(self):
        out = LatticeVectorComplexMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd, N=self.N)
        for i in range(np.prod(self.grid)):
            for n in range(self.Nd):
                out.value[i, n] = np.conj(self.value[i, n, :, :])
        return out

    def adj(self):
        tmp = LatticeVectorComplexMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd, N=self.N)
        tmp = self.conj()
        return tmp.transpose()

    def re(self):
        out = LatticeVectorRealMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd, N=self.N)
        out.value = np.real(self.value)
        return out

    def im(self):
        out = LatticeVectorRealMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd, N=self.N)
        out.value = np.imag(self.value)
        return out

    def __add__(self, rhs):
        out = LatticeVectorComplexMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd, N=self.N)
        if isinstance(rhs, (LatticeVectorComplexMatrix, LatticeVectorRealMatrix)):
            assert(self.value.shape == rhs.value.shape)
            out.value = self.value + rhs.value
        elif isinstance(rhs, (LatticeComplexMatrix, LatticeRealMatrix)):
            assert(self.value[0, 0].shape == rhs.value[0].shape)
            for i in range(np.prod(self.grid)):
                for n in range(self.Nd):
                    out.value[i, n] = self.value[i, n] + rhs.value[i]
        elif isinstance(rhs, (Real, RealMatrix, Complex, ComplexMatrix)):
            assert(self.value[0, 0].shape == rhs.value[0].shape)
            for i in range(np.prod(self.grid)):
                for n in range(self.Nd):
                    out.value[i, n] = self.value[i, n] + rhs.value
        elif isinstance(rhs, (float, int, complex)):
            for i in range(np.prod(self.grid)):
                for n in range(self.Nd):
                    out.value[i, n] = self.value[i, n] + rhs
        return out

    def __sub__(self, rhs):
        out = LatticeVectorComplexMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd, N=self.N)
        if isinstance(rhs, (LatticeVectorComplexMatrix, LatticeVectorRealMatrix)):
            assert(self.value.shape == rhs.value.shape)
            out.value = self.value - rhs.value
        elif isinstance(rhs, (LatticeComplexMatrix, LatticeRealMatrix)):
            assert(self.value[0, 0].shape == rhs.value[0].shape)
            for i in range(np.prod(self.grid)):
                for n in range(self.Nd):
                    out.value[i, n] = self.value[i, n] - rhs.value[i]
        elif isinstance(rhs, (Real, RealMatrix, Complex, ComplexMatrix)):
            assert(self.value[0, 0].shape == rhs.value[0].shape)
            for i in range(np.prod(self.grid)):
                for n in range(self.Nd):
                    out.value[i, n] = self.value[i, n] - rhs.value
        elif isinstance(rhs, (float, int, complex)):
            for i in range(np.prod(self.grid)):
                for n in range(self.Nd):
                    out.value[i, n] = self.value[i, n] - rhs
        return out

    def __mul__(self, rhs):
        out = LatticeVectorComplexMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd, N=self.N)
        if isinstance(rhs, (LatticeVectorComplexMatrix, LatticeVectorRealMatrix)):
            assert(self.value.shape == rhs.value.shape)
            for i in range(np.prod(self.grid)):
                for n in range(self.Nd):
                    out.value[i, n] = np.dot(self.value[i, n], rhs.value[i, n])
        elif isinstance(rhs, (LatticeRealMatrix, LatticeComplexMatrix)):
            for i in range(np.prod(self.grid)):
                for n in range(self.Nd):
                    out.value[i, n] = np.dot(self.value[i, n], rhs.value[i])
        elif isinstance(rhs, (ComplexMatrix, RealMatrix)):
            for i in range(np.prod(self.grid)):
                for n in range(self.Nd):
                    out.value[i, n] = np.dot(self.value[i, n], rhs.value)
        elif isinstance(rhs, (Complex, Real)):
            for i in range(np.prod(self.grid)):
                for n in range(self.Nd):
                    out.value[i, n] = self.value[i, n] * rhs.value
        elif isinstance(rhs, (float, int, complex)):
            assert(self.value.shape == rhs.value.shape)
            for i in range(np.prod(self.grid)):
                for n in range(self.Nd):
                    out.value[i, n] = self.value[i, n] * rhs
        return out
