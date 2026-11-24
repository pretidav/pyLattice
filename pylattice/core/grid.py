from pylattice.core.parallel import *
from pylattice.core.tensors import *
import copy

import numpy as np

# The precision of every lattice (its storage, shifts, reductions and files): double. A lattice keeps the dtype of its
# value, so assigning single-precision data makes that lattice single; nothing is converted on the way.
REAL = np.float64
COMPLEX = np.complex128


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
        # the parity is taken from local coordinates: this equals the global parity only if
        # every local extent is even, so that is required for the checkerboard to be consistent
        if (self.grid % 2).any():
            raise ValueError("even/odd decomposition needs every local grid dimension to be even, got {}".format(self.grid.tolist()))
        # even site: sum of the coordinates is even, so the origin (0,...,0) is even
        O_idx = np.ndarray.flatten(np.ndarray.astype(np.indices(self.grid).sum(axis=0)%2,'bool'))
        E_idx = np.logical_not(O_idx)
        return E_idx, O_idx
       
    # moveforward: the value at x becomes the one that was at x + step*mu
    # movebackward: the value at x becomes the one that was at x - step*mu
    def moveforward(self, mu, step=1):
        self.tensor_idx = np.roll(self.tensor_idx, shift=-step, axis=mu)
        self.update_flat_idx()

    def movebackward(self, mu, step=1):
        self.tensor_idx = np.roll(self.tensor_idx, shift=step, axis=mu)
        self.update_flat_idx()

    def get_idx(self, x):

        # idx = x[-1]
        # for d in reversed(range(len(self.grid)-1)):
        #    idx *= self.grid[d]
        #    idx += x[d]
        idx = np.ravel_multi_index(x,tuple(self.grid),order='C')  
        return idx

    def get_tensor_idx(self, idx: np.ndarray):
        out = np.reshape(idx, self.grid)
        return out

    def get_flat_idx(self):
        return np.arange(np.prod(self.grid))

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

    def moveforward(self, mu, step=1, value=None):
        if step != 1:
            raise ValueError("only step=1 is supported: the halo exchange moves a single slice")
        dummy_tensor_idx = np.reshape(
            np.arange(np.prod(self.grid)), self.grid)
        # value at x becomes the value at x + mu: our first slice goes to the -mu neighbour,
        # the +mu neighbour's first slice becomes our last slice
        out = value
        snd_idx = np.ndarray.flatten(
            self.pick_first_slice(tensor=dummy_tensor_idx, mu=mu))
        snd_halo = np.ndarray.flatten(out[snd_idx])
        new_shape = list(self.grid)
        new_shape.extend(out.shape[1:])
        out = np.reshape(np.roll(np.reshape(out, new_shape),
                         shift=-step, axis=mu), out.shape)
        rcv_halo = self.cartesiancomm.forwardshift(
            mu, snd_buf=snd_halo)
        rcv_idx = np.ndarray.flatten(
            self.pick_last_slice(tensor=dummy_tensor_idx, mu=mu))
        rcv_shape = [len(snd_idx)]+list(out.shape[1:])
        out[rcv_idx] = np.reshape(rcv_halo, rcv_shape)
        return out

    def movebackward(self, mu, step=1, value=None):
        if step != 1:
            raise ValueError("only step=1 is supported: the halo exchange moves a single slice")
        dummy_tensor_idx = np.reshape(
            np.arange(np.prod(self.grid)), self.grid)
        # value at x becomes the value at x - mu: our last slice goes to the +mu neighbour,
        # the -mu neighbour's last slice becomes our first slice
        out = value
        snd_idx = np.ndarray.flatten(
            self.pick_last_slice(tensor=dummy_tensor_idx, mu=mu))
        snd_halo = np.ndarray.flatten(out[snd_idx])
        new_shape = list(self.grid)
        new_shape.extend(out.shape[1:])
        out = np.reshape(np.roll(np.reshape(out, new_shape),
                         shift=step, axis=mu), out.shape)
        rcv_halo = self.cartesiancomm.backwardshift(
            mu, snd_buf=snd_halo)
        rcv_idx = np.ndarray.flatten(
            self.pick_first_slice(tensor=dummy_tensor_idx, mu=mu))
        rcv_shape = [len(snd_idx)]+list(out.shape[1:])
        out[rcv_idx] = np.reshape(rcv_halo, rcv_shape)
        return out

    def copy(self):
        """an independent lattice with the same values (the communicator is shared)"""
        out = copy.copy(self)
        out.value = self.value.copy()
        return out

    def shift_forward(self, mu, step=1):
        """a new lattice whose value at x is the one of this lattice at x + mu (this one is unchanged)"""
        out = self.copy()
        out.moveforward(mu, step)
        return out

    def shift_backward(self, mu, step=1):
        """a new lattice whose value at x is the one of this lattice at x - mu (this one is unchanged)"""
        out = self.copy()
        out.movebackward(mu, step)
        return out

    def ReduceSum(self, value):
        """the sum of value over all the sites of all the ranks, in the precision of the data"""
        dtype = value.dtype if value.dtype.name in self.cartesiancomm.mpitypes else (
            COMPLEX if np.iscomplexobj(value) else REAL)
        out = np.array(0, dtype=dtype)
        snd_buf = np.array(np.sum(value), dtype=dtype)
        mpitype = self.cartesiancomm.mpitypes[np.dtype(dtype).name]
        self.cartesiancomm.comm.Allreduce([snd_buf, mpitype], [out, mpitype], op=MPI.SUM)
        return out

    def Average(self, value):
        return self.ReduceSum(value)/(np.prod(self.grid)*np.prod(self.cartesiancomm.mpigrid))


class LatticeReal(LatticeMPI):
    def __init__(self, grid, cartesiancomm):
        super().__init__(grid=grid, cartesiancomm=cartesiancomm)
        self.value = np.zeros(shape=(np.prod(self.grid)), dtype=REAL)

    # Even/odd half-lattices (see peek_EO_lattices): eo_parity is 0 (even sites) or 1 (odd sites),
    # eo_parent_grid is the grid of the full lattice. Both are None for an ordinary lattice.
    eo_parity = None
    eo_parent_grid = None

    def _new(self):
        # empty lattice of the same shape, keeping the even/odd tags
        out = LatticeReal(grid=self.grid, cartesiancomm=self.cartesiancomm)
        out.eo_parity, out.eo_parent_grid = self.eo_parity, self.eo_parent_grid
        return out

    def _check_eo(self, rhs):
        if rhs.eo_parity != self.eo_parity:
            raise ValueError("cannot combine lattices living on different sublattices "
                             "(even/odd parity {} and {})".format(self.eo_parity, rhs.eo_parity))

    def fill_value(self, n=0):
        if isinstance(n, Real):
            self.value[:] = n.value
        elif isinstance(n, (float, int)):
            self.value[:] = n
        elif isinstance(n, np.ndarray):
            self.value = n

    def peek_EO_lattices(self):
        if self.eo_parity is not None:
            raise ValueError("this is already an even/odd half-lattice")
        E_idx, O_idx = self.get_EO_idx()
        # Each half holds prod(grid)/2 sites, stored compactly in C order. Every line along the
        # last axis has exactly grid[-1]/2 sites of each parity, so the storage is a regular array
        # of shape grid with the last dimension halved.
        # A half-lattice remembers its parity and its parent grid: moveforward/movebackward
        # (always one step) hop to the other sublattice, E -> O and O -> E (see _eo_shift).
        half_grid = self.grid.copy()
        half_grid[-1] //= 2
        E_latt = LatticeReal(grid=half_grid,cartesiancomm=self.cartesiancomm)
        O_latt = LatticeReal(grid=half_grid,cartesiancomm=self.cartesiancomm)
        E_latt.value = self.value[E_idx]
        O_latt.value = self.value[O_idx]
        E_latt.eo_parity, O_latt.eo_parity = 0, 1
        E_latt.eo_parent_grid = O_latt.eo_parent_grid = self.grid.copy()
        return E_latt, O_latt

    def poke_EO_lattices(self,E_lattice=None,O_lattice=None): 
        E_idx, O_idx = self.get_EO_idx()
        for lattice, parity in ((E_lattice, 0), (O_lattice, 1)):
            if lattice is not None and lattice.eo_parity not in (None, parity):
                raise ValueError("a lattice living on the {} sites cannot be poked into the {} sites".format(
                    ('even', 'odd')[lattice.eo_parity], ('even', 'odd')[parity]))
        if E_lattice is not None:
            self.value[E_idx] = E_lattice.value 
        if O_lattice is not None:
            self.value[O_idx] = O_lattice.value

    def _eo_shift(self, mu, step, forward):
        # Values of this half-lattice (on sublattice eo_parity) as seen from the other sublattice:
        # put them on the parent lattice, shift it with the (MPI-aware) full-lattice move and read
        # the result on the opposite parity. forward: value at x + mu, backward: value at x - mu.
        parent = LatticeReal(grid=self.eo_parent_grid, cartesiancomm=self.cartesiancomm)
        E_idx, O_idx = parent.get_EO_idx()
        src, dst = (E_idx, O_idx) if self.eo_parity == 0 else (O_idx, E_idx)
        full = np.zeros(len(E_idx), dtype=self.value.dtype)
        full[src] = self.value
        move = LatticeMPI.moveforward if forward else LatticeMPI.movebackward
        return move(parent, mu=mu, step=step, value=full)[dst]

    def __getitem__(self, idx: int):
        return self.value[idx, :]

    def moveforward(self, mu, step=1):
        if self.eo_parity is not None:
            self.value = self._eo_shift(mu, step, forward=True)
            self.eo_parity = 1 - self.eo_parity         # one step always lands on the other sublattice
        else:
            self.value = super().moveforward(mu=mu, step=step, value=self.value)

    def movebackward(self, mu, step=1):
        if self.eo_parity is not None:
            self.value = self._eo_shift(mu, step, forward=False)
            self.eo_parity = 1 - self.eo_parity
        else:
            self.value = super().movebackward(mu=mu, step=step, value=self.value)

    def reducesum(self):
        return super().ReduceSum(value=self.value)

    def average(self):
        return super().Average(value=self.value)

    def __add__(self, rhs):
        out = self._new()
        if isinstance(rhs, LatticeReal):
            self._check_eo(rhs)
            assert(self.value.shape == rhs.value.shape)
            out.value = self.value + rhs.value
        elif isinstance(rhs, Real):
            out.value = self.value + rhs.value
        elif isinstance(rhs, (int, float)):
            out.value = self.value + rhs
        return out

    def __sub__(self, rhs):
        out = self._new()
        if isinstance(rhs, LatticeReal):
            self._check_eo(rhs)
            assert(self.value.shape == rhs.value.shape)
            out.value = self.value - rhs.value
        elif isinstance(rhs, Real):
            out.value = self.value - rhs.value
        elif isinstance(rhs, (int, float)):
            out.value = self.value - rhs
        return out

    def __mul__(self, rhs):
        out = self._new()
        if isinstance(rhs, LatticeReal):
            self._check_eo(rhs)
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
        self.value = np.zeros(shape=(np.prod(self.grid)), dtype=COMPLEX)

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
        self.value = super().moveforward(mu=mu, step=step, value=self.value)

    def movebackward(self, mu, step=1):
        self.value = super().movebackward(
            mu=mu, step=step, value=self.value)

    def reducesum(self):
        return super().ReduceSum(value=self.value)

    def average(self):
        return super().Average(value=self.value)

    def im(self): 
        out = LatticeReal(grid=self.grid, cartesiancomm=self.cartesiancomm)
        out.value = np.imag(self.value)
        return out

    def re(self): 
        out = LatticeReal(grid=self.grid, cartesiancomm=self.cartesiancomm)
        out.value = np.real(self.value)
        return out

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
        self.value = np.zeros(shape=(np.prod(self.grid), N, N), dtype=REAL)

    def fill_value(self, n: RealMatrix):
        if isinstance(n, RealMatrix):
            self.value[:] = n.value
        if isinstance(n, np.ndarray):
            self.value = n

    def moveforward(self, mu, step=1):
        self.value = super().moveforward(mu=mu, step=step, value=self.value)

    def movebackward(self, mu, step=1):
        self.value = super().movebackward(mu=mu, step=step, value=self.value)

    def __getitem__(self, idx: int):
        return self.value[idx, :]

    def reducesum(self):
        return super().ReduceSum(value=self.value)

    def average(self):
        return super().Average(value=self.value)

    def transpose(self):
        out = LatticeRealMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, N=self.N)
        out.value = np.swapaxes(self.value, -1, -2)
        return out

    def trace(self):
        out = LatticeReal(grid=self.grid, cartesiancomm=self.cartesiancomm)
        out.value = np.trace(self.value, axis1=-2, axis2=-1)
        return out

    def det(self):
        out = LatticeReal(grid=self.grid, cartesiancomm=self.cartesiancomm)
        out.value = np.linalg.det(self.value)
        return out

    def inv(self):
        out = LatticeRealMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, N=self.N)
        out.value = np.linalg.inv(self.value)
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
            out.value = np.matmul(self.value, rhs.value)
        elif isinstance(rhs, RealMatrix):
            assert(self.value[0].shape == rhs.value.shape)
            out.value = np.matmul(self.value, rhs.value)
        elif isinstance(rhs, Real):
            out.value = self.value * rhs.value
        return out


class LatticeComplexMatrix(LatticeMPI):
    def __init__(self, grid, cartesiancomm, N: int):
        super().__init__(grid=grid, cartesiancomm=cartesiancomm)
        self.N = N
        self.value = np.zeros(shape=(np.prod(self.grid), N, N), dtype=COMPLEX)

    def fill_value(self, n: ComplexMatrix):
        if isinstance(n, ComplexMatrix):
            self.value[:] = n.value
        elif isinstance(n, np.ndarray):
            self.value = n

    def moveforward(self, mu, step=1):
        self.value = super().moveforward(mu=mu, step=step, value=self.value)

    def movebackward(self, mu, step=1):
        self.value = super().movebackward(
            mu=mu, step=step, value=self.value)

    def __getitem__(self, idx: int):
        return self.value[idx, :]

    def reducesum(self):
        return super().ReduceSum(value=self.value)

    def average(self):
        return super().Average(value=self.value)

    def transpose(self):
        out = LatticeComplexMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, N=self.N)
        out.value = np.swapaxes(self.value, -1, -2)
        return out

    def trace(self):
        out = LatticeComplex(grid=self.grid, cartesiancomm=self.cartesiancomm)
        out.value = np.trace(self.value, axis1=-2, axis2=-1)
        return out

    def det(self):
        out = LatticeComplex(grid=self.grid, cartesiancomm=self.cartesiancomm)
        out.value = np.linalg.det(self.value)
        return out

    def inv(self):
        out = LatticeComplexMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, N=self.N)
        out.value = np.linalg.inv(self.value)
        return out

    def conj(self):
        out = LatticeComplexMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, N=self.N)
        out.value = np.conj(self.value)
        return out

    def adj(self):
        return self.conj().transpose()

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
            out.value = np.matmul(self.value, rhs.value)
        elif isinstance(rhs, (RealMatrix, ComplexMatrix)):
            assert(self.value[0].shape == rhs.value.shape)
            out.value = np.matmul(self.value, rhs.value)
        elif isinstance(rhs, (Real, Complex)):
            out.value = self.value * rhs.value
        elif isinstance(rhs, (float, int, complex)):
            out.value = self.value * rhs
        return out


class LatticeVectorReal(LatticeMPI):
    def __init__(self, grid, cartesiancomm, Nd: int):
        super().__init__(grid=grid, cartesiancomm=cartesiancomm)
        self.Nd = Nd
        self.value = np.zeros(shape=(np.prod(self.grid), Nd), dtype=REAL)

    def fill_value(self, n: VectorReal):
        if isinstance(n, VectorReal):
            self.value[:] = n.value
        elif isinstance(n, np.ndarray):
            self.value[:] = n

    def moveforward(self, mu, step=1):
        self.value = super().moveforward(mu=mu, step=step, value=self.value)

    def movebackward(self, mu, step=1):
        self.value = super().movebackward(mu=mu, step=step, value=self.value)

    def reducesum(self):
        return super().ReduceSum(value=self.value)

    def average(self):
        return super().Average(value=self.value)

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
        out.value = self.value.copy()                           # the transpose of a vector is the vector
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
            out.value = np.einsum('ij,ij->i', self.value, rhs.value)
        elif isinstance(rhs, LatticeReal):
            out = LatticeVectorReal(
                grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd)
            out.value = self.value * rhs.value[:, None]
        elif isinstance(rhs, Real):
            out = LatticeVectorReal(
                grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd)
            out.value = self.value * rhs.value    
        return out


class LatticeVectorComplex(LatticeMPI):
    def __init__(self, grid, cartesiancomm, Nd: int):
        super().__init__(grid=grid, cartesiancomm=cartesiancomm)
        self.Nd = Nd
        self.value = np.zeros(shape=(np.prod(self.grid), Nd), dtype=COMPLEX)

    def fill_value(self, n: VectorComplex):
        if isinstance(n, VectorComplex):
            self.value[:] = n.value[:]
        elif isinstance(n, np.ndarray):
            self.value[:] = n

    def moveforward(self, mu, step=1):
        self.value = super().moveforward(mu=mu, step=step, value=self.value)

    def movebackward(self, mu, step=1):
        self.value = super().movebackward(
            mu=mu, step=step, value=self.value)

    def reducesum(self):
        return super().ReduceSum(value=self.value)

    def average(self):
        return super().Average(value=self.value)

    def peek_index(self, mu):
        """the lattice of the components in direction mu (a copy)"""
        out = LatticeComplex(grid=self.grid, cartesiancomm=self.cartesiancomm)
        out.value = self.value[:, mu].copy()
        return out

    def poke_index(self, mu, obj):
        """store a lattice in the components of direction mu"""
        self.value[:, mu] = obj.value

    def __getitem__(self, idx: int):
        return self.value[idx, :]

    def transpose(self):
        out = LatticeVectorComplex(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd)
        out.value = self.value.copy()
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
            out = LatticeReal(grid=self.grid, cartesiancomm=self.cartesiancomm)
            assert(self.value.shape == rhs.value.shape)
            out.value = np.einsum('ij,ij->i', self.value, rhs.value)
        elif isinstance(rhs, (LatticeReal, LatticeComplex)):
            out = LatticeVectorComplex(
                grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd)
            out.value = self.value * rhs.value[:, None]
        elif isinstance(rhs, (Complex, Real)):
            out = LatticeVectorComplex(
                grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd)
            out.value = self.value * rhs.value
        return out

    def conj(self):
        out = LatticeVectorComplex(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd)
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
        self.value = np.zeros(shape=(np.prod(self.grid), Nd, N, N), dtype=REAL)

    def fill_value(self, n: VectorRealMatrix):
        if isinstance(n, VectorRealMatrix):
            self.value[:] = n.value
        elif isinstance(n, np.ndarray):
            self.value = n

    def moveforward(self, mu, step=1):
        self.value = super().moveforward(mu=mu, step=step, value=self.value)

    def movebackward(self, mu, step=1):
        self.value = super().movebackward(mu=mu, step=step, value=self.value)

    def reducesum(self):
        return super().ReduceSum(value=self.value)

    def average(self):
        return super().Average(value=self.value)

    def peek_index(self, mu):
        """the lattice of the components in direction mu (a copy)"""
        out = LatticeRealMatrix(grid=self.grid, cartesiancomm=self.cartesiancomm, N=self.N)
        out.value = self.value[:, mu].copy()
        return out

    def poke_index(self, mu, obj):
        """store a lattice in the components of direction mu"""
        self.value[:, mu] = obj.value

    def __getitem__(self, idx: int):
        return self.value[idx, :]

    def transpose(self):
        out = LatticeVectorRealMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd, N=self.N)
        out.value = np.swapaxes(self.value, -1, -2)
        return out

    def trace(self):
        out = LatticeVectorReal(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd)
        out.value = np.trace(self.value, axis1=-2, axis2=-1)
        return out

    def det(self):
        out = LatticeVectorReal(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd)
        out.value = np.linalg.det(self.value)
        return out

    def inv(self):
        out = LatticeVectorRealMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd, N=self.N)
        out.value = np.linalg.inv(self.value)
        return out

    def __add__(self, rhs):
        out = LatticeVectorRealMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd, N=self.N)
        if isinstance(rhs, LatticeVectorRealMatrix):
            assert(self.value.shape == rhs.value.shape)
            out.value = self.value + rhs.value
        elif isinstance(rhs, LatticeRealMatrix):
            assert(self.value[0, 0].shape == rhs.value[0].shape)
            out.value = self.value + rhs.value[:, None]
        elif isinstance(rhs, (Real, RealMatrix)):
            assert(np.ndim(rhs.value) == 0 or self.value[0, 0].shape == rhs.value.shape)
            out.value = self.value + rhs.value
        elif isinstance(rhs, (float, int)):
            out.value = self.value + rhs
        return out

    def __sub__(self, rhs):
        out = LatticeVectorRealMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd, N=self.N)
        if isinstance(rhs, LatticeVectorRealMatrix):
            assert(self.value.shape == rhs.value.shape)
            out.value = self.value - rhs.value
        elif isinstance(rhs, LatticeRealMatrix):
            assert(self.value[0, 0].shape == rhs.value[0].shape)
            out.value = self.value - rhs.value[:, None]
        elif isinstance(rhs, (Real, RealMatrix)):
            assert(np.ndim(rhs.value) == 0 or self.value[0, 0].shape == rhs.value.shape)
            out.value = self.value - rhs.value
        elif isinstance(rhs, (float, int)):
            out.value = self.value - rhs
        return out

    def __mul__(self, rhs):
        out = LatticeVectorRealMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd, N=self.N)
        if isinstance(rhs, LatticeVectorRealMatrix):
            assert(self.value.shape == rhs.value.shape)
            out.value = np.matmul(self.value, rhs.value)
        elif isinstance(rhs, LatticeRealMatrix):
            out.value = np.matmul(self.value, rhs.value[:, None])
        elif isinstance(rhs, RealMatrix):
            out.value = np.matmul(self.value, rhs.value)
        elif isinstance(rhs, Real):
            out.value = self.value * rhs.value
        elif isinstance(rhs, (float, int)):
            out.value = self.value * rhs
        return out


class LatticeVectorComplexMatrix(LatticeMPI):
    def __init__(self, grid, cartesiancomm, Nd: int, N: int):
        super().__init__(grid=grid, cartesiancomm=cartesiancomm)
        self.Nd = Nd
        self.N = N
        self.value = np.zeros(shape=(np.prod(self.grid), Nd, N, N), dtype=COMPLEX)

    def fill_value(self, n: VectorComplexMatrix):
        if isinstance(n, VectorComplexMatrix):
            self.value[:] = n.value
        elif isinstance(n, np.ndarray):
            self.value = n

    def moveforward(self, mu, step=1):
        self.value = super().moveforward(mu=mu, step=step, value=self.value)

    def movebackward(self, mu, step=1):
        self.value = super().movebackward(
            mu=mu, step=step, value=self.value)

    def reducesum(self):
        return super().ReduceSum(value=self.value)

    def average(self):
        return super().Average(value=self.value)

    def peek_index(self, mu):
        """the lattice of the components in direction mu (a copy)"""
        out = LatticeComplexMatrix(grid=self.grid, cartesiancomm=self.cartesiancomm, N=self.N)
        out.value = self.value[:, mu].copy()
        return out

    def poke_index(self, mu, obj):
        """store a lattice in the components of direction mu"""
        self.value[:, mu] = obj.value

    def __getitem__(self, idx: int):
        return self.value[idx, :]

    def transpose(self):
        out = LatticeVectorComplexMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd, N=self.N)
        out.value = np.swapaxes(self.value, -1, -2)
        return out

    def trace(self):
        out = LatticeVectorComplex(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd)
        out.value = np.trace(self.value, axis1=-2, axis2=-1)
        return out

    def det(self):
        out = LatticeVectorComplex(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd)
        out.value = np.linalg.det(self.value)
        return out

    def inv(self):
        out = LatticeVectorComplexMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd, N=self.N)
        out.value = np.linalg.inv(self.value)
        return out

    def conj(self):
        out = LatticeVectorComplexMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd, N=self.N)
        out.value = np.conj(self.value)
        return out

    def adj(self):
        return self.conj().transpose()

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
            out.value = self.value + rhs.value[:, None]
        elif isinstance(rhs, (Real, RealMatrix, Complex, ComplexMatrix)):
            assert(np.ndim(rhs.value) == 0 or self.value[0, 0].shape == rhs.value.shape)
            out.value = self.value + rhs.value
        elif isinstance(rhs, (float, int, complex)):
            out.value = self.value + rhs
        return out

    def __sub__(self, rhs):
        out = LatticeVectorComplexMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd, N=self.N)
        if isinstance(rhs, (LatticeVectorComplexMatrix, LatticeVectorRealMatrix)):
            assert(self.value.shape == rhs.value.shape)
            out.value = self.value - rhs.value
        elif isinstance(rhs, (LatticeComplexMatrix, LatticeRealMatrix)):
            assert(self.value[0, 0].shape == rhs.value[0].shape)
            out.value = self.value - rhs.value[:, None]
        elif isinstance(rhs, (Real, RealMatrix, Complex, ComplexMatrix)):
            assert(np.ndim(rhs.value) == 0 or self.value[0, 0].shape == rhs.value.shape)
            out.value = self.value - rhs.value
        elif isinstance(rhs, (float, int, complex)):
            out.value = self.value - rhs
        return out

    def __mul__(self, rhs):
        out = LatticeVectorComplexMatrix(
            grid=self.grid, cartesiancomm=self.cartesiancomm, Nd=self.Nd, N=self.N)
        if isinstance(rhs, (LatticeVectorComplexMatrix, LatticeVectorRealMatrix)):
            assert(self.value.shape == rhs.value.shape)
            out.value = np.matmul(self.value, rhs.value)
        elif isinstance(rhs, (LatticeRealMatrix, LatticeComplexMatrix)):
            out.value = np.matmul(self.value, rhs.value[:, None])
        elif isinstance(rhs, (ComplexMatrix, RealMatrix)):
            out.value = np.matmul(self.value, rhs.value)
        elif isinstance(rhs, (Complex, Real)):
            out.value = self.value * rhs.value
        elif isinstance(rhs, (float, int, complex)):
            out.value = self.value * rhs
        return out
