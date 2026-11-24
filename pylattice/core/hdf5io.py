
import h5py
import numpy as np
from datetime import datetime
import platform,socket,psutil
from pylattice.core.grid import LatticeReal, LatticeComplex, LatticeRealMatrix, LatticeComplexMatrix, LatticeVectorReal, LatticeVectorComplex, LatticeVectorRealMatrix, LatticeVectorComplexMatrix
from pylattice.core.parallel import pprint
from mpi4py import MPI

def getSystemInfo():
    info={}
    info['platform']=platform.system()
    info['architecture']=platform.machine()
    info['hostname']=socket.gethostname()
    info['processor']=platform.processor()
    info['ram']=str(round(psutil.virtual_memory().total / (1024.0 **3)))+" GB"
    info['platform-release']=platform.release()
    info['platform-version']=platform.version()       
    return info 

_LATTICE_TYPES = (LatticeReal, LatticeComplex, LatticeRealMatrix, LatticeComplexMatrix,
                  LatticeVectorReal, LatticeVectorComplex, LatticeVectorRealMatrix, LatticeVectorComplexMatrix)


class Writer():
    def __init__(self,lattice,cartesiancomm): 
        self.lattice = lattice
        self.cartesiancomm = cartesiancomm

    def write_header(self,file):
            file.attrs['grid'] = self.lattice.grid*self.cartesiancomm.mpigrid
            # a subclass (an IsingField, an SU2Field) is stored as the lattice class it derives from
            base = next((c for c in type(self.lattice).__mro__ if c in _LATTICE_TYPES), None)
            if base is None:
                raise TypeError("cannot write a {}: it is not a lattice".format(type(self.lattice).__name__))
            file.attrs['type'] = base.__name__
            
            if isinstance(self.lattice,(LatticeRealMatrix,LatticeComplexMatrix,LatticeVectorRealMatrix,LatticeVectorComplexMatrix)):
                file.attrs['N'] = self.lattice.N
            if isinstance(self.lattice,(LatticeVectorReal,LatticeVectorComplex,LatticeVectorRealMatrix,LatticeVectorComplexMatrix)):
                file.attrs['Nd'] = self.lattice.Nd

            sys_info = getSystemInfo()
            file.attrs['date']=str(datetime.now())
            file.attrs['platform'] = sys_info['platform']
            file.attrs['platform-release'] = sys_info['platform-release']
            file.attrs['platform-version'] = sys_info['platform-version']
            file.attrs['architecture'] = sys_info['architecture']
            file.attrs['hostname'] = sys_info['hostname']
            file.attrs['processor'] = sys_info['processor']
            file.attrs['ram'] = sys_info['ram']

    def write(self,filename): 
        if filename[-5:]!='.hdf5':
            filename = filename + '.hdf5'
        file = h5py.File(filename, 'w', driver='mpio', comm=self.cartesiancomm.comm)
       
        self.write_header(file=file)
        
        if isinstance(self.lattice, LatticeReal):
            dset = file.create_dataset('value', shape=int(np.prod(self.lattice.grid*self.cartesiancomm.mpigrid)), dtype='float32')
        elif isinstance(self.lattice, LatticeComplex):
            dset = file.create_dataset('value', shape=int(np.prod(self.lattice.grid*self.cartesiancomm.mpigrid)), dtype='complex64')
        elif isinstance(self.lattice, LatticeRealMatrix):
            dset = file.create_dataset('value', shape=(int(np.prod(self.lattice.grid*self.cartesiancomm.mpigrid)), int(self.lattice.N), int(self.lattice.N)), dtype='float32')
        elif isinstance(self.lattice, LatticeComplexMatrix):
            dset = file.create_dataset('value', shape=(int(np.prod(self.lattice.grid*self.cartesiancomm.mpigrid)), int(self.lattice.N), int(self.lattice.N)), dtype='complex64')
        elif isinstance(self.lattice, LatticeVectorReal):
            dset = file.create_dataset('value', shape=(int(np.prod(self.lattice.grid*self.cartesiancomm.mpigrid)), int(self.lattice.Nd)), dtype='float32')
        elif isinstance(self.lattice, LatticeVectorComplex):
            dset = file.create_dataset('value', shape=(int(np.prod(self.lattice.grid*self.cartesiancomm.mpigrid)), int(self.lattice.Nd)), dtype='complex64')
        elif isinstance(self.lattice, LatticeVectorRealMatrix):
            dset = file.create_dataset('value', shape=(int(np.prod(self.lattice.grid*self.cartesiancomm.mpigrid)), int(self.lattice.Nd), int(self.lattice.N), int(self.lattice.N)), dtype='float32')
        elif isinstance(self.lattice, LatticeVectorComplexMatrix):
            dset = file.create_dataset('value', shape=(int(np.prod(self.lattice.grid*self.cartesiancomm.mpigrid)), int(self.lattice.Nd), int(self.lattice.N), int(self.lattice.N)), dtype='complex64')
        else : 
            pprint(self.cartesiancomm.comm,'error')
            exit(1)
       
        flat_glob_idx = np.array([i for i in range(np.prod(self.lattice.grid*self.cartesiancomm.mpigrid))])

        #pprint(self.cartesiancomm.comm,flat_glob_idx)
        for idx in np.ndindex(tuple(self.lattice.grid)):
            coor = self.cartesiancomm.mpicoord
            glob_idx = [] 
            glob_dim = []
            for d in reversed(range(self.lattice.dimensions)): 
                glob_idx.extend( [idx[d],coor[d]] )
                glob_dim.extend( [self.lattice.grid[d],self.cartesiancomm.mpigrid[d]])
            glob_idx = tuple(glob_idx)
            glob_dim = tuple(glob_dim)
            glob_idx_serialized = np.ravel_multi_index(glob_idx, glob_dim,order='F')
            #print('idx {} coor {} serial {} value[{}]={}'.format(idx,coor,glob_idx_serialized,idx,self.lattice.value[self.lattice.get_idx(idx)]))
            dset[glob_idx_serialized]=self.lattice.value[self.lattice.get_idx(idx)]
        file.close()
     
class Reader():
    def __init__(self,cartesiancomm): 
        self.cartesiancomm = cartesiancomm
        self.grid = None
        self.lattice = None

    def read(self,filename): 
        if filename[-5:]!='.hdf5':
            filename = filename + '.hdf5'
        file = h5py.File(filename, 'r', driver='mpio', comm=self.cartesiancomm.comm)

        self.grid = [int(a/b) for a,b in zip(file.attrs['grid'],self.cartesiancomm.mpigrid)]
        # the class name; files written by older versions hold the full path, "<class 'lattice.grid.LatticeReal'>"
        self.latticetype = str(file.attrs['type']).rstrip("'>").split('.')[-1]

        if self.latticetype == 'LatticeReal': 
            self.lattice = LatticeReal(grid=self.grid,cartesiancomm=self.cartesiancomm)
            
        elif self.latticetype == 'LatticeComplex': 
            self.lattice = LatticeComplex(grid=self.grid,cartesiancomm=self.cartesiancomm)

        elif self.latticetype == 'LatticeRealMatrix': 
            self.lattice = LatticeRealMatrix(grid=self.grid,cartesiancomm=self.cartesiancomm, N=int(file.attrs['N']))

        elif self.latticetype == 'LatticeComplexMatrix':     
            self.lattice = LatticeComplexMatrix(grid=self.grid,cartesiancomm=self.cartesiancomm, N=int(file.attrs['N']))

        elif self.latticetype == 'LatticeVectorReal':     
            self.lattice = LatticeVectorReal(grid=self.grid,cartesiancomm=self.cartesiancomm, Nd=int(file.attrs['Nd']))

        elif self.latticetype == 'LatticeVectorComplex':     
            self.lattice = LatticeVectorComplex(grid=self.grid,cartesiancomm=self.cartesiancomm, Nd=int(file.attrs['Nd']))
        
        elif self.latticetype == 'LatticeVectorRealMatrix':     
            self.lattice = LatticeVectorRealMatrix(grid=self.grid,cartesiancomm=self.cartesiancomm, N=int(file.attrs['N']),  Nd=int(file.attrs['Nd']))

        elif self.latticetype == 'LatticeVectorComplexMatrix':     
            self.lattice = LatticeVectorComplexMatrix(grid=self.grid,cartesiancomm=self.cartesiancomm, N=int(file.attrs['N']),  Nd=int(file.attrs['Nd']))

        data = file['value'][...]                      # the dataset is read once, not once per site
        for idx in np.ndindex(tuple(self.lattice.grid)):
            coor = self.cartesiancomm.mpicoord
            glob_idx = [] 
            glob_dim = []
            for d in reversed(range(self.lattice.dimensions)): 
                glob_idx.extend( [idx[d],coor[d]] )
                glob_dim.extend( [self.lattice.grid[d],self.cartesiancomm.mpigrid[d]])
            glob_idx = tuple(glob_idx)
            glob_dim = tuple(glob_dim)
            glob_idx_serialized = np.ravel_multi_index(glob_idx, glob_dim,order='F') 
            self.lattice.value[self.lattice.get_idx(idx)] = data[glob_idx_serialized]
        file.close()
        return self.lattice        
        


def describe(filename, comm):
    """What is on disk: {'attrs': {name: value}, 'datasets': {name: (shape, dtype)}}. Rank 0 opens the
    file and broadcasts; a file that cannot be read raises the same error on every rank."""
    info = None
    if comm.Get_rank() == 0:
        try:
            with h5py.File(filename, 'r') as f:
                attrs = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in f.attrs.items()}
                datasets = {}
                f.visititems(lambda name, obj: datasets.__setitem__(name, (tuple(obj.shape), str(obj.dtype)))
                             if isinstance(obj, h5py.Dataset) else None)
            info = {'attrs': attrs, 'datasets': datasets}
        except Exception as error:
            info = {'error': '{}: {}'.format(type(error).__name__, error)}
    info = comm.bcast(info, root=0)
    if 'error' in info:
        raise OSError("cannot read '{}' as an HDF5 file ({})".format(filename, info['error']))
    return info


def load_into(lattice, filename, log=None, label='restart'):
    """Fills `lattice` with the configuration saved in `filename` (a Checkpointer / Writer file).

    The structure of the file is looked at first and compared with the lattice to fill: type, global grid,
    number of values per site and the shape of the dataset; whatever differs is a ValueError that names
    all the differences, before anything is read. The layout of the ranks may be different from the one that
    wrote the file. The file is read once. A collective call: every rank must make it.

    log: a function that takes a line (e.g. logger.info), to report what is found; label starts the lines.
    """
    log = log if log is not None else (lambda line: None)
    cc = lattice.cartesiancomm
    comm = cc.comm
    info = describe(filename, comm)
    attrs, datasets = info['attrs'], info['datasets']

    log(label + ": file {}".format(filename))
    log(label + ": on disk: type={} grid={} written by {}x{} on {} at {}, step={}".format(
        attrs.get('type'), 'x'.join(str(g) for g in attrs.get('grid', [])),
        attrs.get('platform', '?'), attrs.get('architecture', '?'), attrs.get('hostname', '?'),
        attrs.get('date', '?'), attrs.get('step', 'not recorded')))
    for name, (shape, dtype) in datasets.items():
        log(label + ": on disk: dataset '{}' shape={} dtype={}".format(name, shape, dtype))

    base = next((c for c in type(lattice).__mro__ if c in _LATTICE_TYPES), None)
    want_grid = [int(g) * int(m) for g, m in zip(lattice.grid, cc.mpigrid)]
    sites = int(np.prod(want_grid))
    want_shape = (sites,) + tuple(int(x) for x in (
        [lattice.Nd] if hasattr(lattice, 'Nd') else []) + ([lattice.N, lattice.N] if hasattr(lattice, 'N') else []))
    log(label + ": to fill: {} (a {}) global grid={} on {} ranks ({}), expects dataset shape={}".format(
        type(lattice).__name__, base.__name__ if base else '?', 'x'.join(map(str, want_grid)),
        comm.Get_size(), 'x'.join(map(str, cc.mpigrid)), want_shape))

    problems = []
    if base is None or str(attrs.get('type')).rstrip("'>").split('.')[-1] != base.__name__:
        problems.append("type on disk is {}, the field to fill is a {}".format(
            attrs.get('type'), base.__name__ if base else type(lattice).__name__))
    if [int(g) for g in attrs.get('grid', [])] != want_grid:
        problems.append("global grid on disk is {}, the field has {}".format(
            'x'.join(str(g) for g in attrs.get('grid', [])), 'x'.join(map(str, want_grid))))
    if 'value' not in datasets:
        problems.append("there is no dataset 'value' in the file")
    elif datasets['value'][0] != want_shape:
        problems.append("dataset 'value' has shape {}, expected {}".format(datasets['value'][0], want_shape))
    if problems:
        raise ValueError("cannot restart from '{}': {}".format(filename, '; '.join(problems)))
    log(label + ": the structure on disk matches the field: reading")

    loaded = Reader(cartesiancomm=cc).read(filename)
    lattice.value[...] = loaded.value
    comm.Barrier()
    log(label + ": configuration loaded ({} sites)".format(sites))
    return attrs.get('step')


def restore_rng(rng, filename, comm, mpigrid, log=None):
    """Puts the random number generator of this rank back in the state it had when `filename` was saved
    (by a Checkpointer), so that the run continues exactly as it would have without stopping. Returns True
    if the state was restored. A collective call.

    It needs the same layout of ranks as the run that wrote the file (each rank has its own stream); with a
    different layout, or a file that has no state, the generator is left as it is, with a warning in `log`,
    and the continuation is a valid run but not the same one.
    """
    import json
    log = log if log is not None else (lambda line: None)
    attrs = describe(filename, comm)['attrs']
    if 'rng_states' not in attrs:
        log("restart: WARNING: the file has no random state: the random numbers start afresh")
        return False
    saved = [int(m) for m in attrs['rng_mpigrid']]
    if saved != [int(m) for m in mpigrid]:
        log("restart: WARNING: the random state was saved on ranks {} and this run has {}: each rank has "
            "its own stream, so the random numbers start afresh".format(
                'x'.join(map(str, saved)), 'x'.join(str(int(m)) for m in mpigrid)))
        return False
    states = json.loads(attrs['rng_states'])
    state = states[comm.Get_rank()]
    if state['bit_generator'] != type(rng.bit_generator).__name__:
        raise ValueError("the random state on disk is of a {}, this run uses a {}".format(
            state['bit_generator'], type(rng.bit_generator).__name__))
    rng.bit_generator.state = state
    log("restart: random state of the {} ranks restored ({}): the run continues deterministically".format(
        len(states), state['bit_generator']))
    return True
