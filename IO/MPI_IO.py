
import h5py
import numpy as np
from datetime import datetime
import platform,socket,psutil
from lattice.grid import LatticeReal, LatticeComplex, LatticeRealMatrix, LatticeComplexMatrix, LatticeVectorReal, LatticeVectorComplex, LatticeVectorRealMatrix, LatticeVectorComplexMatrix
from mpi.parallel import pprint
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

class Writer():
    def __init__(self,lattice,cartesiancomm): 
        self.lattice = lattice
        self.cartesiancomm = cartesiancomm

    def write_header(self,file):
            file.attrs['grid'] = self.lattice.grid*self.cartesiancomm.mpigrid
            file.attrs['type'] = str(type(self.lattice))
            
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
        if filename[:-5]!='.hdf5':
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
        if filename[:-5]!='.hdf5':
            filename = filename + '.hdf5'
        file = h5py.File(filename, 'r', driver='mpio', comm=self.cartesiancomm.comm)

        self.grid = [int(a/b) for a,b in zip(file.attrs['grid'],self.cartesiancomm.mpigrid)]
        self.latticetype = file.attrs['type'] 

        if self.latticetype == "<class 'lattice.grid.LatticeReal'>": 
            self.lattice = LatticeReal(grid=self.grid,cartesiancomm=self.cartesiancomm)
            
        elif self.latticetype == "<class 'lattice.grid.LatticeComplex'>": 
            self.lattice = LatticeComplex(grid=self.grid,cartesiancomm=self.cartesiancomm)

        elif self.latticetype == "<class 'lattice.grid.LatticeRealMatrix'>": 
            self.lattice = LatticeRealMatrix(grid=self.grid,cartesiancomm=self.cartesiancomm, N=int(file.attrs['N']))

        elif self.latticetype == "<class 'lattice.grid.LatticeComplexMatrix'>":     
            self.lattice = LatticeComplexMatrix(grid=self.grid,cartesiancomm=self.cartesiancomm, N=int(file.attrs['N']))

        elif self.latticetype == "<class 'lattice.grid.LatticeVectorReal'>":     
            self.lattice = LatticeVectorReal(grid=self.grid,cartesiancomm=self.cartesiancomm, Nd=int(file.attrs['Nd']))

        elif self.latticetype == "<class 'lattice.grid.LatticeVectorComplex'>":     
            self.lattice = LatticeVectorComplex(grid=self.grid,cartesiancomm=self.cartesiancomm, Nd=int(file.attrs['Nd']))
        
        elif self.latticetype == "<class 'lattice.grid.LatticeVectorRealMatrix'>":     
            self.lattice = LatticeVectorRealMatrix(grid=self.grid,cartesiancomm=self.cartesiancomm, N=int(file.attrs['N']),  Nd=int(file.attrs['Nd']))

        elif self.latticetype == "<class 'lattice.grid.LatticeVectorComplexMatrix'>":     
            self.lattice = LatticeVectorComplexMatrix(grid=self.grid,cartesiancomm=self.cartesiancomm, N=int(file.attrs['N']),  Nd=int(file.attrs['Nd']))

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
            self.lattice.value[self.lattice.get_idx(idx)] = list(file['value'])[glob_idx_serialized]
        return self.lattice        
        
