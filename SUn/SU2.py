from linalg.tensors import ComplexMatrix
from lattice.grid import LatticeVectorComplexMatrix, LatticeVectorReal, LatticeComplexMatrix, ComplexMatrix, VectorReal

import numpy as np 
from scipy.linalg import expm

def get_generator(k):
        if k==0: 
            out = ComplexMatrix(value = np.array([[0,1],[1,0]],dtype='complex64')/2 )
        if k==1: 
            out = ComplexMatrix(value =  np.array([[0,-1j],[1j,0]],dtype='complex64')/2 )
        if k==2: 
            out = ComplexMatrix(value =  np.array([[1,0],[0,-1]],dtype='complex64')/2 )
        return out 

def structure_constants(i,j,k): 
    if (i,j,k) in [(1,0,2),(2,1,0),(0,2,1)]: 
        return -1 
    else: 
        if i!=j and j!=k and i!=k: 
            return 1 
        else : 
            return 0 

def random_lie_algebra_vector(): 
    adj_dim = 3
    return VectorReal(value=np.array(np.random.random(size=adj_dim),dtype=float)-0.5)

def unitarize(input): 
    out = ComplexMatrix(N=2)
    u, _ = np.linalg.qr(input.value,mode='complete')
    out.value = u
    return  out

def ProjToAlgebra(input):
    adj_dim = 3    
    return VectorReal(value = np.array([ (-2.0j*get_generator(a)*input).trace().value[0] for a in range(adj_dim)]))

def ExpProjToGroup(input):
    T = np.array([get_generator(a).value for a in range(input.Nd)])
    w = np.expand_dims(input.value,axis=[1,2])
    M = np.sum(T*w*1j,axis=0)
    return ComplexMatrix(value=expm(M))     
   
class SU2Field(LatticeVectorComplexMatrix): 
    def __init__(self,grid,cartesiancomm,Nd): 
        super().__init__(grid=grid,cartesiancomm=cartesiancomm, Nd=Nd, N=2)
        self.Nd = Nd
        self.cartesiancomm = cartesiancomm
        self.N = 2
        self.adj_dim = self.N*self.N - 1  
        self.value = self.trivial_config()

    def trivial_config(self): 
        out = np.zeros(self.value.shape,dtype=complex)
        for mu in range(self.Nd):
            for i in range(np.prod(self.grid)): 
                out[i,mu,:,:]=np.eye(self.N,self.N, dtype=complex)
        return out 

    def random_config(self):
        for i in range(np.prod(self.grid)):
            for mu in range(self.Nd): 
                self.value[i,mu,:,:]=ExpProjToGroup(random_lie_algebra_vector()).value

    def unitarize(self): 
        for i in range(np.prod(self.grid)):
            for mu in range(self.Nd): 
                self.value[i,mu,:,:] = unitarize(ExpProjToGroup(ProjToAlgebra(ComplexMatrix(value=self.value[i,mu,:,:])))).value

    def tepid_lie_algebra_vector(self,eps): 
        r = np.zeros(shape=4)
        r[0]  = np.random.random(size=1)-0.5
        r[1:] = random_lie_algebra_vector().value
        r[0] =np.sign(r[0])*np.sqrt(1-float(eps)*float(eps)) 
        r[1:]=float(eps)*r[1:]/np.dot(r[1:],r[1:])
        return r

    def metropolis_update(self,eps=0.001): 
        for i in range(np.prod(self.grid)):
            for mu in range(self.Nd): 
                r = self.tepid_lie_algebra_vector(eps)
                X = np.eye(self.N,dtype=complex)*r[0]
                for a in range(self.adj_dim):
                    X += (get_generator(a)*r[a+1]*1j).value
                self.value[i,mu,:,:] = np.dot(X,self.value[i,mu,:,:])
    
    def Staple(self): 
        