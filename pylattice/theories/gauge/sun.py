"""SU(2) gauge field: one SU(2) matrix on each link of the lattice, with its updates.
The group and algebra operations it uses are in pylattice.core.groups.sun.

Staple, Plaq and Action are not implemented yet.
"""
import numpy as np

from pylattice.core.grid import LatticeVectorComplexMatrix
from pylattice.core.tensors import ComplexMatrix
from pylattice.core.groups.sun import (get_generator, random_lie_algebra_vector, unitarize,
                                       ProjToAlgebra, ExpProjToGroup)


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
        r[0]  = np.random.random()-0.5
        r[1:] = random_lie_algebra_vector().value
        r[0] =np.sign(r[0])*np.sqrt(1-float(eps)*float(eps)) 
        r[1:]=float(eps)*r[1:]/np.linalg.norm(r[1:])    # |r[1:]| = eps, so r0^2 + |r[1:]|^2 = 1
        return r

    def metropolis_update(self,eps=0.001): 
        for i in range(np.prod(self.grid)):
            for mu in range(self.Nd): 
                r = self.tepid_lie_algebra_vector(eps)
                X = np.eye(self.N,dtype=complex)*r[0]
                for a in range(self.adj_dim):
                    X += (get_generator(a)*r[a+1]*2j).value     # generators are sigma/2: X = r0 + i r.sigma is in SU(2)
                self.value[i,mu,:,:] = np.dot(X,self.value[i,mu,:,:])
    
    def Staple(self): 
        exit(1)
    def Plaq(self):
        exit(1)
    def Action(self): 
        exit(1)