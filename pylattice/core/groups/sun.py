"""SU(2) as a Lie group: the generators of the algebra, the exponential map and projection, and
unitarization. Pure math on small matrices (pylattice.core.tensors): no lattice and no MPI here.
The gauge theory built on these is in pylattice.theories.gauge.sun.

Note: written for SU(2) only (2x2 matrices, 3 generators), not yet for general N.
"""
import numpy as np
from scipy.linalg import expm

from pylattice.core.tensors import ComplexMatrix, VectorReal, VectorComplex


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
    return VectorComplex(value = np.array([ (-2.0j*get_generator(a)*input).trace().value[0] for a in range(adj_dim)]))

def ExpProjToGroup(input):
    T = np.array([get_generator(a).value for a in range(input.Nd)])
    w = np.expand_dims(input.value,axis=[1,2])
    M = np.sum(T*w*1j,axis=0)
    return ComplexMatrix(value=expm(M))
