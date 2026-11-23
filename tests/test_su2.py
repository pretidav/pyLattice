from SUn.SU2 import SU2Field, get_generator, structure_constants,unitarize, random_lie_algebra_vector, ProjToAlgebra, ExpProjToGroup
from linalg.tensors import Kroneker, ComplexMatrix
import numpy as np
from mpi.parallel import CartesianComm

tol = 10e-6
mpigrid = [1, 1]
Nd = 2
grid = [3, 3]
tol = 10e-6
CC = CartesianComm(mpigrid=mpigrid)

SU2 = SU2Field(grid=grid,cartesiancomm=CC,Nd=2)

N=2
AdjD=N**2-1
T=[]
for a in range(AdjD):
    T.append(get_generator(a))

# Traceless generator test
for a in range(AdjD):
    assert( (np.abs(T[a].trace().value)<tol).all() )

# Hermitean generator test
for a in range(AdjD):
    assert( (np.abs(T[a].trace().value - T[a].adj().trace().value)<tol).all() )

# Trace Normalization generator test
for a in range(AdjD):
    for b in range(AdjD): 
        assert( ((T[a]*T[b]).trace().im().value<tol).all() )
        assert( (np.abs((T[a]*T[b]).trace().value-0.5*Kroneker(a,b))<tol).all() )

# Algebra structure Test
for a in range(AdjD): 
    for b in range(AdjD):
        if b!=a: 
            for c in range(AdjD):
                if c!=b and c!=a:
                    assert( (np.abs((T[a]*T[b] - T[b]*T[a] + T[c]*(-1j*structure_constants(a,b,c))).value)<tol).all() ) 

Arand = ComplexMatrix(value= (np.random.random(size=(2,2)) + 1j*np.random.random(size=(2,2)) ))
B = unitarize(ExpProjToGroup(ProjToAlgebra(Arand)))
assert( (np.abs(B.det().value-1) < tol).all() )
assert( (np.abs((B.adj()*B).value - np.eye(2))<tol).all() )
assert( (np.abs((B*(B.adj())).value - np.eye(2))<tol).all() )


A = SU2Field(grid=grid,cartesiancomm=CC,Nd=Nd)
A.random_config()
B = SU2Field(grid=grid,cartesiancomm=CC,Nd=Nd)
B.random_config()
assert( (np.abs(A.det().value-1) < tol).all() )
assert( (np.abs(B.det().value-1) < tol).all() )
assert( (np.abs((A*B).det().value-1)< tol).all() )
assert( (np.abs((A.adj()*A).value - np.eye(2))<tol).all() )
assert( (np.abs((A*(A.adj())).value - np.eye(2))<tol).all() )

# perturbation
A.value += (np.random.random(size=A.value.shape) + 1j*np.random.random(size=A.value.shape))

A.unitarize()
assert( (np.abs(A.det().value-1) < tol).all() )
assert( (np.abs((A.adj()*A).value - np.eye(2))<tol).all() )
assert( (np.abs((A*(A.adj())).value - np.eye(2))<tol).all() )
print(A.value)
A.metropolis_update(eps=0.001)
assert( (np.abs(A.det().value-1) < tol).all() )
assert( (np.abs((A.adj()*A).value - np.eye(2))<tol).all() )
assert( (np.abs((A*(A.adj())).value - np.eye(2))<tol).all() )
print(A.value)
