"""SU(2) generators, algebra and group elements (pylattice.core.groups.sun). Run as a script:
python tests/core/test_groups_sun.py"""
from pylattice.core.groups.sun import get_generator, structure_constants, unitarize, ProjToAlgebra, ExpProjToGroup
from pylattice.core.tensors import Kroneker, ComplexMatrix
import numpy as np

tol = 10e-6

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
