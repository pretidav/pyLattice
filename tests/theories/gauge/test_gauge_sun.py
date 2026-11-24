"""SU(2) gauge field: random configurations, unitarization, Metropolis update
(pylattice.theories.gauge.sun). Run as a script: python tests/theories/gauge/test_gauge_sun.py"""
from pylattice.theories.gauge.sun import SU2Field
from pylattice.core.parallel import CartesianComm
import numpy as np

tol = 10e-6
mpigrid = [1, 1]
Nd = 2
grid = [3, 3]
CC = CartesianComm(mpigrid=mpigrid)

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
