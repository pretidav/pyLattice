from lattice.grid import *
tol = 10e-6

def test_moveforward():
    Grid = LatticeBase(size=[4,4])
    Grid.moveforward(mu=1)
    Grid.moveforward(mu=1)
    assert(np.sum(Grid.coor-np.array([0,2]))<tol)

def test_movebackward():
    Grid = LatticeBase(size=[4,4])
    Grid.movebackward(mu=0)
    assert(np.sum(Grid.coor-np.array([3,0]))<tol)
