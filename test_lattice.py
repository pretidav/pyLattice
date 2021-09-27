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

def test_idx():
    Grid = LatticeBase(size=[4,4])
    Grid.moveforward(mu=1)
    Grid.moveforward(mu=0)
    assert(Grid.get_idx(Grid.coor)==5)


# Grid = LatticeBase(size=[4,4])
# RealField = LatticeReal(lattice=Grid)
# RealField.fill_grid(0)
# print(RealField.value)

# ComplexField = LatticeComplex(lattice=Grid)
# ComplexField.fill_grid(1j)
# print(ComplexField.value)
