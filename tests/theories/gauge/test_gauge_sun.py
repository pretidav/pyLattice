"""SU(2) gauge field (pylattice.theories.gauge.sun): the lattice of links and its updates. The properties of
the group itself (generators, exponential, quaternions, heat bath) are tested in tests/core/test_sun.py.
One process (a lattice on a 1x1 rank grid). Run with:  pytest tests/theories/gauge/test_gauge_sun.py
"""
import numpy as np
import pytest

from pylattice.core.groups import su2 as sun
from pylattice.core.parallel import CartesianComm
from pylattice.theories.gauge.sun import SU2Field

GRID, ND = [4, 6], 2
SITES = 4 * 6
I2 = np.eye(2)


def dagger(M):
    return np.conj(np.swapaxes(M, -1, -2))


def assert_su2(U, tol=1e-12):
    assert np.abs(U @ dagger(U) - I2).max() < tol and np.abs(np.linalg.det(U) - 1).max() < tol


def field(rng=None, grid=GRID, nd=ND):
    return SU2Field(grid=grid, cartesiancomm=CartesianComm(mpigrid=[1] * len(grid)), Nd=nd, rng=rng)


def test_shape_dtype_and_cold_start():
    A = field()
    assert A.value.shape == (SITES, ND, 2, 2) and A.value.dtype == np.complex128
    assert (A.N, A.Nd, A.adj_dim) == (2, ND, 3)
    assert np.array_equal(A.value, np.broadcast_to(I2, A.value.shape))          # every link the identity
    assert np.array_equal(A.trivial_config(), A.value)


def test_random_config_is_su2_on_every_link_and_haar_distributed():
    A = field(np.random.default_rng(1), grid=[40, 40], nd=4)                    # 6400 links
    A.random_config()
    assert_su2(A.value)
    q = sun.to_quaternion(A.value.reshape(-1, 2, 2))
    assert np.abs(q.mean(axis=0)).max() < 0.04 and np.abs(q.T @ q / len(q) - np.eye(4) / 4).max() < 0.03
    assert np.abs(np.linalg.det(A.value) - 1).max() < 1e-12


def test_random_config_uses_its_generator_not_the_global_state():
    a, b, c = (field(np.random.default_rng(s)) for s in (5, 5, 6))
    np.random.seed(0)
    a.random_config()
    np.random.seed(123)                                                         # the global numpy state does not matter
    b.random_config()
    c.random_config()
    assert np.array_equal(a.value, b.value) and not np.array_equal(a.value, c.value)
    a.random_config(np.random.default_rng(9))                                    # a generator given to the call is used instead
    d = field()
    d.random_config(np.random.default_rng(9))
    assert np.array_equal(a.value, d.value)
    assert not np.array_equal(a.value[:, 0], a.value[:, 1])                      # the directions are independent


def test_the_default_generator_is_the_stream_of_the_rank_and_goes_on():
    a, b = field(), field()
    a.random_config()
    b.random_config()
    assert np.array_equal(a.value, b.value)                                      # the same default seed
    a.random_config()
    assert not np.array_equal(a.value, b.value)                                  # the second call continues the stream, it does not restart it


def test_products_and_adjoints_of_fields_of_links_are_su2():
    A, B = field(np.random.default_rng(2)), field(np.random.default_rng(3))
    A.random_config()
    B.random_config()
    assert np.abs(A.det().value - 1).max() < 1e-6 and np.abs((A * B).det().value - 1).max() < 1e-6
    assert np.abs((A.adj() * A).value - I2).max() < 1e-6 and np.abs((A * A.adj()).value - I2).max() < 1e-6
    assert_su2((A * B).value, 1e-6)                                              # the lattice product is the matrix product


def test_unitarize_brings_a_perturbed_field_back_to_the_nearest_su2():
    rng = np.random.default_rng(4)
    A = field(rng)
    A.random_config()
    exact = A.value.copy()
    noise = 1e-3 * (rng.standard_normal(A.value.shape) + 1j * rng.standard_normal(A.value.shape))
    A.value = exact + noise
    A.unitarize()
    assert_su2(A.value)
    assert np.abs(A.value - exact).max() < 5e-3                                  # each link stays next to where it was: not replaced by another
    before = A.value.copy()
    A.unitarize()
    assert np.abs(A.value - before).max() < 1e-13                                # nothing to do on SU(2)


def test_unitarize_of_a_strongly_perturbed_field_is_still_su2():
    rng = np.random.default_rng(5)
    A = field(rng)
    A.random_config()
    A.value += rng.random(A.value.shape) + 1j * rng.random(A.value.shape)
    A.unitarize()
    assert_su2(A.value)


def test_unitarize_removes_the_rounding_errors_of_many_updates():
    A = field(np.random.default_rng(6))
    A.random_config()
    for _ in range(200):
        A.metropolis_update(eps=0.05)
        A.value = (A.value * (1 + 1e-6)).astype(np.complex128)                    # a slow drift away from SU(2)
    assert np.abs(np.linalg.det(A.value) - 1).max() > 1e-5
    A.unitarize()
    assert_su2(A.value)


def test_metropolis_update_is_a_small_step_that_keeps_su2():
    A = field(np.random.default_rng(7))
    A.random_config()
    before = A.value.copy()
    A.metropolis_update(eps=0.01)
    assert_su2(A.value)
    # X U with X at distance eps from the identity: ||X - 1||_F^2 = 4 (1 - a0) = 4 (1 - sqrt(1 - eps^2))
    step = np.linalg.norm((A.value - before).reshape(-1, 4), axis=1)
    assert np.abs(step - np.sqrt(4 * (1 - np.sqrt(1 - 0.01 ** 2)))).max() < 1e-10
    assert (step > 0).all()


def test_metropolis_update_with_zero_eps_changes_nothing_and_default_eps_is_small():
    A = field(np.random.default_rng(8))
    A.random_config()
    before = A.value.copy()
    A.metropolis_update(eps=0.0)
    assert np.abs(A.value - before).max() < 1e-15
    A.metropolis_update()                                                        # the default eps, 0.001
    assert 0 < np.abs(A.value - before).max() < 0.01
    with pytest.raises(ValueError):
        A.metropolis_update(eps=2.0)


def test_metropolis_update_is_a_random_walk_that_explores_the_group():
    A = field(np.random.default_rng(9), grid=[20, 20], nd=2)
    start = A.value.copy()                                                       # cold
    for _ in range(400):
        A.metropolis_update(eps=0.3)
    assert_su2(A.value)
    a0 = sun.to_quaternion(A.value.reshape(-1, 2, 2))[:, 0]
    assert abs(np.mean(a0)) < 0.2 and np.std(a0) > 0.3                           # it has left the identity: spread over the group
    assert np.abs(A.value - start).max() > 0.5


def test_metropolis_update_follows_the_generator_and_is_independent_per_link():
    a, b = field(np.random.default_rng(10)), field(np.random.default_rng(10))
    a.metropolis_update(eps=0.1)
    b.metropolis_update(eps=0.1)
    assert np.array_equal(a.value, b.value)
    c = field(np.random.default_rng(11))
    c.metropolis_update(eps=0.1)
    assert not np.array_equal(a.value, c.value)
    flat = a.value.reshape(-1, 4)
    assert len({tuple(np.round(x, 12)) for x in flat}) == len(flat)              # every link its own step
    d = field()
    d.metropolis_update(eps=0.1, rng=np.random.default_rng(10))                  # a generator given to the call
    assert np.array_equal(d.value, a.value)


def test_tepid_vector_is_a_unit_quaternion_at_distance_eps_from_the_identity():
    A = field(np.random.default_rng(12))
    q = A.tepid_lie_algebra_vector(0.2)
    assert q.shape == (4,) and abs(np.linalg.norm(q) - 1) < 1e-14
    assert abs(np.linalg.norm(q[1:]) - 0.2) < 1e-14 and q[0] > 0                 # near +1, not near -1 (the centre)


def test_the_lattice_shifts_keep_the_links_in_su2():
    A = field(np.random.default_rng(13))
    A.random_config()
    before = A.value.copy()
    A.moveforward(mu=0)
    assert_su2(A.value) and A.value.dtype == np.complex128
    A.movebackward(mu=0)
    assert np.abs(A.value - before).max() < 1e-12                                # forward then backward is the identity


def test_staple_plaquette_and_action_are_not_written_yet():
    A = field()
    for method in (A.Staple, A.Plaq, A.Action):
        with pytest.raises(NotImplementedError):
            method()


def parallel_hdf5():
    try:
        import h5py
        return bool(h5py.get_config().mpi)
    except ImportError:
        return False


needs_hdf5 = pytest.mark.skipif(not parallel_hdf5(), reason='h5py with parallel HDF5 is not installed')


@needs_hdf5
def test_a_checkpoint_of_the_links_keeps_double_precision(tmp_path):
    import h5py
    from pylattice.core.callbacks import Checkpointer
    from pylattice.core.hdf5io import load_into
    A = field(np.random.default_rng(14))
    A.random_config()
    A.metropolis_update(eps=0.2)
    path = Checkpointer(1, directory=str(tmp_path), prefix='su2').save(type('Model', (), {'field': A})(), 7)
    with h5py.File(path, 'r') as f:
        assert f['value'].dtype == np.complex128 and f.attrs['type'] == 'LatticeVectorComplexMatrix' and f.attrs['step'] == 7
    B = field()
    lines = []
    assert load_into(B, path, log=lines.append) == 7
    assert np.array_equal(B.value, A.value)                                      # bit for bit: nothing rounded to complex64
    assert_su2(B.value, 1e-14)                                                   # still unitary to double precision, not 1e-7
    assert not any('WARNING' in l for l in lines)
    single = A.value.astype(np.complex64).astype(np.complex128)
    assert np.abs(np.linalg.det(single) - 1).max() > 1e-8                        # what the old single-precision file would have given
