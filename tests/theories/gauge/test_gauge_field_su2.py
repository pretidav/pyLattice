"""SU(2) gauge field (pylattice.theories.gauge.field): the lattice of links and its updates. The properties of
the group itself (generators, exponential, quaternions, heat bath) are tested in tests/core/test_sun.py.
One process (a lattice on a 1x1 rank grid). Run with:  pytest tests/theories/gauge/test_gauge_field_su2.py
"""
import numpy as np
import pytest

from pylattice.core.groups import su2 as sun
from pylattice.core.parallel import CartesianComm
from pylattice.theories.gauge.field import SU2Field

GRID, ND = [4, 6], 2
SITES = 4 * 6
I2 = np.eye(2)


def dagger(M):
    return np.conj(np.swapaxes(M, -1, -2))


def assert_su2(U, tol=1e-12):
    assert np.abs(U @ dagger(U) - I2).max() < tol and np.abs(np.linalg.det(U) - 1).max() < tol


def field(rng=None, grid=GRID, nd=ND):
    return SU2Field(grid=grid, cartesiancomm=CartesianComm(mpigrid=[1] * len(grid)), Nd=nd, rng=rng)


def link_plaquette_trace(A, mu=0, nu=1):
    """Re tr of U_mu(x) U_nu(x+mu) U_mu(x+nu)^dagger U_nu(x)^dagger, built with the field's own shifts"""
    Umu, Unu = A.peek_index(mu), A.peek_index(nu)
    P = Umu * Unu.shift_forward(mu) * Umu.shift_forward(nu).adj() * Unu.adj()      # built with the lattice operations
    return P.trace().value.real


def test_gauge_transform_returns_a_new_su2_field_and_keeps_the_original():
    A = field(np.random.default_rng(3)); A.random_config()
    before = A.value.copy()
    B = A.gauge_transform()
    assert B is not A and isinstance(B, SU2Field) and np.array_equal(A.value, before)
    assert_su2(B.value) and not np.allclose(B.value, A.value)


def test_gauge_transform_follows_the_definition_on_every_link():
    A = field(np.random.default_rng(4)); A.random_config()
    g = A.random_gauge_transformation()
    B = A.gauge_transform(g)
    grid = np.array(GRID)
    for site in range(SITES):
        x = np.unravel_index(site, GRID)
        for mu in range(ND):
            y = list(x); y[mu] = (y[mu] + 1) % grid[mu]
            ysite = np.ravel_multi_index(y, GRID)
            expected = g[site] @ A.value[site, mu] @ dagger(g[ysite])
            assert np.abs(B.value[site, mu] - expected).max() < 1e-12


def test_gauge_transform_leaves_the_plaquette_invariant_but_not_the_links():
    A = field(np.random.default_rng(5)); A.random_config()
    B = A.gauge_transform()
    for mu, nu in ((0, 1), (1, 0)):
        assert np.abs(link_plaquette_trace(A, mu, nu) - link_plaquette_trace(B, mu, nu)).max() < 1e-12
    assert np.abs(np.trace(A.value, axis1=-2, axis2=-1) - np.trace(B.value, axis1=-2, axis2=-1)).max() > 1e-3


def test_gauge_transform_with_identity_or_a_constant_g():
    A = field(np.random.default_rng(6)); A.random_config()
    ident = np.broadcast_to(I2, (SITES, 2, 2))
    assert np.abs(A.gauge_transform(ident).value - A.value).max() < 1e-14
    h = sun.to_matrix(sun.random_quaternion(np.random.default_rng(7)))
    B = A.gauge_transform(np.broadcast_to(h, (SITES, 2, 2)))                     # global: a conjugation
    assert np.abs(B.value - h @ A.value @ dagger(h)).max() < 1e-12


def test_gauge_transform_inverse_and_composition_and_cold_field():
    A = field(np.random.default_rng(8)); A.random_config()
    g, h = A.random_gauge_transformation(), A.random_gauge_transformation()
    assert np.abs(A.gauge_transform(g).gauge_transform(dagger(g)).value - A.value).max() < 1e-12
    assert np.abs(A.gauge_transform(g).gauge_transform(h).value - A.gauge_transform(h @ g).value).max() < 1e-12
    C = field()                                                                  # cold: U' = g(x) g(x+mu)^dagger
    assert_su2(C.gauge_transform(g).value)


def test_gauge_transform_rejects_a_wrong_shape_and_is_reproducible_with_a_seed():
    A = field(np.random.default_rng(9)); A.random_config()
    with pytest.raises(ValueError):
        A.gauge_transform(np.zeros((SITES, 3, 3)))
    assert np.array_equal(A.gauge_transform(rng=np.random.default_rng(1)).value,
                          A.gauge_transform(rng=np.random.default_rng(1)).value)


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


def brute_plaquette(A, mu, nu, site):
    G = tuple(int(g) for g in A.grid)
    x = np.unravel_index(site, G)
    idx = lambda d, k=1: np.ravel_multi_index([(x[i] + (k if i == d else 0)) % G[i] for i in range(len(G))], G)
    U = A.value
    return U[site, mu] @ U[idx(mu), nu] @ dagger(U[idx(nu), mu]) @ dagger(U[site, nu])


def test_cold_field_has_plaquette_one_and_action_zero():
    A = field(nd=3, grid=[4, 4, 4])
    assert abs(A.Plaq() - 1) < 1e-14 and abs(A.Action(2.3)) < 1e-12 and isinstance(A.Plaq(), complex)
    assert np.abs(A.Staple(1).value - 4 * I2).max() < 1e-14                              # 2 (Nd-1) staples, each the identity


def test_plaquette_field_matches_a_site_by_site_product():
    A = field(np.random.default_rng(11)); A.random_config()
    P = A.Plaquette_field(0, 1).value
    for site in range(SITES):
        assert np.abs(P[site] - brute_plaquette(A, 0, 1, site)).max() < 1e-12
    assert_su2(P)


def test_plaq_and_action_follow_their_definition_in_3d():
    A = field(np.random.default_rng(12), grid=[3, 4, 5], nd=3); A.random_config()
    n = 3 * 4 * 5
    vals = [0.5 * np.trace(brute_plaquette(A, mu, nu, s)).real for mu in range(3) for nu in range(mu + 1, 3) for s in range(n)]
    assert abs(A.Plaq() - np.mean(vals)) < 1e-12 and len(vals) == 3 * n and abs(A.Plaq().imag) < 1e-12
    assert abs(A.Action(2.0) - 2.0 * (len(vals) - sum(vals))) < 1e-10
    assert -1 <= A.Plaq().real <= 1 and A.Action(2.0) >= 0


def test_staple_contracted_with_the_links_gives_four_times_the_plaquettes():
    A = field(np.random.default_rng(13), grid=[4, 4, 4], nd=3); A.random_config()
    link_sum = sum(np.trace(A.value[:, mu] @ A.Staple(mu).value, axis1=-2, axis2=-1).real.sum() for mu in range(3))
    plaq_sum = sum(np.trace(A.Plaquette_field(mu, nu).value, axis1=-2, axis2=-1).real.sum() for mu in range(3) for nu in range(mu + 1, 3))
    assert abs(link_sum - 4 * plaq_sum) < 1e-9


def test_staple_is_a_sum_of_su2_products_and_checks_the_direction():
    A = field(np.random.default_rng(14)); A.random_config()
    S = A.Staple(0).value                                                                 # Nd=2: 2 staples, each in SU(2)
    assert S.shape == (SITES, 2, 2) and np.linalg.norm(S, axis=(-2, -1)).max() <= 2 * np.sqrt(2) + 1e-12
    with pytest.raises(ValueError):
        A.Staple(2)
    with pytest.raises(ValueError):
        A.Plaquette_field(1, 1)
    with pytest.raises(ValueError):
        field(nd=1).Plaq()


def test_plaq_action_and_staple_energy_are_gauge_invariant():
    A = field(np.random.default_rng(15), grid=[4, 4, 4], nd=3); A.random_config()
    B = A.gauge_transform()
    assert abs(A.Plaq() - B.Plaq()) < 1e-12 and abs(A.Action(2.2) - B.Action(2.2)) < 1e-9
    for mu in range(3):                                                             # the staple itself is covariant: S' = g(x+mu...)
        ea = np.trace(A.value[:, mu] @ A.Staple(mu).value, axis1=-2, axis2=-1).real
        eb = np.trace(B.value[:, mu] @ B.Staple(mu).value, axis1=-2, axis2=-1).real
        assert np.abs(ea - eb).max() < 1e-12


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
