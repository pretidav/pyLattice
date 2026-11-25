"""SONField (pylattice.theories.gauge.field): the SO(N) gauge field, compared with brute-force numpy on the whole lattice, with the
same loops as SUNField, and with the known relations of SO(2) = U(1) and SO(3). One process (a 1x1... rank grid). Run with:
    pytest tests/theories/gauge/test_gauge_field_son.py
The same operations on a lattice split over ranks are tested in test_gauge_field_mpi.py.
"""
import numpy as np
import pytest

from pylattice.core.groups import so
from pylattice.core.parallel import CartesianComm
from pylattice.theories.gauge.field import SONField, SUNField, gauge_field, load_links

T = lambda M: np.swapaxes(M, -1, -2)


def field(N, G, seed=1, hot=True):
    cc = CartesianComm(mpigrid=[1] * len(G))
    f = SONField(grid=list(G), cartesiancomm=cc, Nd=len(G), N=N, rng=np.random.default_rng(seed))
    if hot:
        f.random_config()
    return f


def near_identity_field(N, G, seed=2, eps=0.5):
    """a field of links near the identity (positive loops), and the same links as an array U[x..., mu, N, N]"""
    cc = CartesianComm(mpigrid=[1] * len(G))
    U = so.near_identity_so(N, eps, np.random.default_rng(seed), size=tuple(G) + (len(G),))
    f = SONField(grid=list(G), cartesiancomm=cc, Nd=len(G), N=N)
    f.value = U.reshape(-1, len(G), N, N).copy()
    return f, U


def special_orthogonal(U, tol=1e-12):
    N = U.shape[-1]
    return np.abs(U @ T(U) - np.eye(N)).max() < tol and np.abs(np.linalg.det(U) - 1).max() < tol


def ref_line(U, mu, R):
    out = moved = U[..., mu, :, :]
    for _ in range(R - 1):
        moved = np.roll(moved, -1, axis=mu)
        out = out @ moved
    return out


def ref_loop(U, mu, nu, R, T_):
    a, b = ref_line(U, mu, R), ref_line(U, nu, T_)
    return a @ np.roll(b, -R, axis=mu) @ T(np.roll(a, -T_, axis=nu)) @ T(b)


def ref_average(U, R, T_):
    Nd, N = U.ndim - 3, U.shape[-1]
    pairs = [(m, n) for m in range(Nd) for n in range(Nd) if m != n]
    return np.mean([np.trace(ref_loop(U, m, n, R, T_), axis1=-2, axis2=-1) / N for m, n in pairs])


# ----------------------------------------------------------------------------- the field
@pytest.mark.parametrize('N', [2, 3, 4])
def test_shape_dtype_cold_start_and_validation(N):
    f = field(N, [4, 4], hot=False)
    assert f.value.shape == (16, 2, N, N) and f.value.dtype == np.float64
    assert (f.N, f.Nd, f.adj_dim) == (N, 2, N * (N - 1) // 2)
    assert np.array_equal(f.value, np.broadcast_to(np.eye(N), f.value.shape)) and np.array_equal(f.trivial_config(), f.value)
    assert f.Plaq() == 1 and abs(f.Action(2.0)) < 1e-12 and abs(f.Wilson(2, 2) - 1) < 1e-14 and abs(f.Polyakov(1) - 1) < 1e-14
    for bad in (1, 0, 2.5):
        with pytest.raises(ValueError, match='at least 2'):
            SONField(grid=[4, 4], cartesiancomm=CartesianComm(mpigrid=[1, 1]), Nd=2, N=bad)


@pytest.mark.parametrize('N', [2, 3, 4])
def test_hot_start_is_so_n_reproducible_and_haar_distributed(N):
    a, b, c = field(N, [8, 8], seed=3), field(N, [8, 8], seed=3), field(N, [8, 8], seed=4)
    assert special_orthogonal(a.value) and np.array_equal(a.value, b.value) and not np.array_equal(a.value, c.value)
    tr = np.trace(a.value, axis1=-2, axis2=-1)                                           # 128 links
    assert abs(tr.mean()) < 0.6 and abs(np.mean(tr ** 2) - (2 if N == 2 else 1)) < 0.8   # <tr U> = 0 and <tr^2> = 1 (2 for SO(2)), crudely
    a.random_config(np.random.default_rng(9))
    assert special_orthogonal(a.value) and not np.array_equal(a.value, c.value)          # an rng can be given


@pytest.mark.parametrize('N', [2, 3, 4])
def test_unitarize_removes_the_rounding_and_leaves_so_n_unchanged(N):
    f = field(N, [4, 4])
    before = f.value.copy()
    f.unitarize()
    assert np.abs(f.value - before).max() < 1e-12
    f.value = before * (1 + 1e-6) + 1e-6 * np.random.default_rng(5).standard_normal(before.shape)
    assert not special_orthogonal(f.value, 1e-9)
    f.unitarize()
    assert special_orthogonal(f.value) and np.abs(f.value - before).max() < 1e-4


@pytest.mark.parametrize('N', [2, 3, 4])
def test_the_bare_metropolis_proposal_stays_in_so_n_and_moves_the_links(N):
    f = field(N, [4, 4], seed=6)
    before = f.value.copy()
    f.metropolis_update(eps=0.2)
    assert special_orthogonal(f.value) and 0 < np.abs(f.value - before).max() < 0.8
    f.value = before.copy()
    f.metropolis_update(eps=0.0)
    assert np.abs(f.value - before).max() < 1e-12
    near = f.near_identity(0.1)
    assert near.shape == f.value.shape and special_orthogonal(near) and np.abs(near - np.eye(N)).max() < 0.4
    assert f.near_identity(0.1, size=7).shape == (7, N, N)


# ----------------------------------------------------------------------------- plaquette, staples, action, loops
@pytest.mark.parametrize('N', [2, 3, 4])
def test_plaquette_action_and_staples_equal_a_brute_force_computation(N):
    G = [4, 4, 4]
    f = field(N, G, seed=7)
    U = f.value.reshape(tuple(G) + (3, N, N))
    P = {(m, n): ref_loop(U, m, n, 1, 1) for m in range(3) for n in range(3) if m != n}
    plaq = np.mean([np.trace(P[k], axis1=-2, axis2=-1) / N for k in P])
    assert abs(f.Plaq() - plaq) < 1e-12 and f.Plaq().imag == 0                            # real links: exactly real
    beta = 1.7
    action = beta * sum((1 - np.trace(P[m, n], axis1=-2, axis2=-1) / N).sum() for m in range(3) for n in range(m + 1, 3))
    assert abs(f.Action(beta) - action) < 1e-9
    link_sum = sum(np.trace(f.value[:, mu] @ f.Staple(mu).value, axis1=-2, axis2=-1).sum() for mu in range(3))
    assert abs(link_sum - 4 * sum(np.trace(P[m, n], axis1=-2, axis2=-1).sum() for m in range(3) for n in range(m + 1, 3))) < 1e-9
    assert np.abs(f.Plaquette_field(0, 1).value.reshape(tuple(G) + (N, N)) - P[0, 1]).max() < 1e-12
    assert f.Staple(1).value.dtype == np.float64
    with pytest.raises(ValueError):
        f.Staple(3)
    with pytest.raises(ValueError):
        f.Plaquette_field(1, 1)


@pytest.mark.parametrize('N', [2, 3, 4])
def test_wilson_loops_and_the_polyakov_loop_equal_a_brute_force_computation(N):
    G = [4, 4, 4]
    f, U = near_identity_field(N, G)
    for R, T_ in ((1, 1), (1, 2), (2, 2), (2, 3), (3, 3)):
        assert abs(f.Wilson(R, T_) - ref_average(U, R, T_)) < 1e-12, (R, T_)
        assert f.Wilson(R, T_).imag == 0                                                  # real links: a loop and its reverse agree
    t = 2
    assert abs(f.Polyakov(t) - (np.trace(ref_line(U, t, G[t]), axis1=-2, axis2=-1) / N).mean()) < 1e-12
    assert abs(f.Polyakov(t)) > 1e-6 and f.Polyakov(t).imag == 0


def test_a_field_of_so_n_links_that_are_also_special_unitary_gives_the_same_numbers_as_the_sun_field():
    # a real orthogonal matrix with det 1 is in SU(N): the same links in a SUNField (complex) give the same observables
    G = [4, 4, 4]
    for N in (2, 3):
        f = field(N, G, seed=12)
        g = SUNField(grid=G, cartesiancomm=CartesianComm(mpigrid=[1, 1, 1]), Nd=3, N=N)
        g.value = f.value.astype(complex)
        for key in (lambda x: x.Plaq(), lambda x: x.Action(2.3), lambda x: x.Wilson(1, 2), lambda x: x.Wilson(2, 2),
                    lambda x: x.Polyakov(2), lambda x: x.Staple(1).value.sum()):
            assert abs(key(f) - key(g)) < 1e-12


def test_so2_in_two_dimensions_is_a_u1_theory_with_plaquette_cos_of_the_angle_sum():
    # SO(2) links are rotations by angles a_mu(x); the plaquette is the rotation by the lattice curl and (1/2) tr P = cos(curl)
    G = [4, 4]
    cc = CartesianComm(mpigrid=[1, 1])
    f = SONField(grid=G, cartesiancomm=cc, Nd=2, N=2)
    a = np.random.default_rng(13).uniform(-np.pi, np.pi, size=tuple(G) + (2,))
    f.value = so.embed_so2(a, 0, 1, 2).reshape(-1, 2, 2, 2)
    curl = a[..., 0] + np.roll(a[..., 1], -1, axis=0) - np.roll(a[..., 0], -1, axis=1) - a[..., 1]
    assert abs(f.Plaq() - np.cos(curl).mean()) < 1e-12 and abs(f.Action(1.0) - (1 - np.cos(curl)).sum()) < 1e-10
    # a gauge transformation by the angles chi(x): a_mu(x) -> a_mu(x) + chi(x) - chi(x + mu)
    chi = np.random.default_rng(14).uniform(-np.pi, np.pi, size=tuple(G))
    h = f.gauge_transform(so.embed_so2(chi, 0, 1, 2).reshape(-1, 2, 2))
    new = a + chi[..., None] - np.stack([np.roll(chi, -1, axis=0), np.roll(chi, -1, axis=1)], axis=-1)
    assert np.abs(h.value - so.embed_so2(new, 0, 1, 2).reshape(-1, 2, 2, 2)).max() < 1e-12


# ----------------------------------------------------------------------------- gauge transformations
@pytest.mark.parametrize('N', [2, 3, 4])
def test_gauge_transformations_keep_the_type_follow_the_formula_and_leave_the_observables_invariant(N):
    G = [4, 4, 4]
    f, U = near_identity_field(N, G, seed=8)
    before = f.value.copy()
    g = f.random_gauge_transformation(np.random.default_rng(3))
    assert g.shape == (64, N, N) and special_orthogonal(g)
    h = f.gauge_transform(g)
    assert type(h) is SONField and h is not f and h.N == N and np.array_equal(f.value, before) and h.value.dtype == np.float64
    gg = g.reshape(tuple(G) + (N, N))
    for mu in range(3):
        want = gg @ U[..., mu, :, :] @ T(np.roll(gg, -1, axis=mu))
        assert np.abs(h.value[:, mu].reshape(tuple(G) + (N, N)) - want).max() < 1e-12
    assert special_orthogonal(h.value)
    for key in (lambda x: x.Plaq(), lambda x: x.Action(2.0), lambda x: x.Wilson(2, 2), lambda x: x.Wilson(1, 3),
                lambda x: x.Polyakov(2), lambda x: x.Polyakov(0)):
        assert abs(key(f) - key(h)) < 1e-11
    assert np.abs(h.gauge_transform(T(g)).value - before).max() < 1e-12                       # the inverse transformation
    assert not np.allclose(h.value, before) and special_orthogonal(f.gauge_transform().value)
    with pytest.raises(ValueError, match='shape'):
        f.gauge_transform(np.zeros((64, N + 1, N + 1)))


# ----------------------------------------------------------------------------- the factory
def test_the_factory_picks_the_field_of_the_group():
    cc = CartesianComm(mpigrid=[1, 1])
    assert type(gauge_field([4, 4], cc, 2, 3, group='so')) is SONField and type(gauge_field([4, 4], cc, 2, 3)) is SUNField
    assert type(gauge_field([4, 4], cc, 2, 3, group='su')) is SUNField and gauge_field([4, 4], cc, 2, 2, group='so').value.dtype == np.float64
    with pytest.raises(ValueError, match="'su' or 'so'"):
        gauge_field([4, 4], cc, 2, 3, group='u')


# ----------------------------------------------------------------------------- checkpoints
def parallel_hdf5():
    try:
        import h5py
        return bool(h5py.get_config().mpi)
    except ImportError:
        return False


@pytest.mark.skipif(not parallel_hdf5(), reason='h5py with parallel HDF5 is not installed')
def test_a_checkpoint_of_an_so3_field_keeps_double_precision_loads_back_and_refuses_other_matrices(tmp_path):
    import h5py
    from pylattice.core.callbacks import Checkpointer
    f = field(3, [4, 4], seed=11)
    path = Checkpointer(1, directory=str(tmp_path), prefix='so3').save(type('Model', (), {'field': f})(), 3)
    with h5py.File(path, 'r') as h:
        assert h['value'].dtype == np.float64 and h['value'].shape == (16, 2, 3, 3) and h.attrs['N'] == 3 and h.attrs['step'] == 3
    g = field(3, [4, 4], hot=False)
    assert load_links(g, path) == 3 and np.array_equal(g.value, f.value)
    with pytest.raises(ValueError):                                                         # an SO(4) field does not fit
        load_links(field(4, [4, 4], hot=False), path)
    bad = field(3, [4, 4], hot=False)
    bad.value = bad.value * 1.1                                                             # real, but not orthogonal
    path = Checkpointer(1, directory=str(tmp_path), prefix='bad').save(type('Model', (), {'field': bad})(), 1)
    ok = field(3, [4, 4], hot=False)
    with pytest.raises(ValueError, match='SO.3. matrices'):
        load_links(ok, path)
    assert np.array_equal(ok.value, np.broadcast_to(np.eye(3), ok.value.shape))             # the field is left as it was
