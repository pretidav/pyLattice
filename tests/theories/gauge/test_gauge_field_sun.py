"""SUNField (pylattice.theories.gauge.field): the SU(N) gauge field for N = 3, 4, ..., compared with brute-force numpy on the
whole lattice, and with SU2Field for N = 2. One process (a lattice on a 1x1... rank grid). Run with:
    pytest tests/theories/gauge/test_gauge_field_sun.py
The same operations on a lattice split over ranks are tested for the SU(2) case in test_gauge_field_mpi.py (the code does not
depend on N).
"""
import numpy as np
import pytest

from pylattice.core.groups import su
from pylattice.core.parallel import CartesianComm
from pylattice.theories.gauge.field import SU2Field, SUNField

dag = lambda M: np.conj(np.swapaxes(M, -1, -2))


def field(N, G, seed=1, hot=True):
    cc = CartesianComm(mpigrid=[1] * len(G))
    f = SUNField(grid=list(G), cartesiancomm=cc, Nd=len(G), N=N, rng=np.random.default_rng(seed))
    if hot:
        f.random_config()
    return f


def near_identity_field(N, G, seed=2, eps=0.5):
    """a field of links near the identity (positive loops), and the same links as an array U[x..., mu, N, N]"""
    cc = CartesianComm(mpigrid=[1] * len(G))
    U = su.near_identity_su(N, eps, np.random.default_rng(seed), size=tuple(G) + (len(G),))
    f = SUNField(grid=list(G), cartesiancomm=cc, Nd=len(G), N=N)
    f.value = U.reshape(-1, len(G), N, N).copy()
    return f, U


def special_unitary(U, tol=1e-12):
    N = U.shape[-1]
    return np.abs(U @ dag(U) - np.eye(N)).max() < tol and np.abs(np.linalg.det(U) - 1).max() < tol


def ref_line(U, mu, R):
    out = moved = U[..., mu, :, :]
    for _ in range(R - 1):
        moved = np.roll(moved, -1, axis=mu)
        out = out @ moved
    return out


def ref_loop(U, mu, nu, R, T):
    a, b = ref_line(U, mu, R), ref_line(U, nu, T)
    return a @ np.roll(b, -R, axis=mu) @ dag(np.roll(a, -T, axis=nu)) @ dag(b)


def ref_average(U, R, T, pairs=None):
    Nd, N = U.ndim - 3, U.shape[-1]
    pairs = [(m, n) for m in range(Nd) for n in range(Nd) if m != n] if pairs is None else pairs
    return np.mean([np.trace(ref_loop(U, m, n, R, T), axis1=-2, axis2=-1) / N for m, n in pairs])


# ----------------------------------------------------------------------------- the field
@pytest.mark.parametrize('N', [3, 4])
def test_shape_dtype_cold_start_and_the_relation_to_su2field(N):
    f = field(N, [4, 4], hot=False)
    assert f.value.shape == (16, 2, N, N) and f.value.dtype == np.complex128
    assert (f.N, f.Nd, f.adj_dim) == (N, 2, N * N - 1)
    assert np.array_equal(f.value, np.broadcast_to(np.eye(N), f.value.shape)) and np.array_equal(f.trivial_config(), f.value)
    assert abs(f.Plaq() - 1) < 1e-14 and abs(f.Action(2.0)) < 1e-12 and abs(f.Wilson(2, 2) - 1) < 1e-14 and abs(f.Polyakov(1) - 1) < 1e-14
    two = SU2Field(grid=[4, 4], cartesiancomm=CartesianComm(mpigrid=[1, 1]), Nd=2)
    assert isinstance(two, SUNField) and two.N == 2 and two.value.shape == (16, 2, 2, 2)
    for bad in (1, 0, 2.5):
        with pytest.raises(ValueError, match='at least 2'):
            SUNField(grid=[4, 4], cartesiancomm=CartesianComm(mpigrid=[1, 1]), Nd=2, N=bad)


@pytest.mark.parametrize('N', [3, 4])
def test_hot_start_is_su_n_reproducible_and_haar_distributed(N):
    a, b, c = field(N, [8, 8], seed=3), field(N, [8, 8], seed=3), field(N, [8, 8], seed=4)
    assert special_unitary(a.value) and np.array_equal(a.value, b.value) and not np.array_equal(a.value, c.value)
    tr = np.trace(a.value, axis1=-2, axis2=-1)                                           # 128 links
    assert abs(np.mean(np.abs(tr) ** 2) - 1) < 0.5 and abs(tr.mean()) < 0.5               # <|tr U|^2> = 1 and <tr U> = 0, crudely
    a.random_config(np.random.default_rng(9))
    assert special_unitary(a.value) and not np.array_equal(a.value, c.value)              # an rng can be given


@pytest.mark.parametrize('N', [3, 4])
def test_unitarize_removes_the_rounding_and_leaves_su_n_unchanged(N):
    f = field(N, [4, 4])
    before = f.value.copy()
    f.unitarize()
    assert np.abs(f.value - before).max() < 1e-12
    f.value = before * (1 + 1e-6) + 1e-6 * np.random.default_rng(5).standard_normal(before.shape)
    assert not special_unitary(f.value, 1e-9)
    f.unitarize()
    assert special_unitary(f.value) and np.abs(f.value - before).max() < 1e-4


@pytest.mark.parametrize('N', [3, 4])
def test_the_bare_metropolis_proposal_stays_in_su_n_and_moves_the_links(N):
    f = field(N, [4, 4], seed=6)
    before = f.value.copy()
    f.metropolis_update(eps=0.2)
    assert special_unitary(f.value) and 0 < np.abs(f.value - before).max() < 0.8
    f.value = before.copy()
    f.metropolis_update(eps=0.0)                                                          # no step: no change
    assert np.abs(f.value - before).max() < 1e-12
    near = f.near_identity(0.1)
    assert near.shape == f.value.shape and special_unitary(near) and np.abs(near - np.eye(N)).max() < 0.4


# ----------------------------------------------------------------------------- plaquette, staples, action
@pytest.mark.parametrize('N', [3, 4])
def test_plaquette_action_and_staples_equal_a_brute_force_computation(N):
    G = [4, 4, 4]
    f = field(N, G, seed=7)
    U = f.value.reshape(tuple(G) + (3, N, N))
    P = {(m, n): ref_loop(U, m, n, 1, 1) for m in range(3) for n in range(3) if m != n}
    plaq = np.mean([np.trace(P[k], axis1=-2, axis2=-1) / N for k in P])
    assert abs(f.Plaq() - plaq) < 1e-12 and abs(f.Plaq().imag) < 1e-12                    # P(nu,mu) = P(mu,nu)^+: real
    beta = 1.7
    action = beta * sum((1 - np.trace(P[m, n], axis1=-2, axis2=-1).real / N).sum() for m in range(3) for n in range(m + 1, 3))
    assert abs(f.Action(beta) - action) < 1e-9
    link_sum = sum(np.trace(f.value[:, mu] @ f.Staple(mu).value, axis1=-2, axis2=-1).real.sum() for mu in range(3))
    assert abs(link_sum - 4 * sum(np.trace(P[m, n], axis1=-2, axis2=-1).real.sum() for m in range(3) for n in range(m + 1, 3))) < 1e-9
    one = f.Plaquette_field(0, 1)
    assert np.abs(one.value.reshape(tuple(G) + (N, N)) - P[0, 1]).max() < 1e-12            # P_{mu nu}(x) site by site
    with pytest.raises(ValueError):
        f.Staple(3)
    with pytest.raises(ValueError):
        f.Plaquette_field(1, 1)


@pytest.mark.parametrize('N', [3, 4])
def test_wilson_loops_and_the_polyakov_loop_equal_a_brute_force_computation(N):
    G = [4, 4, 4]
    f, U = near_identity_field(N, G)
    for R, T in ((1, 1), (1, 2), (2, 2), (2, 3), (3, 3)):
        assert abs(f.Wilson(R, T) - ref_average(U, R, T)) < 1e-12, (R, T)
        if R == T or N == 2:
            assert abs(f.Wilson(R, T).imag) < 1e-12                                          # a loop and its reverse: real
    t = 2
    polyakov = (np.trace(ref_line(U, t, G[t]), axis1=-2, axis2=-1) / N).mean()
    assert abs(f.Polyakov(t) - polyakov) < 1e-12
    pairs = [(m, n) for m in range(3) for n in range(3) if m != n and t in (m, n)]
    assert abs(f.Plaq(pairs) - ref_average(U, 1, 1, pairs)) < 1e-12                           # the plaquette of some planes only
    assert abs(f.Polyakov(t).imag) > 1e-6                                                     # complex for N >= 3 (not zero against zero)
    assert abs(f.Wilson(1, 2).imag) > 1e-4                                                    # and so is a loop with R != T: only Re is physical


# ----------------------------------------------------------------------------- gauge transformations, the center
@pytest.mark.parametrize('N', [3, 4])
def test_gauge_transformations_keep_the_type_follow_the_formula_and_leave_the_observables_invariant(N):
    G = [4, 4, 4]
    f, U = near_identity_field(N, G, seed=8)
    before = f.value.copy()
    g = f.random_gauge_transformation(np.random.default_rng(3))
    assert g.shape == (64, N, N) and special_unitary(g)
    h = f.gauge_transform(g)
    assert type(h) is SUNField and h is not f and h.N == N and np.array_equal(f.value, before)
    gg = g.reshape(tuple(G) + (N, N))
    for mu in range(3):
        want = gg @ U[..., mu, :, :] @ dag(np.roll(gg, -1, axis=mu))
        assert np.abs(h.value[:, mu].reshape(tuple(G) + (N, N)) - want).max() < 1e-12
    assert special_unitary(h.value)
    for key in (lambda x: x.Plaq(), lambda x: x.Action(2.0), lambda x: x.Wilson(2, 2), lambda x: x.Wilson(1, 3),
                lambda x: x.Polyakov(2), lambda x: x.Polyakov(0)):
        assert abs(key(f) - key(h)) < 1e-11
    back = h.gauge_transform(dag(g))
    assert np.abs(back.value - before).max() < 1e-12                                          # the inverse transformation
    assert not np.allclose(h.value, before) and special_unitary(f.gauge_transform().value)    # g=None: a random one
    with pytest.raises(ValueError, match='shape'):
        f.gauge_transform(np.zeros((64, N + 1, N + 1)))


def test_the_center_of_su_3_leaves_the_action_alone_and_rotates_the_polyakov_loop():
    # multiplying the links of the time direction on one time slice by a center element z = exp(2 pi i / 3) changes no plaquette
    # (each contains the link and its inverse) and multiplies the Polyakov loop by z: the Z_3 symmetry of pure SU(3)
    G, t = [4, 4, 4], 2
    f, U = near_identity_field(3, G, seed=9)
    z = np.exp(2j * np.pi / 3)
    before = (f.Plaq(), f.Action(2.3), f.Wilson(2, 2), f.Polyakov(t), f.Polyakov(0))
    slice0 = np.indices(G)[t].reshape(-1) == 1
    f.value[slice0, t] *= z
    assert special_unitary(f.value)
    assert abs(f.Plaq() - before[0]) < 1e-12 and abs(f.Action(2.3) - before[1]) < 1e-9 and abs(f.Wilson(2, 2) - before[2]) < 1e-12
    assert abs(f.Polyakov(0) - before[4]) < 1e-12                                              # the other directions: unchanged
    assert abs(f.Polyakov(t) - z * before[3]) < 1e-12 and abs(f.Polyakov(t) - before[3]) > 1e-3


# ----------------------------------------------------------------------------- N = 2: the same as SU2Field
def test_for_n_equal_2_the_general_field_and_the_su2_field_agree():
    G = [4, 4, 4]
    cc = CartesianComm(mpigrid=[1, 1, 1])
    two = SU2Field(grid=G, cartesiancomm=cc, Nd=3, rng=np.random.default_rng(1))
    two.random_config()
    general = SUNField(grid=G, cartesiancomm=cc, Nd=3, N=2)
    general.value = two.value.copy()
    for key in (lambda x: x.Plaq(), lambda x: x.Action(2.3), lambda x: x.Wilson(1, 2), lambda x: x.Wilson(2, 2),
                lambda x: x.Polyakov(2), lambda x: x.Staple(1).value.sum()):
        assert abs(key(two) - key(general)) < 1e-12
    assert np.abs(two.gauge_transform(two.random_gauge_transformation(np.random.default_rng(2))).value
                  - general.gauge_transform(two.random_gauge_transformation(np.random.default_rng(2))).value).max() < 1e-12
    assert type(two.gauge_transform()) is SU2Field                                           # the subclass is kept
    general.random_config()
    assert special_unitary(general.value)                                                    # the general hot start works for N = 2 too


# ----------------------------------------------------------------------------- checkpoints
def parallel_hdf5():
    try:
        import h5py
        return bool(h5py.get_config().mpi)
    except ImportError:
        return False


@pytest.mark.skipif(not parallel_hdf5(), reason='h5py with parallel HDF5 is not installed')
def test_a_checkpoint_of_an_su3_field_keeps_double_precision_and_loads_back(tmp_path):
    import h5py
    from pylattice.core.callbacks import Checkpointer
    from pylattice.core.hdf5io import load_into
    f = field(3, [4, 4], seed=11)
    path = Checkpointer(1, directory=str(tmp_path), prefix='su3').save(type('Model', (), {'field': f})(), 3)
    with h5py.File(path, 'r') as h:
        assert h['value'].dtype == np.complex128 and h['value'].shape == (16, 2, 3, 3) and h.attrs['N'] == 3 and h.attrs['step'] == 3
    g = field(3, [4, 4], hot=False)
    assert load_into(g, path) == 3 and np.array_equal(g.value, f.value)
    with pytest.raises(ValueError):                                                         # an SU(4) field does not fit
        load_into(field(4, [4, 4], hot=False), path)
