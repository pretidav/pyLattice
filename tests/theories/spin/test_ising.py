"""Tests of pylattice.theories.spin.ising: IsingField, IsingModel (observables, neighbours, local
energy, Metropolis test, one sweep, run_mc, plots) and the command line driver.

Every check is made on the whole lattice (gathered over the ranks and compared with NumPy), so the
same tests run on one process or on a lattice split over 2 or 4 ranks:

    pytest tests/theories/spin/test_ising.py
    mpiexec -n 2 python tests/theories/spin/test_ising.py        # 2 ranks (1 or 4 also work)

The model has H = -sum_<ij> s_i s_j with periodic boundaries; the sampling is checked against the
exact averages of the 4x4 lattice, which has only 2^16 states.
"""
import contextlib
import io
import logging
import os
import tempfile

import numpy as np
import pytest
from mpi4py import MPI

from pylattice.core.grid import LatticeReal, LatticeVectorReal
from pylattice.core.parallel import CartesianComm
from pylattice.theories.spin.ising import IsingField, IsingModel, main


# ----------------------------------------------------------------------------- helpers
def mpigrid_for(dims):
    """rank grid for the number of processes of this run: 1, 2 or 4"""
    size = MPI.COMM_WORLD.Get_size()
    assert size in (1, 2, 4), 'run with 1, 2 or 4 ranks'
    return {1: [1] * dims, 2: [2] + [1] * (dims - 1), 4: [2, 2] + [1] * (dims - 2)}[size]


def setup(G):
    """communicator and local grid for a global lattice G (every local size must be even)"""
    mg = mpigrid_for(len(G))
    return CartesianComm(mpigrid=mg), [g // m for g, m in zip(G, mg)]


def block(cc, loc):
    return tuple(slice(c * g, (c + 1) * g) for c, g in zip(cc.mpicoord, loc))


def to_global(cc, loc, values):
    """flat local values of every rank -> the global array, the same on all ranks"""
    values = np.asarray(values)
    out = np.zeros([g * m for g, m in zip(loc, cc.mpigrid)], dtype=values.dtype)
    for b, v in cc.comm.allgather((block(cc, loc), values.reshape(loc))):
        out[b] = v
    return out


def random_config(G, seed):
    return np.random.default_rng(seed).choice([-1.0, 1.0], size=G)


def make(glob, beta=0.5):
    """field holding the global configuration glob (each rank its block), and a model on it"""
    cc, loc = setup(list(glob.shape))
    field = IsingField(grid=loc, cartesiancomm=cc)
    field.value[:] = glob[block(cc, loc)].reshape(-1)
    return cc, loc, field, IsingModel(field=field, logger=None, beta=beta)


def checkerboard(G):
    return 1.0 - 2.0 * (np.indices(G).sum(axis=0) % 2)          # +1 on the even sites, -1 on the odd ones


def nn_sum(s):
    return sum(np.roll(s, r, axis=mu) for mu in range(s.ndim) for r in (1, -1))


def energy_per_site(s):
    return -sum((s * np.roll(s, 1, axis=mu)).sum() for mu in range(s.ndim)) / s.size


@contextlib.contextmanager
def in_tempdir():
    old = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        try:
            yield d
        finally:
            os.chdir(old)


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
        yield out


# ----------------------------------------------------------------------------- IsingField
def test_field_initializations():
    cc, loc = setup([8, 8])
    n = int(np.prod(loc))
    for init, value in (('hot', 1.0), ('cold', -1.0)):             # both are ordered starts
        f = IsingField(grid=loc, cartesiancomm=cc, initialization=init)
        assert isinstance(f, LatticeReal) and f.value.dtype == np.float32 and f.value.shape == (n,)
        assert (f.value == value).all()
    assert (IsingField(grid=loc, cartesiancomm=cc).value == -1).all()                       # default: cold
    a = IsingField(grid=loc, cartesiancomm=cc, initialization='random', seed=1)
    b = IsingField(grid=loc, cartesiancomm=cc, initialization='random', seed=1)
    c = IsingField(grid=loc, cartesiancomm=cc, initialization='random', seed=2)
    assert set(np.unique(a.value)) <= {-1.0, 1.0} and a.value.dtype == np.float32
    assert (a.value == b.value).all() and not (a.value == c.value).all()                   # the seed decides
    assert a.initialization == 'random' and a.seed == 1


# ----------------------------------------------------------------------------- observables
def test_magnetization_is_the_mean_spin():
    for glob in (np.ones([4, 4]), -np.ones([4, 4]), random_config([4, 4], 3), checkerboard([4, 4])):
        _, _, _, model = make(glob)
        assert abs(float(model.magnetization()) - glob.mean()) < 1e-6


def test_energy_per_site():
    for G in ([4, 4], [4, 4, 4]):
        d = len(G)
        assert abs(float(make(np.ones(G))[3].energy()) + d) < 1e-6                         # all aligned: -d
        assert abs(float(make(checkerboard(G))[3].energy()) - d) < 1e-6                    # all anti-aligned: +d
        for seed in (1, 2, 3):
            glob = random_config(G, seed)
            assert abs(float(make(glob)[3].energy()) - energy_per_site(glob)) < 1e-6


# ----------------------------------------------------------------------------- neighbours
def test_nn_field_holds_the_neighbours():
    # component mu: the value at x - mu; component mu + d: the value at x + mu
    for G in ([4, 4], [4, 4, 4]):
        d = len(G)
        glob = random_config(G, 5)
        cc, loc, _, model = make(glob)
        assert isinstance(model.nn_field, LatticeVectorReal) and model.nn_field.Nd == 2 * d
        for mu in range(d):
            back = to_global(cc, loc, model.nn_field.peek_index(mu).value)
            fwd = to_global(cc, loc, model.nn_field.peek_index(mu + d).value)
            assert (back == np.roll(glob, 1, axis=mu)).all()
            assert (fwd == np.roll(glob, -1, axis=mu)).all()


def test_local_energy_is_spin_times_its_neighbours():
    for G in ([4, 4], [4, 4, 4]):
        glob = random_config(G, 7)
        cc, loc, field, model = make(glob)
        got = to_global(cc, loc, model.local_energy(field).value)
        assert np.allclose(got, glob * nn_sum(glob))
    # it reads the neighbours from the cache nn_field: it is stale until get_nn_field() is called
    glob = random_config([4, 4], 8)
    cc, loc, field, model = make(glob)
    new = -glob
    field.value[:] = new[block(cc, loc)].reshape(-1)
    assert np.allclose(to_global(cc, loc, model.local_energy(field).value), new * nn_sum(glob))
    model.nn_field = model.get_nn_field()
    assert np.allclose(to_global(cc, loc, model.local_energy(field).value), new * nn_sum(new))


# ----------------------------------------------------------------------------- Metropolis test
class Delta:
    def __init__(self, v):
        self.value = np.asarray(v, dtype='float32')


def test_metropolis_acceptance_probability():
    np.random.seed(11)
    n = 200000
    _, _, _, model = make(np.ones([4, 4]))
    for beta, dE in ((0.5, 2.0), (0.25, 4.0), (1.0, 1.0), (0.3, 2.0), (0.7, 8.0)):
        model.beta = beta
        acc = model.metropolis_test(Delta(np.full(n, dE)))
        p = np.exp(-beta * dE)
        assert acc.dtype == bool and acc.shape == (n,)
        assert abs(acc.mean() - p) < 5 * np.sqrt(p * (1 - p) / n), (beta, dE, acc.mean(), p)
    model.beta = 0.5
    assert model.metropolis_test(Delta(np.full(1000, -8.0))).all()                         # lowering the energy
    assert model.metropolis_test(Delta(np.zeros(1000))).all()                              # or keeping it
    model.beta = 0.0
    assert model.metropolis_test(Delta(np.full(1000, 8.0))).all()                          # infinite temperature
    model.beta = 5.0
    assert not model.metropolis_test(Delta(np.full(100000, 8.0))).any()                    # exp(-40)


# ----------------------------------------------------------------------------- one sweep
def test_global_update_extreme_cases():
    G = [4, 4]
    # beta = 0: every flip is accepted, so the whole configuration is reversed
    glob = random_config(G, 4)
    cc, loc, field, model = make(glob, beta=0.0)
    acc = model.global_update()
    assert acc == field.value.size
    assert np.array_equal(to_global(cc, loc, field.value), -glob)
    # beta large: the ground state is frozen
    cc, loc, field, model = make(np.ones(G), beta=5.0)
    assert model.global_update() == 0 and (field.value == 1).all()
    # the count is the number of spins that changed, each spin being tried once
    glob = random_config(G, 6)
    cc, loc, field, model = make(glob, beta=0.4)
    before = field.value.copy()
    acc = model.global_update()
    assert acc == int((field.value != before).sum())
    assert set(np.unique(field.value)) <= {-1.0, 1.0}


def test_global_update_odd_sites_see_the_updated_even_sites():
    # anti-aligned start (even +1, odd -1): all the even spins flip (dE = -8), which leaves everything
    # aligned at -1, and then no odd spin flips (dE = +8). With the neighbours of before the even
    # update, the odd spins would flip too and the antiferromagnet would come back.
    cc, loc, field, model = make(checkerboard([4, 4]), beta=5.0)
    acc = model.global_update()
    assert acc == field.value.size // 2
    assert (field.value == -1).all()
    # the cache is up to date for the next sweep
    back = to_global(cc, loc, model.nn_field.peek_index(0).value)
    assert (back == -1).all() and (model.get_nn_field().peek_index(0).value == back[block(cc, loc)].reshape(-1)).all()


# ----------------------------------------------------------------------------- sampling
def exact_4x4(beta, L=4):
    st = ((np.arange(2 ** (L * L))[:, None] >> np.arange(L * L)) & 1) * 2 - 1
    s = st.reshape(-1, L, L)
    bond = (s * np.roll(s, 1, 1) + s * np.roll(s, 1, 2)).sum(axis=(1, 2))
    w = np.exp(beta * bond)
    Z = w.sum()
    m = st.mean(1)
    return -(w * bond).sum() / Z / L ** 2, (w * np.abs(m)).sum() / Z, (w * m ** 2).sum() / Z


def test_sampling_matches_the_exact_4x4_averages():
    # energy per site, |m| and m^2 from run_mc against the sum over all the 65536 states
    for beta in (0.3, 0.6):
        cc, loc = setup([4, 4])
        field = IsingField(grid=loc, cartesiancomm=cc, initialization='random', seed=cc.parallel_RNG(seed=11))
        model = IsingModel(field=field, logger=None, beta=beta)
        with quiet():
            model.run_mc(steps=3300, online_obs=True, progress=False)
        e, m = np.array(model.elist[300:], float), np.array(model.mlist[300:], float)
        obs = np.stack([e, np.abs(m), m ** 2], axis=1)
        mean = obs.mean(0)
        err = obs[:len(obs) // 20 * 20].reshape(20, -1, 3).mean(1).std(0, ddof=1) / np.sqrt(20)
        for name, got, want, sigma in zip(('e', '|m|', 'm2'), mean, exact_4x4(beta), err):
            assert abs(got - want) < 4 * sigma + 2e-3, (beta, name, got, want, sigma)


# ----------------------------------------------------------------------------- run_mc, plots, driver
def test_run_mc_bookkeeping():
    log = logging.getLogger('ising-test')
    cc, loc = setup([4, 4])
    field = IsingField(grid=loc, cartesiancomm=cc, initialization='random', seed=cc.parallel_RNG(seed=2))
    model = IsingModel(field=field, logger=log, beta=0.4)
    assert model.log is log and model.beta == 0.4
    assert (model.mlist, model.elist, model.nlist, model.acclist) == ([], [], [], [])
    with quiet():
        model.run_mc(steps=6, online_obs=True, progress=False)
    assert model.nlist == list(range(7))                                                    # 0, 1, ..., steps
    assert len(model.mlist) == len(model.elist) == 7 and len(model.acclist) == 6
    assert all(0.0 <= a <= 1.0 for a in model.acclist)                                      # fraction of flipped spins
    assert float(model.mlist[-1]) == float(model.magnetization()) and float(model.elist[-1]) == float(model.energy())
    assert -2.0 <= min(float(x) for x in model.elist) and max(float(x) for x in model.elist) <= 2.0
    # without online_obs nothing is recorded, and no progress bar means no output
    other = IsingModel(field=IsingField(grid=loc, cartesiancomm=cc), logger=None, beta=0.4)
    with quiet() as out:
        other.run_mc(steps=4, progress=False)
    assert (other.mlist, other.elist, other.nlist, other.acclist) == ([], [], [], [])
    assert out.getvalue() == ''
    with quiet():
        other.run_mc(steps=0, online_obs=True)                                              # zero steps: just the initial values
    assert other.nlist == [0] and other.acclist == []


def test_progress_bar_only_on_rank_zero():
    cc, loc = setup([4, 4])
    model = IsingModel(field=IsingField(grid=loc, cartesiancomm=cc), logger=None, beta=0.4)
    with quiet() as out:
        model.run_mc(steps=3)                       # progress=True: a progress bar from rank 0 (one line when not a terminal)
    assert (('MONTE CARLO' in out.getvalue() and '3/3' in out.getvalue()) if cc.rank == 0 else out.getvalue() == '')


def test_plot_mc_writes_the_figures():
    cc, loc = setup([4, 4])
    model = IsingModel(field=IsingField(grid=loc, cartesiancomm=cc), logger=None, beta=0.4)
    with in_tempdir():
        with quiet():
            model.run_mc(steps=5, online_obs=True, progress=False)
            model.plot_mc()
        for name in ('mag.png', 'ene.png', 'acc.png'):
            if cc.rank == 0:
                with open(name, 'rb') as f:
                    assert f.read(8) == b'\x89PNG\r\n\x1a\n' and os.path.getsize(name) > 5000
            else:
                assert not os.path.exists(name)                                             # only rank 0 plots
    with in_tempdir():                                                                      # nothing recorded: no plots
        fresh = IsingModel(field=IsingField(grid=loc, cartesiancomm=cc), logger=None, beta=0.4)
        with quiet() as out:
            fresh.plot_mc()
        assert os.listdir('.') == []
        assert ('NO PLOTS' in out.getvalue()) == (cc.rank == 0)


def test_command_line_driver():
    cc, loc = setup([4, 4])
    argv = ['--grid=' + 'x'.join(map(str, loc)), '--mpigrid=' + 'x'.join(map(str, cc.mpigrid)),
            '--steps=4', '--beta=0.5', '--seed=3']
    with in_tempdir():
        with quiet() as out:
            main(argv)
        if cc.rank == 0:
            assert all(os.path.getsize(f) > 5000 for f in ('mag.png', 'ene.png', 'acc.png'))
            assert 'saved to ./mag.png' in out.getvalue() and 'MONTE CARLO' in out.getvalue()
    for handler in list(logging.getLogger('pylattice').handlers):                          # the Logger of main()
        logging.getLogger('pylattice').removeHandler(handler)
    with quiet(), pytest.raises(SystemExit) as exc:                                         # a bad command line stops early
        main(['--grid=4x4', '--mpigrid=2x2x2'])
    assert exc.value.code == 2


if __name__ == '__main__':
    import traceback
    tests = [(name, f) for name, f in sorted(globals().items()) if name.startswith('test_') and callable(f)]
    for name, f in tests:
        try:
            f()
        except BaseException:
            # a rank that fails must stop them all: the others would wait for it in a collective call
            print('[x] {} failed on rank {}'.format(name, MPI.COMM_WORLD.Get_rank()), flush=True)
            traceback.print_exc()
            MPI.COMM_WORLD.Abort(1)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('[+] {} passed on {} rank(s)'.format(name, MPI.COMM_WORLD.Get_size()))
