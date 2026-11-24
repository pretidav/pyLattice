from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from pylattice.core.rng import rank_rng
import logging
import warnings
from contextlib import contextmanager, nullcontext

try:
    import plot_style
except ImportError:                 # plots still work, in the default matplotlib style
    plot_style = None

def pprint(comm, msg):
    if comm.Get_rank() == 0:
        print(msg)

@contextmanager
def _quiet_font_weight():
    """The style uses a bold font and a fallback family without a bold weight (Monaco on macOS) makes
    matplotlib log "findfont: Failed to find font weight bold ..." for every figure. This hides just
    that message, only while plotting: the other font warnings are kept."""
    log = logging.getLogger('matplotlib.font_manager')
    drop = lambda record: not record.getMessage().startswith('findfont: Failed to find font weight')
    log.addFilter(drop)
    try:
        yield
    finally:
        log.removeFilter(drop)

def pplot(comm,x,y,title,file): 
    """Line plot of y against x, written by rank 0 in the style of plot_style (without it, in the
    default matplotlib style). The style is applied only inside this function."""
    if comm.Get_rank() == 0:
        if plot_style is None:
            warnings.warn("plot_style is not installed (pip install -r requirements.txt): "
                          "plotting in the default matplotlib style")
        with _quiet_font_weight(), (plot_style.context() if plot_style is not None else nullcontext()):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(x,y,s=1.5)
            ax.set_xlabel('TRJ')
            ax.set_title(title.upper())
            if plot_style is not None:
                plot_style.style_axes(ax)
                plot_style.savefig(fig, file)
            else:
                fig.savefig(file)
        plt.close(fig)
        print('plot: {} - saved to {}'.format(title,file))

def allsum(comm, values):
    """The element-wise sum over all the ranks of comm of a number or of a list / array of numbers (every rank
    gets the result). The type of the numbers is kept: integer counts stay exact."""
    return comm.allreduce(np.asarray(values), op=MPI.SUM)


class CartesianComm():
    def __init__(self, mpigrid, periodicity=True):
        self.comm, self.size, self.rank = self.create_comm()
        self.mpigrid = mpigrid
        self.mpidim = len(self.mpigrid)
        self.periodicity = [periodicity]*self.mpidim
        self.mpicoord, self.cartesian = self.create_cartesian()
        self.mpitypes = {'float32': MPI.FLOAT, 'complex64': MPI.COMPLEX,
                         'float64': MPI.DOUBLE, 'complex128': MPI.DOUBLE_COMPLEX}
        self.cartesian.barrier()

    def rng(self, seed=0):
        """the numpy Generator of this rank: an independent stream for every rank (see pylattice.core.rng)"""
        return rank_rng(seed, self.comm)

    def parallel_RNG(self,seed=0): 
        """An integer seed for this rank, for code that wants a number: rng() gives the stream itself.
        It is drawn from the independent streams of rank_rng, so ranks do not share seeds; nothing is
        communicated."""
        return int(np.random.SeedSequence(seed, spawn_key=(self.rank,)).generate_state(1)[0])

    def find_nn_mpi(self, mu, displacement=1):
        left, right = tuple(MPI.Cartcomm.Shift(
            self.cartesian, int(mu), displacement))
        return left, right

    def forwardshift(self, mu, snd_buf):
        # Returns the buffer held by the +mu neighbour (send to -mu, receive from +mu),
        # i.e. the data at x + mu.
        # Receive into a separate buffer: send and receive buffers must not overlap in MPI.
        # The datatype is that of snd_buf (single or double precision, real or complex): nothing is converted.
        rec_buf = snd_buf.copy()
        l, r = self.find_nn_mpi(mu)
        self.cartesian.Sendrecv(sendbuf=snd_buf, dest=l, sendtag=0,
                                recvbuf=rec_buf, source=r, recvtag=0)
        return rec_buf

    def backwardshift(self, mu, snd_buf):
        # Returns the buffer held by the -mu neighbour (send to +mu, receive from -mu),
        # i.e. the data at x - mu.
        rec_buf = snd_buf.copy()
        l, r = self.find_nn_mpi(mu)
        self.cartesian.Sendrecv(sendbuf=snd_buf, dest=r, sendtag=0,
                                recvbuf=rec_buf, source=l, recvtag=0)
        return rec_buf

    def create_comm(self):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        return comm, size, rank

    def create_cartesian(self):
        cartesian = self.comm.Create_cart(dims=self.mpigrid, periods=self.periodicity,
                                          reorder=False)
        mpicoord = cartesian.Get_coords(self.rank)
        return mpicoord, cartesian

    # def parallel_print_cnfg(self,value):
    #     break 

if __name__ == '__main__':
    CC = CartesianComm(mpigrid=[2, 2])
    print(CC.mpicoord)
    print(CC.forwardshift(mu=1, snd_buf=np.array(
        [[2, 2], [1, 1]], dtype='float')))
    CC.backwardshift(mu=1, snd_buf=np.array([[2, 2], [1, 1]], dtype='float'))
