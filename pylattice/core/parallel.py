from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

def pprint(comm, msg):
    if comm.Get_rank() == 0:
        print(msg)

def pplot(comm,x,y,title,file='./ising.png'): 
    if comm.Get_rank() == 0:
        plt.figure()
        plt.plot(x,y,'-k')
        plt.xlabel('trj')
        plt.title(title)
        plt.savefig(file)
        print('plot: {} - saved to {}'.format(title,file))

class CartesianComm():
    def __init__(self, mpigrid, periodicity=True):
        self.comm, self.size, self.rank = self.create_comm()
        self.mpigrid = mpigrid
        self.mpidim = len(self.mpigrid)
        self.periodicity = [periodicity]*self.mpidim
        self.mpicoord, self.cartesian = self.create_cartesian()
        self.mpitypes = {'float32': MPI.FLOAT, 'complex64': MPI.COMPLEX}
        self.cartesian.barrier()

    def parallel_RNG(self,seed=0): 
        if self.rank == 0: 
            np.random.seed(seed=seed)
            snd_seed = np.random.randint(low=0,high=9999,dtype='i',size=np.prod(self.mpigrid))
        else : 
            snd_seed = None
        recv_seed = np.array(1,dtype='i')
        self.cartesian.Scatter(snd_seed,recv_seed,root=0)
        self.cartesian.barrier()
        return recv_seed 

    def find_nn_mpi(self, mu, displacement=1):
        left, right = tuple(MPI.Cartcomm.Shift(
            self.cartesian, int(mu), displacement))
        return left, right

    def forwardshift(self, mu, snd_buf=np.array([1], dtype='float32'), dtype='float32'):
        # Returns the buffer held by the +mu neighbour (send to -mu, receive from +mu),
        # i.e. the data at x + mu.
        # Receive into a separate buffer: send and receive buffers must not overlap in MPI.
        # The datatype is inferred from the arrays (`dtype` is kept only for compatibility).
        rec_buf = snd_buf.copy()
        l, r = self.find_nn_mpi(mu)
        self.cartesian.Sendrecv(sendbuf=snd_buf, dest=l, sendtag=0,
                                recvbuf=rec_buf, source=r, recvtag=0)
        return rec_buf

    def backwardshift(self, mu, snd_buf=np.array([1], dtype='float32'), dtype='float32'):
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
    CC.backwardshift(mu=1)
