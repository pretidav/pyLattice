from mpi4py import MPI
import numpy as np

def pprint(comm,msg): 
    if comm.Get_rank()==0: 
        print(msg)

class CartesianComm():
    def __init__(self, mpigrid, periodicity=True):
        self.comm, self.size, self.rank = self.create_comm()
        self.mpigrid = mpigrid
        self.mpidim = len(self.mpigrid)
        self.periodicity = [periodicity]*self.mpidim
        self.mpicoord, self.cartesian = self.create_cartesian()
        self.cartesian.barrier()
        
    def find_nn_mpi(self,mu,displacement=1): 
        left,right = tuple(MPI.Cartcomm.Shift(self.cartesian,int(mu),displacement))
        return left, right 

    def forwardshift(self,mu,snd_buf=np.array([1],dtype='float')):
        rec_buf = snd_buf 
        l,r = self.find_nn_mpi(mu)
        self.cartesian.Sendrecv(sendbuf=[snd_buf, MPI.FLOAT],dest=r,sendtag=0,
                            recvbuf=[rec_buf, MPI.FLOAT],source=l,recvtag=0)
        return rec_buf

    def backwardshift(self,mu,snd_buf=np.array([1],dtype='float')):
        rec_buf = snd_buf 
        l,r = self.find_nn_mpi(mu)
        self.cartesian.Sendrecv(sendbuf=[snd_buf, MPI.FLOAT],dest=l,sendtag=0,
                            recvbuf=[rec_buf, MPI.FLOAT],source=r,recvtag=0)
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

if __name__=='__main__':
    CC = CartesianComm(mpigrid=[2,2])
    print(CC.mpicoord)
    print(CC.forwardshift(mu=1,snd_buf=np.array([[2,2],[1,1]],dtype='float')))
    CC.backwardshift(mu=1)
