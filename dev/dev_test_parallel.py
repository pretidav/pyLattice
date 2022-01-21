from lattice.grid import *


#def test_realfield():
Grid = LatticeParallel(grid=[20,20,20,20],pgrid=[2,2,2,1])
# # # RealField = LatticeReal(lattice=Grid)
# # # RealField.fill_value(3)
# # # RealField2 = LatticeReal(lattice=Grid)
# # # RealField2.fill_value(2)

# MatrixField = LatticeVectorRealMatrix(lattice=Grid,Nd=2,N=3)
# MatrixField2 = LatticeVectorRealMatrix(lattice=Grid,Nd=2,N=3)
# V = VectorRealMatrix(Nd=2,N=3,value=np.array([[[1,2,3],[1,2,3],[1,2,3]],[[4,5,6],[4,5,6],[4,5,6]]]))
# MatrixField.fill_value(V)
# MatrixField2.fill_value(V)


# import time
# start_time = time.time()
# MatrixField.det()
# print("--- %s seconds ---" % (time.time() - start_time))

# RealMatrixField  = LatticeRealMatrix(lattice=Grid,N=2)
# RealMatrixField2 = LatticeRealMatrix(lattice=Grid,N=2)

# r = RealMatrix(value=np.array([[1,2],[3,4]]))
# RealMatrixField.fill_value(n=r)
# print((RealMatrixField.det()).value)

from multiprocessing import Process, Manager, Array
import time 

def fn(q):
    while not q.empty():
        work = q.get() 
        print('process #{}'.format(work))
        arr[work]=1
        time.sleep(2)
        q.task_done()
    

Q = Manager().Queue(maxsize=0)
global arr 
arr = Array('i',np.array([0,0,0,0,0,0]),lock=False)
print(np.frombuffer(arr,dtype='int32'))


for i in range(5): 
    Q.put(i) 

for i in range(2):
    process = Process(target=fn, args=[Q]) 
    process.start()
Q.join()

print(np.frombuffer(arr,dtype='int32'))
#print(arr)

print('OUT')