#from lattice.grid import *

from threading import Thread

def threaded(N):
    def inner(fn):
        def wrapper(*args, **kwargs):
            threads = [Thread(target=fn,args=args,kwargs=kwargs) for _ in range(N)]
            for t in threads:
                print('started') 
                t.start()
            for t in threads: 
                t.join()          
        return wrapper 
    return inner 

@threaded(N=3)
def f(a,b,i):
    print(a[i]**b[i])

#f([1,2],[1,3],1)



class Worker(Thread):
    def __init__(self, target, a,b,i):
      Thread.__init__(self)
      self.target = target
      self.i = i
      self.a = a 
      self.b = b
    def run(self):
        self.target(a=self.a,b=self.b,i=self.i)

def threaded(N):
    def inner(fn):
        def wrapper(a,b):
            threads = [Worker(target=fn,a=a,b=b,i=i) for i in range(N)]
            for t in threads:
                print('started') 
                t.start()
            for t in threads: 
                t.join()          
        return wrapper 
    return inner 

@threaded(N=2)
def f(a,b,i=None):
    print(a[i]**b[i])

f([1,2],[1,3])