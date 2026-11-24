import numpy as np
import matplotlib.pyplot as plt
import math
data = [4.035101890563965, 2.142939567565918, 1.300379753112793]
baseline = 4.168240785598755
x     = [1,2,4]
h1 = plt.scatter(x=x, y=data,marker='v',alpha=0.6,color='green')
h4 = plt.hlines(xmin=x[0]-0.5, xmax=x[-1]+0.5, y=baseline,linestyles='--',color='black')
plt.legend([h1, h4],['MPI','Naive'])
ew_list = range(math.floor(min(x)), math.ceil(max(x))+1)
plt.xticks(ew_list)
plt.xlabel('processes')
plt.ylabel('time [s]')
plt.grid()
plt.savefig('perf.png')

