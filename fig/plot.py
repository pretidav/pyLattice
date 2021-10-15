import numpy as np
import matplotlib.pyplot as plt
dataQ = [77.97438859939575, 46.04473257064819, 31.80805730819702, 38.320847511291504, 39.833889961242676]
dataP = [87.73139786720276, 57.25099849700928, 43.5087628364563, 38.92410230636597]
dataT = [79.55425310134888, 192.892076253891, 233.64544987678528, 285.8275969028473]
baseline = 79.00875163078308
x     = [1,2,4,8,16]
h1 = plt.scatter(x=x, y=dataQ,marker='v',alpha=0.6,color='green')
h2 = plt.scatter(x=x[:-1], y=dataP,marker='s',alpha=0.6,color='blue')
h3 = plt.scatter(x=x[:-1], y=dataT,marker='o',alpha=0.6,color='red')
h4 = plt.hlines(xmin=x[0]-0.5, xmax=x[-1]+0.5, y=baseline,linestyles='--',color='black')
plt.legend([h1,h2,h3,h4],['Queue Processes','Processes','Threads','Naive'])
plt.xlabel('processes')
plt.ylabel('seconds')
plt.grid()
plt.savefig('perf.png')

