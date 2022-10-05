# import of library
import random
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import scipy.stats as ss
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot_2samples
import pylab as py
import statsmodels.api as smi


# function for algorithm bucket
def bucket(depth, TLP):

    D = 0  # depth of Bucket
    steps_to_overflow=0
    proc_exec = []

    while D <= depth:  # check  overflow of bucket

        steps_to_overflow=steps_to_overflow+1

        if random.random() < TLP:  # check if process is smaller of TLP
            D = D + 1  # Add a token on the bucket

        else:  # check if process is smaller of TLP
            D = D - 1  # Remove a token on the bucket

            if D == -1:  # Avoid D negative
                D = 0

    return steps_to_overflow  # steps to bucket overflow


depth = 5  #depth of bucket
TLP = 0.46  #threshold
nruns=10   #number of times to execute the function
iterations = int(100000/nruns) #number o interations

StepsEnd = [] #list to save average steps until bucket overflow

for j in range(nruns):

    Steps = []  #list to save the number of steps to overflow of bucket

    for i in range(iterations):

        l = bucket(depth, TLP)  # return of function

        Steps.append(l)
        #E = ss.expon.fit(passos)
        #total_proc_exec.append(proc_exec)
    StepsEnd.append(np.mean(Steps))


E = ss.expon.fit(Steps) #An exponential continuous random variable.
e0 = round(E[0], 2)  #take the loc
e1 = round(E[1], 2)  #take the scale

#process of least square error
exp1 = ss.expon.rvs(loc=e0, scale=e1, size=iterations)
exp1.sort()  #ordering the exp1
Steps.sort()  #ordering the Steps
diff = Steps-exp1   #take the difference
diff2 = round(np.mean(list(map(abs, diff))), 2) #map extend absolute number logic to each element in the list.
#print('diff2', diff2)
MSE= round(np.dot(diff,diff)/iterations, 2) #make the scale product of two matrix
Ep = round(np.sqrt(MSE),2)
print('Erro padrão:', Ep)


mu = round(np.mean(StepsEnd), 6)
std = round(np.std(StepsEnd), 6)
var = round(np.var(StepsEnd), 6)
intconfinf = round(mu - 1.96 * std / np.sqrt(nruns), 6)
intconfsup = round(mu + 1.96 * std / np.sqrt(nruns), 6)


#hist plot
plt.hist(Steps, bins=300)
#plt.title(f"TLP = {TLP} \nDepth = {depth} \nIterations = {iterations}")
plt.xlabel('Steps to bucket overflow')
plt.ylabel('Frequency')
plt.show()


#qq plot
qqx = sm.ProbPlot(np.array(Steps))
qqy = sm.ProbPlot(np.array(exp1))
qqplot_2samples(qqx, qqy, line='45')
#plt.title(f"MSE: {MSE} Erro Padrão: {Ep}")
plt.grid()
plt.xlabel('quantiles of steps until a false alarm', fontsize= 17)
plt.ylabel('quantiles of exponential distribution', fontsize= 17)
plt.tick_params(labelsize=17)
plt.show()












