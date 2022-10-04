#import library

import heapq as hq
import random
import random as rdm
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Start of simulation

rdm.seed(datetime.now())
SimulationTime = 0

#State Variable

NumSysTransaction = 0

#System Parameters
lambdaG = 1/4 #rate of arrival

mu = 1/2  #rate of service


for i in range(0, 2000):

    if NumSysTransaction == 0:

        TimeNewTransactionG = random.expovariate(lambdaG) #return a random number with exponential distribution following lambdaG
        print(TimeNewTransactionG)

        NumSysTransaction = NumSysTransaction + 1  #add the number of transaction on system
        SimulationTime = SimulationTime + TimeNewTransactionG #update the simulation time


        print('Número de transações no sistema:', NumSysTransaction)
        print('Time de procesos executados:', SimulationTime)


    if NumSysTransaction > 0:

        TimeNewTransactionG = random.expovariate(lambdaG) #return a random number with exponential distribution following lambdaG
        print(TimeNewTransactionG)
        TimeNaturalRecovery = random.expovariate(mu) #return a random number with exponential distribution following mu
        print(TimeNaturalRecovery)

        Action = min(TimeNewTransactionG, TimeNaturalRecovery) #take the smallest of the variables
        print(Action)

        if TimeNewTransactionG == Action:
            NumSysTransaction = NumSysTransaction + 1 #add the number of transaction on system
            SimulationTime = SimulationTime + TimeNewTransactionG #update the simulation time

        if TimeNaturalRecovery == Action:
            NumSysTransaction = NumSysTransaction - 1 #delete 1 transaction on system
            SimulationTime = SimulationTime + TimeNaturalRecovery  #update the simulation time

        print('Número de transações no sistema:', NumSysTransaction)
        print('Time de procesos executados:', SimulationTime)


