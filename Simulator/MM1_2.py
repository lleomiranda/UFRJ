import heapq as hq
import random
import random as rdm
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Start of simulation

rdm.seed(datetime.now())
SimulationTime = 0

#start the simulation

NumSysTransaction = 0

#System Parameters
lambdaG = 1/4

mu = 1/2


for i in range(0, 2000):

    if NumSysTransaction == 0:

        TimeNewTransactionG = random.expovariate(lambdaG)
        print(TimeNewTransactionG)

        NumSysTransaction = NumSysTransaction + 1
        SimulationTime = SimulationTime + TimeNewTransactionG


        print('Número de transações no sistema:', NumSysTransaction)
        print('Time de procesos executados:', SimulationTime)


    if NumSysTransaction > 0:

        TimeNewTransactionG = random.expovariate(lambdaG)
        print(TimeNewTransactionG)
        TimeNaturalRecovery = random.expovariate(mu)
        print(TimeNaturalRecovery)

        Action = min(TimeNewTransactionG, TimeNaturalRecovery)
        print(Action)

        if TimeNewTransactionG == Action:
            NumSysTransaction = NumSysTransaction + 1
            SimulationTime = SimulationTime + TimeNewTransactionG

        if TimeNaturalRecovery == Action:
            NumSysTransaction = NumSysTransaction - 1
            SimulationTime = SimulationTime + TimeNaturalRecovery

        print('Número de transações no sistema:', NumSysTransaction)
        print('Time de procesos executados:', SimulationTime)

#print('********************************Bucket Estourou*******************************************')
#return StepstoBucketOverflow, SimulationTime


