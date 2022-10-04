#import of library

import heapq as hq
import random
import random as rdm
import numpy as np
from datetime import datetime

rdm.seed(datetime.now())
SimulationTime = 0

# State variable
StateMMPP = 'G'
NumSysTransactionG = 0
NumSysTransactionB = 0

# System parameters
lambdaG = 1/2

lambdaB = 1/10

lambdaGA = 1/12

lambdaAG = 1/12

muG = 1/2

muB = 1/2

TimeinStateG = 0

TimeinStateA = 0

TimeAllTransition = 0


#start the simulation
for i in range(0, 1000):

    if StateMMPP == 'G':

        if NumSysTransactionG == 0 and NumSysTransactionB == 0:

            TimeNewTransactionG = random.expovariate(lambdaG) #return a random number with exponential distribution following lambdaG
            print(TimeNewTransactionG)
            TimeNaturalRecoveryG = random.expovariate(muG) #return a random number with exponential distribution following muG
            print(TimeNaturalRecoveryG)
            TimeNaturalRecoveryB = random.expovariate(muB) #return a random number with exponential distribution following muB
            print(TimeNaturalRecoveryB)
            TimeGATransition = random.expovariate(lambdaGA) #return a random number with exponential distribution following lambdaGA
            print(TimeGATransition)

            Action = min(TimeNewTransactionG, TimeGATransition) #take the smallest of the variables
            print(Action)

            if TimeNewTransactionG == Action:
                NumSysTransactionG = NumSysTransactionG + 1  #Add the number of transaction on system
                SimulationTime = SimulationTime + TimeNewTransactionG  #update the simulation time
                TimeinStateG = TimeinStateG + Action #update the time in state G

            if TimeGATransition == Action:
                StateMMPP = 'A' #change the state
                SimulationTime = SimulationTime + TimeGATransition #update the simulation time
                TimeAllTransition = TimeAllTransition + Action #update the total transition time

            print('Número de transações G no sistema:', NumSysTransactionG)
            print('Número de transações B no sistema:', NumSysTransactionB)
            print('state', StateMMPP)
            print('Time de procesos executados:', SimulationTime)

        if NumSysTransactionG > 0 and NumSysTransactionB == 0:

            TimeNewTransactionG = random.expovariate(lambdaG) #return a random number with exponential distribution following lambdaG
            print(TimeNewTransactionG)
            TimeNaturalRecoveryG = random.expovariate(muG) #return a random number with exponential distribution following muG
            print(TimeNaturalRecoveryG)
            TimeNaturalRecoveryB = random.expovariate(muB) #return a random number with exponential distribution following mub
            print(TimeNaturalRecoveryB)
            TimeGATransition = random.expovariate(lambdaGA) #return a random number with exponential distribution following lambdaGA
            print(TimeGATransition)

            Action = min(TimeNewTransactionG, TimeNaturalRecoveryG, TimeGATransition) #take the smallest of the variables
            print(Action)

            if TimeNewTransactionG == Action:
                NumSysTransactionG = NumSysTransactionG + 1  #Add the number of transaction on system
                SimulationTime = SimulationTime + TimeNewTransactionG #update the simulation time
                TimeinStateG = TimeinStateG + Action #update the time on state G

            if TimeNaturalRecoveryG == Action:
                NumSysTransactionG = NumSysTransactionG - 1 #delete 1 transaction on system
                SimulationTime = SimulationTime + TimeNaturalRecoveryG #update the simulation time
                TimeinStateG = TimeinStateG + Action  #update the time on state G

            if TimeGATransition == Action:
                StateMMPP = 'A' #change the state
                SimulationTime = SimulationTime + TimeGATransition #update the simulation time
                TimeAllTransition = TimeAllTransition + Action #update total transaction time

            print('Número de transações G no sistema:', NumSysTransactionG)
            print('Número de transações B no sistema:', NumSysTransactionB)
            print('state', StateMMPP)
            print('Time de procesos executados:', SimulationTime)

        if NumSysTransactionG == 0 and NumSysTransactionB > 0:

            TimeNewTransactionG = random.expovariate(lambdaG)  #return a random number with exponential distribution following lambdaG
            print(TimeNewTransactionG)
            TimeNaturalRecoveryG = random.expovariate(muG)  #return a random number with exponential distribution following muG
            print(TimeNaturalRecoveryG)
            TimeNaturalRecoveryB = random.expovariate(muB)  #return a random number with exponential distribution following muB
            print(TimeNaturalRecoveryB)
            TimeGATransition = random.expovariate(lambdaGA)  #return a random number with exponential distribution following lambdaGA
            print(TimeGATransition)

            Action = min(TimeNewTransactionG, TimeNaturalRecoveryB, TimeGATransition) #take the smallest of the variables
            print(Action)

            if TimeNewTransactionG == Action:
                NumSysTransactionG = NumSysTransactionG + 1 #Add the number of transaction on system
                SimulationTime = SimulationTime + TimeNewTransactionG #update the simulation time
                TimeinStateG = TimeinStateG + Action #update the time on state G

            if TimeNaturalRecoveryB == Action:
                NumSysTransactionB = NumSysTransactionB - 1  #delete 1 transaction on system
                SimulationTime = SimulationTime + TimeNaturalRecoveryB #update the simulation time
                TimeinStateG = TimeinStateG + Action #update the time on state G

            if TimeGATransition == Action:
                StateMMPP = 'A'  #change the state
                SimulationTime = SimulationTime + TimeGATransition #update the simulation time
                TimeAllTransition = TimeAllTransition + Action #update the total time of transition

            print('Número de transações G no sistema:', NumSysTransactionG)
            print('Número de transações B no sistema:', NumSysTransactionB)
            print('state', StateMMPP)
            print('Time de procesos executados:', SimulationTime)

        if NumSysTransactionG > 0 and NumSysTransactionB > 0:

            TimeNewTransactionG = random.expovariate(lambdaG) #return a random number with exponential distribution following lambdaG
            print(TimeNewTransactionG)
            TimeNaturalRecoveryG = random.expovariate(muG) #return a random number with exponential distribution following muG
            print(TimeNaturalRecoveryG)
            TimeNaturalRecoveryB = random.expovariate(muB) #return a random number with exponential distribution following muB
            print(TimeNaturalRecoveryB)
            TimeGATransition = random.expovariate(lambdaGA) #return a random number with exponential distribution following lambdaGA
            print(TimeGATransition)

            Action = min(TimeNewTransactionG, TimeNaturalRecoveryG, TimeNaturalRecoveryB, TimeGATransition) #take the smallest of the variables
            print(Action)

            if TimeNewTransactionG == Action:
                NumSysTransactionG = NumSysTransactionG + 1 #Add the number of transaction on system
                SimulationTime = SimulationTime + TimeNewTransactionG  #update the simulation time
                TimeinStateG = TimeinStateG + Action #update the time on state G

            if TimeNaturalRecoveryG == Action:
                NumSysTransactionG = NumSysTransactionG - 1 #delete 1 transaction on system
                SimulationTime = SimulationTime + TimeNaturalRecoveryG #update the simulation time
                TimeinStateG = TimeinStateG + Action #Update the time on state G

            if TimeNaturalRecoveryB == Action:
                NumSysTransactionB = NumSysTransactionB - 1 #delete 1 transaction on system
                SimulationTime = SimulationTime + TimeNaturalRecoveryB #update the simulation time
                TimeinStateG = TimeinStateG + Action #Update the time on state G

            if TimeGATransition == Action:
                StateMMPP = 'A' #change the state
                SimulationTime = SimulationTime + TimeGATransition #update the simulation time
                TimeAllTransition = TimeAllTransition + Action #Update the total time on state G

            print('Número de transações G no sistema:', NumSysTransactionG)
            print('Número de transações B no sistema:', NumSysTransactionB)
            print('state', StateMMPP)
            print('Time de procesos executados:', SimulationTime)


    #The same comments has done on state G are applicable on state A
    if StateMMPP == 'A':

        if NumSysTransactionG == 0 and NumSysTransactionB == 0:

            TimeNewTransactionG = random.expovariate(lambdaG)
            print(TimeNewTransactionG)
            TimeNewTransactionB = random.expovariate(lambdaB)
            print(TimeNewTransactionB)
            TimeNaturalRecoveryG = random.expovariate(muG)
            print(TimeNaturalRecoveryG)
            TimeNaturalRecoveryB = random.expovariate(muB)
            print(TimeNaturalRecoveryB)
            TimeAGTransition = random.expovariate(lambdaAG)
            print(TimeAGTransition)

            Action = min(TimeNewTransactionG, TimeNewTransactionB, TimeAGTransition)
            print(Action)

            if TimeNewTransactionG == Action:  # probabilidade de ocorrer uma chegada
                NumSysTransactionG = NumSysTransactionG + 1
                SimulationTime = SimulationTime + TimeNewTransactionG
                TimeinStateA = TimeinStateA + Action

            if TimeNewTransactionB == Action:  # probabilidade de ocorrer uma chegada
                NumSysTransactionB = NumSysTransactionB + 1
                SimulationTime = SimulationTime + TimeNewTransactionB
                TimeinStateA = TimeinStateA + Action

            if TimeAGTransition == Action:
                StateMMPP = 'G'
                SimulationTime = SimulationTime + TimeAGTransition
                TimeAllTransition = TimeAllTransition + Action

            print('Número de transações G no sistema:', NumSysTransactionG)
            print('Número de transações B no sistema:', NumSysTransactionB)
            print('state', StateMMPP)
            print('Time de procesos executados:', SimulationTime)

        if NumSysTransactionG > 0 and NumSysTransactionB == 0:

            TimeNewTransactionG = random.expovariate(lambdaG)
            print(TimeNewTransactionG)
            TimeNewTransactionB = random.expovariate(lambdaB)
            print(TimeNewTransactionB)
            TimeNaturalRecoveryG = random.expovariate(muG)
            print(TimeNaturalRecoveryG)
            TimeNaturalRecoveryB = random.expovariate(muB)
            print(TimeNaturalRecoveryB)
            TimeAGTransition = random.expovariate(lambdaAG)
            print(TimeAGTransition)

            Action = min(TimeNewTransactionG, TimeNewTransactionB,TimeNaturalRecoveryG, TimeAGTransition)
            print(Action)

            if TimeNewTransactionG == Action:  # probabilidade de ocorrer uma chegada
                NumSysTransactionG = NumSysTransactionG + 1
                SimulationTime = SimulationTime + TimeNewTransactionG
                TimeinStateA = TimeinStateA + Action

            if TimeNewTransactionB == Action:  # probabilidade de ocorrer uma chegada
                NumSysTransactionB = NumSysTransactionB + 1
                SimulationTime = SimulationTime + TimeNewTransactionB
                TimeinStateA = TimeinStateA + Action

            if TimeNaturalRecoveryG == Action:
                NumSysTransactionG = NumSysTransactionG - 1
                SimulationTime = SimulationTime + TimeNaturalRecoveryG
                TimeinStateA = TimeinStateA + Action

            if TimeAGTransition == Action:
                StateMMPP = 'G'
                SimulationTime = SimulationTime + TimeAGTransition
                TimeAllTransition = TimeAllTransition + Action

            print('Número de transações G no sistema:', NumSysTransactionG)
            print('Número de transações B no sistema:', NumSysTransactionB)
            print('state', StateMMPP)
            print('Time de procesos executados:', SimulationTime)

        if NumSysTransactionG == 0 and NumSysTransactionB > 0:

            TimeNewTransactionG = random.expovariate(lambdaG)
            print(TimeNewTransactionG)
            TimeNewTransactionB = random.expovariate(lambdaB)
            print(TimeNewTransactionB)
            TimeNaturalRecoveryG = random.expovariate(muG)
            print(TimeNaturalRecoveryG)
            TimeNaturalRecoveryB = random.expovariate(muB)
            print(TimeNaturalRecoveryB)
            TimeAGTransition = random.expovariate(lambdaAG)
            print(TimeAGTransition)

            Action = min(TimeNewTransactionG, TimeNewTransactionB, TimeNaturalRecoveryB, TimeAGTransition)
            print(Action)

            if TimeNewTransactionG == Action:  # probabilidade de ocorrer uma chegada
                NumSysTransactionG = NumSysTransactionG + 1
                SimulationTime = SimulationTime + TimeNewTransactionG
                TimeinStateA = TimeinStateA + Action

            if TimeNewTransactionB == Action:  # probabilidade de ocorrer uma chegada
                NumSysTransactionB = NumSysTransactionB + 1
                SimulationTime = SimulationTime + TimeNewTransactionG
                TimeinStateA = TimeinStateG + Action

            if TimeNaturalRecoveryB == Action:
                NumSysTransactionB = NumSysTransactionB - 1
                SimulationTime = SimulationTime + TimeNaturalRecoveryB
                TimeinStateA = TimeinStateA + Action

            if TimeAGTransition == Action:
                StateMMPP = 'G'
                SimulationTime = SimulationTime + TimeAGTransition
                TimeAllTransition = TimeAllTransition + Action

            print('Número de transações G no sistema:', NumSysTransactionG)
            print('Número de transações B no sistema:', NumSysTransactionB)
            print('state', StateMMPP)
            print('Time de procesos executados:', SimulationTime)

        if NumSysTransactionG > 0 and NumSysTransactionB > 0:

            TimeNewTransactionG = random.expovariate(lambdaG)
            print(TimeNewTransactionG)
            TimeNewTransactionB = random.expovariate(lambdaB)
            print(TimeNewTransactionB)
            TimeNaturalRecoveryG = random.expovariate(muG)
            print(TimeNaturalRecoveryG)
            TimeNaturalRecoveryB = random.expovariate(muB)
            print(TimeNaturalRecoveryB)
            TimeAGTransition = random.expovariate(lambdaAG)
            print(TimeAGTransition)

            Action = min(TimeNewTransactionG, TimeNewTransactionB, TimeNaturalRecoveryG, TimeNaturalRecoveryB, TimeAGTransition)
            print(Action)

            if TimeNewTransactionG == Action:  # probabilidade de ocorrer uma chegada
                NumSysTransactionG = NumSysTransactionG + 1
                SimulationTime = SimulationTime + TimeNewTransactionG
                TimeinStateG = TimeinStateG + Action

            if TimeNaturalRecoveryG == Action:
                NumSysTransactionG = NumSysTransactionG - 1
                SimulationTime = SimulationTime + TimeNaturalRecoveryG
                TimeinStateG = TimeinStateG + Action

            if TimeNewTransactionG == Action:  # probabilidade de ocorrer uma chegada
                NumSysTransactionG = NumSysTransactionG + 1
                SimulationTime = SimulationTime + TimeNewTransactionG
                TimeinStateG = TimeinStateG + Action

            if TimeNaturalRecoveryB == Action:
                NumSysTransactionB = NumSysTransactionB - 1
                SimulationTime = SimulationTime + TimeNaturalRecoveryB
                TimeinStateG = TimeinStateG + Action

            if TimeAGTransition == Action:
                StateMMPP = 'G'
                SimulationTime = SimulationTime + TimeAGTransition
                TimeAllTransition = TimeAllTransition + Action

            print('Número de transações G no sistema:', NumSysTransactionG)
            print('Número de transações B no sistema:', NumSysTransactionB)
            print('state', StateMMPP)
            print('Time de procesos executados:', SimulationTime)

print('Número de transações G no sistema:', NumSysTransactionG)
print('Número de transações B no sistema:', NumSysTransactionB)
print('state', StateMMPP)
print('Time de procesos executados:', SimulationTime)
print('Tempo no estado G',TimeinStateG)
print('Tempo no estado A',TimeinStateA)
print('Tempo de transição',TimeAllTransition)














'''''''''''''''''''''''''''
    if StateMMPP == 'G' and NumSysTransactionA == 0 and NumSysTransactionG == 0:

        TimeNewTransactionG = random.expovariate(lambdaG)
        print(TimeNewTransactionG)
        TimeGATransition = random.expovariate(lambdaGA)
        print(TimeGATransition)
        print('__________________________________')

        Action = min(TimeNewTransactionG,TimeGATransition)
        print(Action)

        if TimeNewTransactionG == Action: #probabilidade de ocorrer uma chegada
            NumSysTransactionG =NumSysTransactionG +1
            SimulationTime = SimulationTime + TimeNewTransactionG
            TimeinStateG = TimeinStateG+Action

        if TimeGATransition == Action:
            StateMMPP = 'A'
            SimulationTime = SimulationTime + TimeGATransition
            TimeAllTransition = TimeAllTransition + Action

        print('Número de transações no sistema:', NumSysTransactionG)
        print('state', StateMMPP)
        print('Time de procesos executados:', SimulationTime)

    if StateMMPP == 'G' and NumSysTransactionG > 0 and NumSysTransactionA == 0:

        TimeNewTransactionG = random.expovariate(lambdaG)
        print(TimeNewTransactionG)
        TimeNaturalRecoveryG = random.expovariate(muG)
        print(TimeNaturalRecoveryG)
        TimeNaturalRecoveryA = random.expovariate(muA)
        print(TimeNaturalRecoveryA)
        TimeGATransition = random.expovariate(lambdaGA)
        print(TimeGATransition)

        Action = min(TimeNewTransactionG, TimeNaturalRecoveryG, TimeGATransition)
        print(Action)

        if TimeNewTransactionG == Action:
            NumSysTransactionG =NumSysTransactionG +1
            SimulationTime = SimulationTime + TimeNewTransactionG
            TimeinStateG = TimeinStateG + Action


        if TimeNaturalRecoveryG == Action:
            NumSysTransactionG = NumSysTransactionG - 1
            SimulationTime = SimulationTime + TimeNaturalRecoveryG
            TimeinStateG = TimeinStateG + Action


        if TimeGATransition == Action:
            StateMMPP = 'A'
            SimulationTime = SimulationTime + TimeGATransition
            TimeAllTransition = TimeAllTransition + Action

        print('Número de transações no sistema:', NumSysTransactionG)
        print('state', StateMMPP)
        print('Time de procesos executados:', SimulationTime)


    if StateMMPP == 'G' and NumSysTransactionG == 0 and NumSysTransactionA > 0:

        TimeNewTransactionG = random.expovariate(lambdaG)
        print(TimeNewTransactionG)
        TimeNaturalRecovery = random.expovariate(muG)
        print(TimeNaturalRecovery)
        TimeNaturalRecovery = random.expovariate(muA)
        print(TimeNaturalRecovery)
        TimeGATransition = random.expovariate(lambdaGA)
        print(TimeGATransition)

        Action = min(TimeNewTransactionG, TimeNaturalRecovery, TimeGATransition)
        print(Action)

        if TimeNewTransactionG == Action:
            NumSysTransactionG = NumSysTransactionG + 1
            SimulationTime = SimulationTime + TimeNewTransactionG
            TimeinStateG = TimeinStateG + Action

        if TimeNaturalRecovery == Action:
            NumSysTransactionG = NumSysTransactionG - 1
            SimulationTime = SimulationTime + TimeNaturalRecovery
            TimeinStateG = TimeinStateG + Action

        if TimeGATransition == Action:
            StateMMPP = 'A'
            SimulationTime = SimulationTime + TimeGATransition
            TimeAllTransition = TimeAllTransition + Action

        print('Número de transações no sistema:', NumSysTransactionG)
        print('state', StateMMPP)
        print('Time de procesos executados:', SimulationTime)

    if StateMMPP == 'G' and NumSysTransactionG > 0 and NumSysTransactionA > 0:

        TimeNewTransactionG = random.expovariate(lambdaG)
        print(TimeNewTransactionG)
        TimeNaturalRecoveryG = random.expovariate(muG)
        print(TimeNaturalRecoveryG)
        TimeNaturalRecoveryB = random.expovariate(muB)
        print(TimeNaturalRecoveryB)
        TimeGATransition = random.expovariate(lambdaGA)
        print(TimeGATransition)

        Action = min(TimeNewTransactionG, TimeNaturalRecoveryG, TimeNaturalRecoveryB, TimeGATransition)
        print(Action)

        if TimeNewTransactionG == Action:
            NumSysTransactionG = NumSysTransactionG + 1
            SimulationTime = SimulationTime + TimeNewTransactionG
            TimeinStateG = TimeinStateG + Action

        if TimeNaturalRecoveryG == Action:
            NumSysTransactionG = NumSysTransactionG - 1
            SimulationTime = SimulationTime + TimeNaturalRecoveryG
            TimeinStateG = TimeinStateG + Action

        if TimeNaturalRecoveryB == Action:
            NumSysTransactionB = NumSysTransactionB - 1
            SimulationTime = SimulationTime + TimeNaturalRecoveryB
            TimeinStateG = TimeinStateG + Action

        if TimeGATransition == Action:
            StateMMPP = 'A'
            SimulationTime = SimulationTime + TimeGATransition
            TimeAllTransition = TimeAllTransition + Action

        print('Número de transações no sistema:', NumSysTransactionG)
        print('state', StateMMPP)
        print('Time de procesos executados:', SimulationTime)


    if StateMMPP == 'A' and NumSysTransactionG == 0 and NumSysTransactionA == 0:

        TimeNewTransactionG = random.expovariate(lambdaG)
        print(TimeNewTransactionG)
        TimeNewTransactionB = random.expovariate(lambdaB)
        print(TimeNewTransactionB)
        TimeAGTransition = random.expovariate(lambdaAG)
        print(TimeAGTransition)

        Action = min(TimeNewTransactionG, TimeNewTransactionB, TimeAGTransition)
        print(Action)

        if TimeNewTransactionG == Action:  # probabilidade de ocorrer uma chegada
            NumSysTransactionG = NumSysTransactionG +1
            SimulationTime = SimulationTime + TimeNewTransactionG
            TimeinStateA = TimeinStateA + Action


        if TimeNewTransactionB == Action:  # probabilidade de ocorrer uma chegada
            NumSysTransactionB = NumSysTransactionB +1
            SimulationTime = SimulationTime + TimeNewTransactionB
            TimeinStateA = TimeinStateA + Action


        if TimeAGTransition == Action:
            StateMMPP = 'G'
            SimulationTime = SimulationTime + TimeAGTransition
            TimeAllTransition = TimeAllTransition + Action

        print('Número de transações no sistema:', NumSysTransactionG)
        print('Número de transações no sistema:', NumSysTransactionB)
        print('state', StateMMPP)
        print('Time de procesos executados:', SimulationTime)



    if StateMMPP == 'A' and NumSysTransactionG == 0 and NumSysTransactionA == 0:

        TimeNewTransactionG = random.expovariate(lambdaG)
        print(TimeNewTransactionG)
        TimeNewTransactionB = random.expovariate(lambdaB)
        print(TimeNewTransactionB)
        TimeNaturalRecovery = random.expovariate(mu)
        print(TimeNaturalRecovery)
        TimeAGTransition = random.expovariate(lambdaAG)
        print(TimeAGTransition)

        Action = min(TimeNewTransactionG, TimeNewTransactionB, TimeNaturalRecovery, TimeAGTransition)
        print(Action)

        if TimeNewTransactionG == Action:
            NumSysTransaction = NumSysTransaction +1
            SimulationTime = SimulationTime + TimeNewTransactionG
            TimeinStateA = TimeinStateA + Action


        if TimeNewTransactionB == Action:  # probabilidade de ocorrer uma chegada
            NumSysTransaction = NumSysTransaction +1
            SimulationTime = SimulationTime + TimeNewTransactionB
            TimeinStateA = TimeinStateA + Action


        if TimeNaturalRecovery == Action:
            NumSysTransaction = NumSysTransaction - 1
            SimulationTime = SimulationTime + TimeNaturalRecovery
            TimeinStateA = TimeinStateA + Action

        if TimeAGTransition == Action:
            StateMMPP = 'G'
            SimulationTime = SimulationTime + TimeAGTransition
            TimeAllTransition = TimeAllTransition + Action

        print('Número de transações no sistema:', NumSysTransaction)
        print('state', StateMMPP)
        print('Time de procesos executados:', SimulationTime)


print('NumSysTransaction', NumSysTransaction)
print('Tenpo de simulação', SimulationTime)
print('Tempo in State G', TimeinStateG)
print('Tempo in State A', TimeinStateA)
print('Tempo de todas as Transições', TimeAllTransition)
'''''''''''''''''''''''''''''''''''''''''''''''