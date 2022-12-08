import heapq as hq
import random
import random as rdm
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


def bucket_MMPP(mu, lambdaG, lambdaA, lambdaAG, lambdaGA, threshold, depth):
    rdm.seed(datetime.now())
    SimulationTime = 0
    # System parameters
    TimeinStateG = 0
    TimeinStateA = 0
    TimeAllTransition = 0

    list_arrival_time = []
    list_departure_time = []
    response_time = []
    response_time_final = []
    list_TTS = []

    FP = []
    TP= []
    TN = []
    FN = []

    fp = 0
    tp = 0
    tn = 0
    fn = 0

    while SimulationTime <= 5000:
        # State variable
        StateMMPP = 'G'
        D = 0
        NumTotTransaction = 0

        while D <= depth:

            if StateMMPP == 'G':

                if NumTotTransaction == 0:  # no transaction on the system in state G

                    TimeArrivalNewTransaction = random.expovariate(lambdaG)  # return a random number with exponential distribution following lambdaG

                    TimeGATransition = random.expovariate(lambdaGA)  # return a random number with exponential distribution following lambdaGA

                    Action = min(TimeArrivalNewTransaction, TimeGATransition)  # take the smallest of the variables
                    SimulationTime = SimulationTime + Action  # update the simulation time

                    if TimeArrivalNewTransaction == Action:  # Arrival a new transaction on the system
                        NumTotTransaction = NumTotTransaction + 1  # increase the total number of transaction on system
                        #SimulationTime = SimulationTime + Action  # update the simulation time
                        TimeinStateG = TimeinStateG + Action  # update the time in state G
                        list_arrival_time.append(SimulationTime)  # add the time that the new transaction arrival on the system


                    if TimeGATransition == Action:  # Transition to state A
                        StateMMPP = 'A'  # change the state
                        #SimulationTime = SimulationTime + Action  # update the simulation time
                        TimeAllTransition = TimeAllTransition + Action  # update the total transition time

                if NumTotTransaction > 0:  # There are 1 ou more transaction on the system in state G

                    TimeArrivalNewTransaction = random.expovariate(lambdaG)  # return a random number with exponential distribution following lambdaG

                    TimeDepartureTransaction = random.expovariate(mu)  # return a random number with exponential distribution following mu

                    TimeGATransition = random.expovariate(lambdaGA)  # return a random number with exponential distribution following lambdaGA

                    Action = min(TimeArrivalNewTransaction, TimeDepartureTransaction,TimeGATransition)  # take the smallest of the variables
                    SimulationTime = SimulationTime + Action  # update the simulation time

                    if TimeArrivalNewTransaction == Action:  # Arrival a new transaction on the system
                        NumTotTransaction = NumTotTransaction + 1  # increase the total number of transaction on system
                        #SimulationTime = SimulationTime + Action  # update the simulation time
                        TimeinStateG = TimeinStateG + Action  # update the time on state G
                        list_arrival_time.append(SimulationTime)  # add the time that the new transaction arrival on the system


                    if TimeDepartureTransaction == Action:  # Departure a transaction on the system
                        NumTotTransaction = NumTotTransaction - 1  # decrease the total number of transaction on system
                        #SimulationTime = SimulationTime + Action  # update the simulation time
                        TimeinStateG = TimeinStateG + Action  # update the time on state G
                        list_departure_time.append(SimulationTime)  # add the time that the transaction leave on the system


                        if len(list_departure_time) != 0 and len(list_arrival_time) != 0:  # checking if there are transactions in the arrivals and departures lists
                            response_time.append(list_departure_time[0] - list_arrival_time[0])  # taking the response time of transaction
                            response_time_final.append(response_time[0])  # add the response time on the list of total response time of system

                            if response_time[0] > threshold:  # check if process is bigger than Threshold
                                D = D + 1  # Add a token on the bucket
                                tn += 1
                                TN.append(SimulationTime)

                            else:  # check if process is smaller than Threshold
                                D = D - 1  # Remove a token on the bucket

                                if D == -1:  # Avoid D negative
                                    D = 0
                            list_departure_time.pop(0)  # clearing the departure list
                            list_arrival_time.pop(0)  # clearing the arrival list
                            response_time.pop(0)  # clearing the response time list

                    if TimeGATransition == Action:  # Transition to state A
                        StateMMPP = 'A'  # change to state A
                        #SimulationTime = SimulationTime + Action  # update the simulation time
                        TimeAllTransition = TimeAllTransition + Action  # update total transaction time

            # The same comments has done on state G are applicable on state A
            if StateMMPP == 'A':

                if NumTotTransaction == 0:  # no transaction on the system in state G

                    TimeArrivalNewTransaction = random.expovariate(lambdaA)  # return a random number with exponential distribution following lambda A

                    TimeAGTransition = random.expovariate(lambdaAG)  # return a random number with exponential distribution following lambda AG

                    Action = min(TimeArrivalNewTransaction, TimeAGTransition)  # take the smallest of the variables
                    SimulationTime = SimulationTime + Action  # update the simulation time

                    if TimeArrivalNewTransaction == Action:  # Arrival a new transaction on the system
                        NumTotTransaction = NumTotTransaction + 1  # increase the total number of transaction on system
                        #SimulationTime = SimulationTime + Action  # update the simulation time
                        TimeinStateA = TimeinStateA + Action  # update the time on state A
                        list_arrival_time.append(SimulationTime)  # add the time that the new transaction arrival on the system


                    if TimeAGTransition == Action:  # Transition to state G
                        StateMMPP = 'G'  # change to state G
                        #SimulationTime = SimulationTime + Action  # update the simulation time
                        TimeAllTransition = TimeAllTransition + Action  # update total transaction time

                if NumTotTransaction > 0:  # There are 1 ou more transaction on the system in state G

                    TimeArrivalNewTransaction = random.expovariate(lambdaA)  # return a random number with exponential distribution following lambda A

                    TimeDepartureTransaction = random.expovariate(mu)  # return a random number with exponential distribution following mu

                    TimeAGTransition = random.expovariate(lambdaAG)  # return a random number with exponential distribution following lambda AG

                    Action = min(TimeArrivalNewTransaction, TimeDepartureTransaction,TimeAGTransition)  # take the smallest of the variables
                    SimulationTime = SimulationTime + Action  # update the simulation time

                    if TimeArrivalNewTransaction == Action:  # Arrival a new transaction on the system
                        NumTotTransaction = NumTotTransaction + 1  # increase the total number of transaction on system
                        #SimulationTime = SimulationTime + Action  # update the simulation time
                        TimeinStateA = TimeinStateA + Action  # update the time on state A
                        list_arrival_time.append(SimulationTime)  # add the time that the new transaction arrival on the system


                    if TimeDepartureTransaction == Action:  # Departure a transaction on the system
                        NumTotTransaction = NumTotTransaction - 1  # decrease the total number of transaction on system
                        #SimulationTime = SimulationTime + Action  # update the simulation time
                        TimeinStateA = TimeinStateA + Action  # update the time on state A
                        list_departure_time.append(SimulationTime)  # add the time that the transaction departure on the system


                        if len(list_departure_time) != 0 and len(list_arrival_time) != 0:  # checking if there are transactions in the arrivals and departures lists
                            response_time.append(list_departure_time[0] - list_arrival_time[0])  # taking the response time of transaction
                            response_time_final.append(response_time[0])  # add the response time on the list of total response time of system

                            if response_time[0] > threshold:  # check if process is bigger than threshold
                                D = D + 1  # Add a token on the bucket
                                fn += 1
                                FN.append(SimulationTime)

                            else:  # check if process is smaller than threshold
                                D = D - 1  # Remove a token on the bucket

                                if D == -1:  # Avoid D negative
                                    D = 0
                            list_departure_time.pop(0)  # clearing the departure list
                            list_arrival_time.pop(0)  # clearing the arrival list
                            response_time.pop(0)  # clearing the response time list

                    if TimeAGTransition == Action:  # Transition to state G
                        StateMMPP = 'G'  # change to state G
                        #SimulationTime = SimulationTime + Action  # update the simulation time
                        TimeAllTransition = TimeAllTransition + Action  # update total transaction time

        list_TTS.append(NumTotTransaction)
        if StateMMPP == 'G':
            fp += 1
            FP.append(SimulationTime)
            TN.pop()

        if StateMMPP == 'A':
            tp += 1
            TP.append(SimulationTime)
            FN.pop()

    return fp, tp, tn, fn, list_TTS, FP, TP, TN, FN,TimeinStateG, TimeinStateA


mu = 1/6 #0,16
#lambdaA = 1/4 #0,1
lambdaG = 1/12 # 0,083
lambdaGA = 1/15 #0,04
lambdaAG = 1/8 #0,04
threshold = 13.37
depth = 12

A = []
B = []
C = []
D = []
x = []
Tot_Transaction =[]
Time_G = []
Time_A = []
Times_FP = []
Times_TP = []
Times_TN = []
Times_FN = []

for i in range(500):
    lambdaA = (1+i)*0.01
    x.append(lambdaA)
    a, b, c, d, TTS, Times_FP, Times_TP, Times_TN, Times_FN, timeg, timea = bucket_MMPP(mu, lambdaG, lambdaA, lambdaAG, lambdaGA, threshold, depth)
    A.append(a)
    B.append(b)
    C.append(c)
    D.append(d)
    Time_G.append(timeg)
    Time_A.append(timea)
    Tot_Transaction.append(np.mean(TTS))

print('Média de transações no Sistema:', np.mean(Tot_Transaction))
print('Tempo no estado G:', np.mean(Time_G))
print('Tempo no estado A:', np.mean(Time_A))
print('Média de Falso Positivos:', np.mean(A))
print('Média de Verdadeiro Positivos:', np.mean(B))
print('Média de Verdadeiro Negativos:', np.mean(C))
print('Média de Falso Negativo:', np.mean(D))
print('Tempos dos Falso Positivos', Times_FP)
print('Tempos dos Verdadeiros Positivos', Times_TP)
print('Tempos dos Verdadeiros Negativos', Times_TN)
print('Tempos dos Falso Negativos', Times_FN)



'''''''''
plt.plot(x,A)
plt.title("FALSE POSITIVES")
plt.show()

plt.plot(x,B)
plt.title("TRUE POSITIVES")
plt.show()

plt.plot(x,C)
plt.title("TRUE NEGATIVE")
plt.show()

plt.plot(x,D)
plt.title("FALSE NEGATIVE")
plt.show()


#print(B)
#print(C)
#print(D)
'''''