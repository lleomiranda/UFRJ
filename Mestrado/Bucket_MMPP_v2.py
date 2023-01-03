import heapq as hq
import random
import random as rdm
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


def bucket_MMPP(mu, lambdaG, lambdaA, deltaAG, deltaGA, threshold, depth):

    simulationtime = 0
    # System parameters
    timeinstate_G = 0
    timeinstate_A = 0
    timealltransition = 0

    List_Arrival_Time = []
    List_Departure_Time = []
    Response_Time = []
    Response_Time_Final = []
    List_TTS = []

    FP = []
    TP= []
    TN = []
    FN = []
    L0 = []
    L1 = []

    fp = 0
    tp = 0
    tn = 0
    fn = 0
    stateMMPP = 'G'

    while simulationtime <= 5000:
        # State variable
        D = 0
        numtottransaction = 0

        while D <= depth:

            if stateMMPP == 'G':

                if numtottransaction == 0:  # no transaction on the system in state G

                    time_arrival_newtransaction = random.expovariate(lambdaG)  # return a random number with exponential distribution following lambdaG

                    time_GA_transition = random.expovariate(deltaGA)  # return a random number with exponential distribution following lambdaGA

                    time_to_next_event = min(time_arrival_newtransaction, time_GA_transition)  # take the smallest of the variables
                    simulationtime = simulationtime + time_to_next_event  # update the simulation time
                    L0.append(time_to_next_event)
                    timeinstate_G += time_to_next_event  # update the time in state G


                    if time_arrival_newtransaction == time_to_next_event:  # Arrival a new transaction on the system
                        numtottransaction = numtottransaction + 1  # increase the total number of transaction on system
                        List_Arrival_Time.append(simulationtime)  # add the time that the new transaction arrival on the system

                    if time_GA_transition == time_to_next_event:  # Transition to state A
                        stateMMPP = 'A'  # change the state
                        timealltransition = timealltransition + time_to_next_event  # update the total transition time


                if numtottransaction > 0:  # There are 1 ou more transaction on the system in state G

                    time_arrival_newtransaction = random.expovariate(lambdaG)  # return a random number with exponential distribution following lambdaG

                    time_departure_transaction = random.expovariate(mu)  # return a random number with exponential distribution following mu

                    time_GA_transition = random.expovariate(deltaGA)  # return a random number with exponential distribution following lambdaGA

                    time_to_next_event = min(time_arrival_newtransaction, time_departure_transaction,time_GA_transition)  # take the smallest of the variables
                    simulationtime = simulationtime + time_to_next_event  # update the simulation time
                    L1.append(time_to_next_event)
                    timeinstate_G += time_to_next_event  # update the time in state G


                    if time_arrival_newtransaction == time_to_next_event:  # Arrival a new transaction on the system
                        numtottransaction = numtottransaction + 1  # increase the total number of transaction on system
                        List_Arrival_Time.append(simulationtime)  # add the time that the new transaction arrival on the system

                    if time_departure_transaction == time_to_next_event:  # Departure a transaction on the system
                        numtottransaction = numtottransaction - 1  # decrease the total number of transaction on system
                        List_Departure_Time.append(simulationtime)  # add the time that the transaction leave on the system

                        if len(List_Departure_Time) != 0 and len(List_Arrival_Time) != 0:  # checking if there are transactions in the arrivals and departures lists
                            Response_Time.append(List_Departure_Time[0] - List_Arrival_Time[0])  # taking the response time of transaction
                            Response_Time_Final.append(Response_Time[0])  # add the response time on the list of total response time of system

                            if Response_Time[0] > threshold:  # check if process is bigger than Threshold
                                D = D + 1  # Add a token on the bucket
                                tn += 1
                                TN.append(simulationtime)

                            else:  # check if process is smaller than Threshold
                                D = D - 1  # Remove a token on the bucket

                                if D == -1:  # Avoid D negative
                                    D = 0
                            List_Departure_Time.pop(0)  # clearing the departure list
                            List_Arrival_Time.pop(0)  # clearing the arrival list
                            Response_Time.pop(0)  # clearing the response time list

                    if time_GA_transition == time_to_next_event:  # Transition to state A
                        stateMMPP = 'A'  # change to state A
                        timealltransition = timealltransition + time_to_next_event  # update total transaction time


            if stateMMPP == 'A':

                if numtottransaction == 0:  # no transaction on the system in state G

                    time_arrival_newtransaction = random.expovariate(lambdaA)  # return a random number with exponential distribution following lambda A

                    time_AG_transition = random.expovariate(deltaAG)  # return a random number with exponential distribution following lambda AG

                    time_to_next_event = min(time_arrival_newtransaction, time_AG_transition)  # take the smallest of the variables
                    simulationtime = simulationtime + time_to_next_event  # update the simulation time
                    L0.append(time_to_next_event)
                    timeinstate_A += time_to_next_event  # update the time in state G

                    if time_arrival_newtransaction == time_to_next_event:  # Arrival a new transaction on the system
                        numtottransaction = numtottransaction + 1  # increase the total number of transaction on system
                        List_Arrival_Time.append(simulationtime)  # add the time that the new transaction arrival on the system

                    if time_AG_transition == time_to_next_event:  # Transition to state G
                        stateMMPP = 'G'  # change to state G
                        timealltransition = timealltransition + time_to_next_event  # update total transaction time


                if numtottransaction > 0:  # There are 1 ou more transaction on the system in state G

                    time_arrival_newtransaction = random.expovariate(lambdaA)     # return a random number with exponential distribution following lambda A

                    time_departure_transaction = random.expovariate(mu)  # return a random number with exponential distribution following mu

                    time_AG_transition = random.expovariate(deltaAG)  # return a random number with exponential distribution following lambda AG

                    time_to_next_event = min(time_arrival_newtransaction, time_departure_transaction,time_AG_transition)  # take the smallest of the variables
                    simulationtime = simulationtime + time_to_next_event  # update the simulation time
                    L1.append(time_to_next_event)
                    timeinstate_A += time_to_next_event  # update the time in state G

                    if time_arrival_newtransaction == time_to_next_event:  # Arrival a new transaction on the system
                        numtottransaction = numtottransaction + 1  # increase the total number of transaction on system
                        List_Arrival_Time.append(simulationtime)  # add the time that the new transaction arrival on the system

                    if time_departure_transaction == time_to_next_event:  # Departure a transaction on the system
                        numtottransaction = numtottransaction - 1  # decrease the total number of transaction on system
                        List_Departure_Time.append(simulationtime)  # add the time that the transaction departure on the system

                        if len(List_Departure_Time) != 0 and len(List_Arrival_Time) != 0:  # checking if there are transactions in the arrivals and departures lists
                            Response_Time.append(List_Departure_Time[0] - List_Arrival_Time[0])  # taking the response time of transaction
                            Response_Time_Final.append(Response_Time[0])  # add the response time on the list of total response time of system

                            if Response_Time[0] > threshold:  # check if process is bigger than threshold
                                D = D + 1  # Add a token on the bucket
                                fn += 1
                                FN.append(simulationtime)

                            else:  # check if process is smaller than threshold
                                D = D - 1  # Remove a token on the bucket

                                if D == -1:  # Avoid D negative
                                    D = 0
                            List_Departure_Time.pop(0)  # clearing the departure list
                            List_Arrival_Time.pop(0)  # clearing the arrival list
                            Response_Time.pop(0)  # clearing the response time list

                    if time_AG_transition == time_to_next_event:  # Transition to state G
                        stateMMPP = 'G'  # change to state G
                        timealltransition = timealltransition + time_to_next_event  # update total transaction time


        List_TTS.append(numtottransaction)
        if stateMMPP == 'G':
            fp += 1
            FP.append(simulationtime)
            TN.pop()

        if stateMMPP == 'A':
            tp += 1
            TP.append(simulationtime)
            FN.pop()

    return fp, tp, tn, fn, List_TTS, FP, TP, TN, FN, timeinstate_G, timeinstate_A, Response_Time_Final, L0, L1


mu = 1/6 #0,16
#lambdaA = 1/10
lambdaG = 1/12
deltaGA = 1/15
deltaAG = 1/15
threshold = 13.37
depth = 12

List_FP = []
List_TP = []
List_TN = []
List_FN = []
x = []
Tot_Transaction =[]
Time_G = []
Time_A = []
Times_FP = []
Times_TP = []
Times_TN = []
Times_FN = []
RTF = []
Utilizacao = []
Acuracia = []
Precisao = []
Recall = []
F1 = []

rdm.seed(datetime.now())
for i in range(500):
    lambdaA = (1+i)*0.01
    x.append(lambdaA)
    num_fp, num_tp, num_tn, num_fn, TTS, Times_FP, Times_TP, Times_TN, Times_FN, timeg, timea, \
    rtf, load_0, load_1 = bucket_MMPP(mu, lambdaG, lambdaA, deltaAG, deltaGA, threshold, depth)
    List_FP.append(num_fp)
    List_TP.append(num_tp)
    List_TN.append(num_tn)
    List_FN.append(num_fn)
    Time_G.append(timeg)
    Time_A.append(timea)
    Tot_Transaction.append(np.mean(TTS))
    RTF.append(np.mean(rtf))
    Utilizacao.append(sum(load_1)/(sum(load_1)+sum(load_0)))
    Acuracia.append((len(Times_TP) + len(Times_TN))/ (len(Times_FN) + len(Times_TP)+
                    len(Times_FP) + len(Times_TN)))
    Precisao.append(len(Times_TP)/(len(Times_TP)+len(Times_FP)))

    #if len(Times_TP) and len(Times_FN) != 0:
    Recall.append(len(Times_TP)/(len(Times_TP)+len(Times_FN)))

    #if len(Times_TP)/(len(Times_TP)+len(Times_FP)) + len(Times_TP)/(len(Times_TP)+len(Times_FN)) != 0:
    #    F1.append(2*(len(Times_TP)/(len(Times_TP)+len(Times_FP)))*(len(Times_TP)/(len(Times_TP)+len(Times_FN)))/
    #    ((len(Times_TP)/(len(Times_TP)+len(Times_FP)))+(len(Times_TP)/(len(Times_TP)+len(Times_FN)))))


#Métricas
print('Média de transações no Sistema:', np.mean(Tot_Transaction)) #média do total das transações restantes no sistema das 500 amostras
print('Tempo no estado G:', np.mean(Time_G)) #média de tempo em G das 500 amostras
print('Tempo no estado A:', np.mean(Time_A)) #média de tempo em A das 500 amostras
print('Média de Falso Positivos:', np.mean(List_FP)) #média de falso positivos das 500 amostras
print('Média de Verdadeiro Positivos:', np.mean(List_TP)) #média de verdadeiro positivos das 500 amostras
print('Média de Verdadeiro Negativos:', np.mean(List_TN)) #média de verdadeiro negativos das 500 amostras
print('Média de Falso Negativo:', np.mean(List_FN)) #média de falso negativos das 500 amostras
print('Tempo médio de resposta final', np.mean(RTF))
print('Utilização Média', np.mean(Utilizacao))
print('Acurácia Média', np.mean(Acuracia))
print('Precisão Médio', np.mean(Precisao))
print('Recall Médio', np.mean(Recall))
#print('F1 Médio', np.mean(F1))

#Tempos que ocorreram os FP,TP,TN,FN
print('Tempos dos Falso Positivos', Times_FP) #Tempos que ocorreram falso positivos
print('Tempos dos Verdadeiros Positivos', Times_TP) #Tempos que ocorreram verdadeiro positivos
print('Tempos dos Verdadeiros Negativos', Times_TN) #Tempos que ocorreram verdadeiro negativos
print('Tempos dos Falso Negativos', Times_FN) #Tempos que ocorreram falso negativos

plt.rcParams.update({'font.size': 18})

plt.hist(Utilizacao, 100)
plt.xlabel('Utilização do Sistema variando λA')
plt.ylabel('Frequência')
plt.show()

plt.hist(Acuracia, 100)
plt.xlabel('Acurácia do Sistema variando λA')
plt.ylabel('Frequência')
plt.show()

plt.hist(Precisao, 100)
plt.xlabel('Precisão do Sistema variando λA')
plt.ylabel('Frequência')
plt.show()

plt.hist(Recall, 100)
plt.xlabel('Recall do Sistema variando λA')
plt.ylabel('Frequência')
plt.show()

#plt.hist(F1, 100)
#plt.show()

plt.plot(x,List_FP)
plt.xlabel('λA')
plt.ylabel('Falso Positivos')
plt.show()

plt.plot(x,List_TP)
plt.xlabel('λA')
plt.ylabel('Verdadeiro Positivos')
plt.show()

plt.plot(x,List_TN)
plt.xlabel('λA')
plt.ylabel('Verdadeiro Negativos')
plt.show()

plt.plot(x,List_FN)
plt.xlabel('λA')
plt.ylabel('Falso Negativos')
plt.show()
