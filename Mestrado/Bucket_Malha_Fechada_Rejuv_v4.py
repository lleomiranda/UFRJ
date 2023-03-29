import heapq as hq
import random
import random as rdm
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


def bucket_MMPP (mu, lambdaG, lambdaA, deltaAG, deltaGA, deltaRG, alpha_O, lambdaR, threshold, depth):

    Final_Tot_Transaction_System = []
    timeinstate_G = 0
    timeinstate_A = 0
    FP = []
    TN = []
    TP = []
    FN = []
    Time_Transition_GA = []
    Arrival_Time_A = []
    Time_Transition_AG = []
    Arrival_Time_G = []
    Time_Transition_RG = []
    Departure_Time = []

    L0 = []
    L1 = []
    Time_Observer = []
    Time_Between_Rejuv = []
    time_between_rejuv = 0
    Rejuvenation = []
    Rejuvenation_Time = []

    simulationtime = 0
    numtottransaction = 0
    stateMMPP = 'G'
    Steps_to_over_flow = 0
    D = 0
    Rejuv_G = 0
    Rejuv_A = 0
    time_off = 0

    Observer_Winer = []
    Observer_Tot = []

    while simulationtime <= 5000:
        #D = 0
        while D <= depth:
            Steps_to_over_flow += 1
            if simulationtime > 5000:
                break

            if stateMMPP == 'G' and numtottransaction == 0 and D == 0:

                time_arrival_newtransaction = random.expovariate(lambdaG)  # return a random number with exponential distribution following lambdaG

                time_GA_transition = random.expovariate(deltaGA)  # return a random number with exponential distribution following lambdaGA

                time_Observer = random.expovariate(alpha_O)
                Observer_Tot.append(time_Observer)

                time_to_next_event = min(time_arrival_newtransaction,time_GA_transition, time_Observer)  # take the smallest of the variables
                simulationtime = simulationtime + time_to_next_event  # update the simulation time
                L0.append(time_to_next_event)
                timeinstate_G += time_to_next_event  # update the time in state G

                if time_arrival_newtransaction == time_to_next_event:  # Arrival a new transaction on the system
                    numtottransaction = numtottransaction + 1  # increase the total number of transaction on system
                    Arrival_Time_G.append(simulationtime)

                if time_GA_transition == time_to_next_event:  # Transition to state A
                    stateMMPP = 'A'  # change the state
                    Time_Transition_GA.append(simulationtime)

                if time_Observer == time_to_next_event:
                    action = random.uniform(0, 1)
                    Observer_Winer.append(time_to_next_event)
                    Time_Observer.append(simulationtime)
                    #simulationtime += time_Observer

                    if action < threshold:  # check if process is bigger than Threshold
                        D = D + 1  # Add a token on the bucket
                        TN.append(simulationtime)


            if stateMMPP == 'G' and numtottransaction == 0 and D > 0:

                time_arrival_newtransaction = random.expovariate(lambdaG)  # return a random number with exponential distribution following lambdaG

                time_GA_transition = random.expovariate(deltaGA)  # return a random number with exponential distribution following lambdaGA

                time_Observer = random.expovariate(alpha_O)
                Observer_Tot.append(time_Observer)

                time_to_next_event = min(time_arrival_newtransaction, time_GA_transition,time_Observer)  # take the smallest of the variables
                simulationtime = simulationtime + time_to_next_event  # update the simulation time
                L0.append(time_to_next_event)
                timeinstate_G += time_to_next_event  # update the time in state G

                if time_arrival_newtransaction == time_to_next_event:  # Arrival a new transaction on the system
                    numtottransaction = numtottransaction + 1  # increase the total number of transaction on system
                    Arrival_Time_G.append(simulationtime)

                if time_GA_transition == time_to_next_event:  # Transition to state A
                    stateMMPP = 'A'  # change the state
                    Time_Transition_GA.append(simulationtime)

                if time_Observer == time_to_next_event:
                    action = random.uniform(0, 1)
                    Observer_Winer.append(time_to_next_event)
                    Time_Observer.append(simulationtime)
                    #simulationtime += time_Observer

                    if action < threshold:  # check if process is bigger than Threshold
                        D = D + 1  # Add a token on the bucket
                        TN.append(simulationtime)

                    else:  # check if process is smaller than Threshold
                        D = D - 1  # Remove a token on the bucket


            if stateMMPP == 'G' and numtottransaction > 0 and D == 0:

                time_arrival_newtransaction = random.expovariate(lambdaG)  # return a random number with exponential distribution following lambdaG

                time_departure_transaction = random.expovariate(mu)  # return a random number with exponential distribution following mu

                time_GA_transition = random.expovariate(deltaGA)  # return a random number with exponential distribution following lambdaGA

                time_Observer = random.expovariate(alpha_O)
                Observer_Tot.append(time_Observer)

                time_to_next_event = min(time_arrival_newtransaction, time_departure_transaction ,time_GA_transition,time_Observer)  # take the smallest of the variables
                simulationtime = simulationtime + time_to_next_event  # update the simulation time
                L1.append(time_to_next_event)
                timeinstate_G += time_to_next_event  # update the time in state G

                if time_arrival_newtransaction == time_to_next_event:  # Arrival a new transaction on the system
                    numtottransaction = numtottransaction + 1  # increase the total number of transaction on system
                    Arrival_Time_G.append(simulationtime)

                if time_departure_transaction == time_to_next_event:  # Departure a transaction on the system
                    numtottransaction = numtottransaction - 1  # decrease the total number of transaction on system
                    Departure_Time.append(simulationtime)

                if time_GA_transition == time_to_next_event:  # Transition to state A
                    stateMMPP = 'A'  # change the state
                    Time_Transition_GA.append(simulationtime)

                if time_Observer == time_to_next_event:
                    action = random.uniform(0, 1)
                    Observer_Winer.append(time_to_next_event)
                    Time_Observer.append(simulationtime)
                    #simulationtime += time_Observer

                    if action < threshold:  # check if process is bigger than Threshold
                        D = D + 1  # Add a token on the bucket
                        TN.append(simulationtime)


            if stateMMPP == 'G' and numtottransaction > 0 and D > 0:

                time_arrival_newtransaction = random.expovariate(lambdaG)  # return a random number with exponential distribution following lambdaG

                time_departure_transaction = random.expovariate(mu)  # return a random number with exponential distribution following mu

                time_GA_transition = random.expovariate(deltaGA)  # return a random number with exponential distribution following lambdaGA

                time_Observer = random.expovariate(alpha_O)
                Observer_Tot.append(time_Observer)

                time_to_next_event = min(time_arrival_newtransaction, time_departure_transaction,time_GA_transition,time_Observer)  # take the smallest of the variables
                simulationtime = simulationtime + time_to_next_event  # update the simulation time
                L1.append(time_to_next_event)
                timeinstate_G += time_to_next_event  # update the time in state G

                if time_arrival_newtransaction == time_to_next_event:  # Arrival a new transaction on the system
                    numtottransaction = numtottransaction + 1  # increase the total number of transaction on system
                    Arrival_Time_G.append(simulationtime)

                if time_departure_transaction == time_to_next_event:  # Departure a transaction on the system
                    numtottransaction = numtottransaction - 1  # decrease the total number of transaction on system
                    Departure_Time.append(simulationtime)

                if time_GA_transition == time_to_next_event:  # Transition to state A
                    stateMMPP = 'A'  # change the state
                    Time_Transition_GA.append(simulationtime)

                if time_Observer == time_to_next_event:
                    action = random.uniform(0, 1)
                    Observer_Winer.append(time_to_next_event)
                    Time_Observer.append(simulationtime)
                    #simulationtime += time_Observer

                    if action < threshold:  # check if process is bigger than Threshold
                        D = D + 1  # Add a token on the bucket
                        TN.append(simulationtime)

                    else:  # check if process is smaller than Threshold
                        D = D - 1  # Remove a token on the bucket


            if stateMMPP == 'A' and numtottransaction == 0 and D == 0:

                time_arrival_newtransaction = random.expovariate(lambdaA)  # return a random number with exponential distribution following lambda A

                time_AG_transition = random.expovariate(deltaAG)  # return a random number with exponential distribution following lambda AG

                time_Observer = random.expovariate(alpha_O)
                Observer_Tot.append(time_Observer)

                time_to_next_event = min(time_arrival_newtransaction, time_AG_transition, time_Observer)  # take the smallest of the variables
                simulationtime = simulationtime + time_to_next_event  # update the simulation time
                L0.append(time_to_next_event)
                timeinstate_A += time_to_next_event  # update the time in state G

                if time_arrival_newtransaction == time_to_next_event:  # Arrival a new transaction on the system
                    numtottransaction = numtottransaction + 1  # increase the total number of transaction on system
                    Arrival_Time_A.append(simulationtime)

                if time_AG_transition == time_to_next_event:  # Transition to state G
                    stateMMPP = 'G'  # change to state G
                    Time_Transition_AG.append(simulationtime)

                if time_Observer == time_to_next_event:
                    action = random.uniform(0, 1)
                    Observer_Winer.append(time_to_next_event)
                    #simulationtime += time_Observer
                    Time_Observer.append(simulationtime)

                    if action < threshold:  # check if process is bigger than Threshold
                        D = D + 1  # Add a token on the bucket
                        FN.append(simulationtime)

            if stateMMPP == 'A' and numtottransaction == 0 and D > 0:

                time_arrival_newtransaction = random.expovariate(lambdaA)  # return a random number with exponential distribution following lambda A

                time_AG_transition = random.expovariate(deltaAG)  # return a random number with exponential distribution following lambda AG

                time_Observer = random.expovariate(alpha_O)
                Observer_Tot.append(time_Observer)

                time_to_next_event = min(time_arrival_newtransaction, time_AG_transition, time_Observer)  # take the smallest of the variables
                simulationtime = simulationtime + time_to_next_event  # update the simulation time
                L0.append(time_to_next_event)
                timeinstate_A += time_to_next_event  # update the time in state G

                if time_arrival_newtransaction == time_to_next_event:  # Arrival a new transaction on the system
                    numtottransaction = numtottransaction + 1  # increase the total number of transaction on system
                    Arrival_Time_A.append(simulationtime)

                if time_AG_transition == time_to_next_event:  # Transition to state G
                    stateMMPP = 'G'  # change to state G
                    Time_Transition_AG.append(simulationtime)

                if time_Observer == time_to_next_event:
                    action = random.uniform(0, 1)
                    Observer_Winer.append(time_to_next_event)
                    #simulationtime += time_Observer
                    Time_Observer.append(simulationtime)

                    if action < threshold:  # check if process is bigger than Threshold
                        D = D + 1  # Add a token on the bucket
                        FN.append(simulationtime)

                    else:  # check if process is smaller than Threshold
                        D = D - 1  # Remove a token on the bucket

            if stateMMPP == 'A' and numtottransaction > 0 and D == 0:

                time_arrival_newtransaction = random.expovariate(lambdaA)  # return a random number with exponential distribution following lambda A

                time_departure_transaction = random.expovariate(mu)  # return a random number with exponential distribution following mu

                time_AG_transition = random.expovariate(deltaAG)  # return a random number with exponential distribution following lambda AG

                time_Observer = random.expovariate(alpha_O)
                Observer_Tot.append(time_Observer)

                time_to_next_event = min(time_arrival_newtransaction, time_departure_transaction, time_AG_transition,time_Observer)  # take the smallest of the variables
                simulationtime = simulationtime + time_to_next_event  # update the simulation time
                L1.append(time_to_next_event)
                timeinstate_A += time_to_next_event  # update the time in state G

                if time_arrival_newtransaction == time_to_next_event:  # Arrival a new transaction on the system
                    numtottransaction = numtottransaction + 1  # increase the total number of transaction on system
                    Arrival_Time_A.append(simulationtime)

                if time_departure_transaction == time_to_next_event:  # Departure a transaction on the system
                    numtottransaction = numtottransaction - 1  # decrease the total number of transaction on system
                    Departure_Time.append(simulationtime)

                if time_AG_transition == time_to_next_event:  # Transition to state G
                    stateMMPP = 'G'  # change to state G
                    Time_Transition_AG.append(simulationtime)

                if time_Observer == time_to_next_event:
                    action = random.uniform(0, 1)
                    Observer_Winer.append(time_to_next_event)
                    #simulationtime += time_Observer
                    Time_Observer.append(simulationtime)

                    if action < threshold:  # check if process is bigger than Threshold
                        D = D + 1  # Add a token on the bucket
                        FN.append(simulationtime)


            if stateMMPP == 'A' and numtottransaction > 0 and D > 0:

                time_arrival_newtransaction = random.expovariate(lambdaA)  # return a random number with exponential distribution following lambda A

                time_departure_transaction = random.expovariate(mu)  # return a random number with exponential distribution following mu

                time_AG_transition = random.expovariate(deltaAG)  # return a random number with exponential distribution following lambda AG

                time_Observer = random.expovariate(alpha_O)
                Observer_Tot.append(time_Observer)

                time_to_next_event = min(time_arrival_newtransaction, time_departure_transaction, time_AG_transition,time_Observer)  # take the smallest of the variables
                simulationtime = simulationtime + time_to_next_event  # update the simulation time
                L1.append(time_to_next_event)
                timeinstate_A += time_to_next_event  # update the time in state G

                if time_arrival_newtransaction == time_to_next_event:  # Arrival a new transaction on the system
                    numtottransaction = numtottransaction + 1  # increase the total number of transaction on system
                    Arrival_Time_A.append(simulationtime)

                if time_departure_transaction == time_to_next_event:  # Departure a transaction on the system
                    numtottransaction = numtottransaction - 1  # decrease the total number of transaction on system
                    Departure_Time.append(simulationtime)

                if time_AG_transition == time_to_next_event:  # Transition to state G
                    stateMMPP = 'G'  # change to state G
                    Time_Transition_AG.append(simulationtime)

                if time_Observer == time_to_next_event:
                    action = random.uniform(0, 1)
                    Observer_Winer.append(time_to_next_event)
                    #simulationtime += time_Observer
                    Time_Observer.append(simulationtime)

                    if action < threshold:  # check if process is bigger than Threshold
                        D = D + 1  # Add a token on the bucket
                        FN.append(simulationtime)

                    else:  # check if process is smaller than Threshold
                        D = D - 1  # Remove a token on the bucket


        Final_Tot_Transaction_System.append(numtottransaction)

        if stateMMPP == 'G':
            FP.append(simulationtime)
            TN.pop()
            Rejuv_G +=1
            #continue

        if stateMMPP == 'A':
            TP.append(simulationtime)
            FN.pop()
            Rejuv_A +=1

        rejuvenation_time = random.expovariate(lambdaR)
        Rejuvenation_Time.append(rejuvenation_time)

        time_RG_transition = random.expovariate(deltaRG)
        Time_Transition_RG.append(time_RG_transition)

        simulationtime += (time_RG_transition + rejuvenation_time)
        time_off += (rejuvenation_time + time_RG_transition)

        Rejuvenation.append(simulationtime)

        D = 0
        numtottransaction = 0
        stateMMPP = 'G'

    #for i in range(len(Rejuvenation) - 1):
    #    Time_Between_Rejuv.append(Rejuvenation[i + 1] - Rejuvenation[i])
    #    time_between_rejuv = np.mean(Time_Between_Rejuv)

    #print(Rejuvenation)
    return Final_Tot_Transaction_System, timeinstate_G, timeinstate_A, len(FP), len(TN), len(TP), len(FN), \
            L0, L1, Time_Observer, Observer_Winer, Observer_Tot, time_off, simulationtime, Steps_to_over_flow, \
            Rejuvenation, Rejuv_G, Rejuv_A, len(Arrival_Time_G), len(Arrival_Time_A), len(Departure_Time), \
            FP, Arrival_Time_G, Arrival_Time_A, Time_Transition_GA, Time_Transition_AG, Rejuvenation_Time, Time_Transition_RG

# System parameters
mu = 1/6
lambdaA = 1/4
lambdaG = 1/12
deltaGA = 1/100
deltaAG = 1/1000
deltaRG = 1/10
#alpha_O = 1/20
lambdaR = 1/10
threshold = 0.46
depth = 12


FTTS = []
Time_G = []
Time_A = []
FP = []
TN = []
TP = []
FN = []
L_0 = []
L_1 = []
Observer = []
Rejuv = []
Inative_Time = []
OW = 0
OT = 0

x=[]

Utilizacao = []
Acuracia = []
Precisao = []

rejuv = []
STOF = []

Time_Off = []
Simulation_Time = []
Time_Off_Relativo = []

Rejuvenation_G =[]
Rejuvenation_A =[]
ATA = []
ATG = []
DT = []

Time_Transition_GA = []
Time_Transition_AG = []
Arrival_Time_G = []
Arrival_Time_A = []
Time_FP = []

TTGA = []
TTAG = []

rdm.seed(datetime.now())
for i in range(500):
    alpha_O = (1+i)*0.01
    #depth = i
    x.append(alpha_O)
    ftts, timeG, timeA, fp, tn, tp, fn, l0, l1, observer, OW, OT, system_off, st, stof, rejuv, rejuv_g, rejuv_a, \
      atg, ata, dt, Time_FP, Arrival_Time_G, Arrival_Time_A, \
        Time_Transition_GA, Time_Transition_AG, RT, Time_Transition_RG = bucket_MMPP(mu, lambdaG, lambdaA, deltaAG, deltaGA, deltaRG, alpha_O, lambdaR, threshold, depth)
    FTTS.append(np.mean(ftts))
    Time_G.append(timeG)
    Time_A.append(timeA)
    FP.append(fp)
    TN.append(tn)
    TP.append(tp)
    FN.append(fn)
    Utilizacao.append(sum(l1)/ (sum(l1) + sum(l0)))
    Acuracia.append((sum(TP) + sum(TN)) / (sum(FN) + sum(TP) + sum(FP) + sum(TN)))
    Precisao.append(sum(TP)/(sum(TP)+sum(FP)))
    Observer.append(observer)
    Time_Off.append(system_off)
    Simulation_Time.append(st)
    Time_Off_Relativo.append(system_off/st)
    STOF.append(stof)
    Rejuv.append(len(rejuv))
    Rejuvenation_G.append(rejuv_g)
    Rejuvenation_A.append(rejuv_a)
    ATG.append(atg)
    ATA.append(ata)
    DT.append(dt)
    TTGA.append(len(Time_Transition_GA))
    TTAG.append(len(Time_Transition_AG))
    #Inative_Time.append(np.mean(inative_time))
    #print(Observer)

#Métrica média das 500 amostras
print('Média de transações no Sistema:', np.mean(FTTS)) #média do total das transações restantes no sistema
print('Tempo médio de permanência no estado G:', np.mean(Time_G))
print('Tempo médio de permanência no estado A:', np.mean(Time_A))
#print('Média de FP', FP)
#print('Média de TN', np.mean(TN))
#print('Média de TP', np.mean(TP))
#print('Média de FN', np.mean(FN))
print('Utilizacao Média do sistema', np.mean(Utilizacao))
#print('Acurácia do Sistema', np.mean(Acuracia))
#print('Precisão Média do Sistema', np.mean(Precisao))
#print(Observer)
print('Número médio de Rejuvenescimentos', np.mean(Rejuv))
#print('Tempo médio de Inatividade', np.mean(Inative_Time))

print('Tempo médio de Simulação',np.mean(Simulation_Time))
print('Tempo médio de Inatividade por rejuv',np.mean(Time_Off))
print('Porcentagem de Inatividade por rejuv',np.mean(Time_Off_Relativo))
print('Média de rejuvenescimentos aplicados em G', np.mean(Rejuvenation_G))
print('Média de rejuvenescimentos aplicados em A', np.mean(Rejuvenation_A))
#print('Média chegada de jobs', (np.mean(ATG)+np.mean(ATA)))
#print('Média partida de jobs', np.mean(DT))

#print(np.mean(RT))
#print(np.mean(Time_Transition_RG))


#print('Transição GA', Time_Transition_GA)
#print('Transição AG', Time_Transition_AG)
#print('chegada_G', Arrival_Time_G)
#print('Chegada A', Arrival_Time_A)

#print('FP', FP)
'''''''''
#maior = max(Time_Transition_GA[len(Time_Transition_GA)-1], Time_Transition_AG[len(Time_Transition_AG)-1])
maior_GA = max(Time_Transition_GA)
maior_AG = max(Time_Transition_AG)
maior = max(maior_GA,maior_AG)
print(maior_GA, maior_AG)
print(maior)

p= Arrival_Time_G[0]
print(p)

List_Detected_Transition_GA = []
List_Detected_Transition_AG = []
while p < maior:
    for i in range(len(Arrival_Time_A)-1):
        if Arrival_Time_A[i] > p:
            p=Arrival_Time_A[i]
            List_Detected_Transition_GA.append(p)
            break
    for j in range(len(Arrival_Time_G)-1):
        if Arrival_Time_G[j] > p:
            p=Arrival_Time_G[j]
            List_Detected_Transition_AG.append(p)
            break


print('Transição GA', Time_Transition_GA)
print('Transição AG', Time_Transition_AG)
print('chegada_G', Arrival_Time_G)
print('Chegada A', Arrival_Time_A)

print('Tempos das transições GA', Time_Transition_GA)
print('Transição detectada de GA', List_Detected_Transition_GA)

print('Tempos das transições AG', Time_Transition_AG)
print('Transição detectada de AG', List_Detected_Transition_AG)

T_0 = 0
N_0 = 0
T_1 = 0
N_1 = 0

if List_Detected_Transition_AG[0] > List_Detected_Transition_GA[0] and len(List_Detected_Transition_AG) == len(List_Detected_Transition_GA):
    for i in range(len(List_Detected_Transition_AG)):
        T_0 += (List_Detected_Transition_AG[i] - List_Detected_Transition_GA[i])
        N_0 += 1

    for i in range(len(List_Detected_Transition_GA) - 1):
        T_1 += (List_Detected_Transition_GA[i + 1] - List_Detected_Transition_AG[i])
        N_1 += 1

if List_Detected_Transition_AG[0] > List_Detected_Transition_GA[0] and len(List_Detected_Transition_AG) != len(List_Detected_Transition_GA):
    for i in range(len(List_Detected_Transition_AG)):
        T_0 += (List_Detected_Transition_AG[i] - List_Detected_Transition_GA[i])
        N_0 += 1

    for i in range(len(List_Detected_Transition_GA) - 1):
        T_1 += (List_Detected_Transition_GA[i + 1] - List_Detected_Transition_AG[i])
        N_1 += 1


print('Instantes de Tempo quando ocorreu FP', Time_FP)
print('Num de FP', len(Time_FP))
print(N_1, N_0)

print('Taxa de transição de GA', N_0/T_0, "----", deltaGA)
print('Taxa de transição de AG', N_1/T_1, "----", deltaAG)


print('TTGA',np.mean(TTGA))
print('TTAG',np.mean(TTAG))

'''
plt.rcParams.update({'font.size': 18})

plt.plot(x,Time_A)
plt.xlabel('θ')
plt.ylabel('Tempo')
plt.show()

#plt.plot(x,Rejuv)
#plt.xlabel('depth of bucket')
#plt.ylabel('frequency')
#plt.show()

#plt.hist(Time_Off_Relativo,50)
#plt.xlabel('Percentual de Inatividade \n por rejuvenescimento')
#plt.ylabel('Frequência')
#plt.show()

#plt.hist(STOF,50)
#plt.xlabel('Passos até o estouro do bucket')
#plt.ylabel('Frequência')
#plt.show()

#plt.hist(Rejuv,50)
#plt.xlabel('Rejuvenescimentos')
#plt.ylabel('Frequência')
#plt.show()


