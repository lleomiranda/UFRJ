# importação das Bibliotecas
import math
import random
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import scipy.stats as ss


x = range(1,29,1)
Simula = (5.435454,10.458536,16.80069,24.323132,33.111801,42.3991,52.685159,63.740594,75.540224,87.685359,100.063496,113.373247,127.381238
,140.746555,155.048085,169.562516,184.578448,199.591489,214.537524,229.947475,245.657866,261.924082,276.106561,292.568077
,308.689069,323.379788,340.585819,357.314331)

GS = (5.446778213,10.49053917,16.85010077,24.37650446,32.9376549,42.41641095,52.70889273,63.72298034,75.37698257,87.59845624,100.3231593,113.4941224
,127.0608255,140.9784679,155.2073206,169.7121522,184.4617199,199.4283177,214.587376,229.917107,245.3981893,261.0134886,276.7478106,292.5876811
,308.5211512,324.5376246,340.6277049,356.783059)

Reju_9 = (8.17,19.3917,40.3141,75.8624,138.5335,247.705,432.8554,764.5207,1349.4621,2390.3672,4138.3037
,7226.6336,12610.2448,22236.4164,37331.3457,65370.9466,115277.8925,199482.3302,354071.4756,612667.7214
,1077634.067,1791836.771,3222581.471,5661958.244,9349992.235,16850849.22,29122293.33,48817686.99)

Reju_91 = (8.382,18.921,37.393,69.151,127.627,225.539,402.795,632.94,1119.21,1939.591,3249.405,5513.119
,9335.482,16045.255,27626.916,45627.246,78864.927,141212.271,221415.421,389333.329,661312.438,1062841.148
,1947642.438,3159982.948,5218949.573,9065225.818,16177674.64,26437286.25)

Reju_92 = (7.959,17.869,36.735,65.346,121.725,202.951,329.205,560.324,956.872,1588.833,2595.211
,4343.224,7136.04,11803.742,19279.461,32318.956,55561.804,93819.183,142238.898,256753.167,391732.048
,652840.289,1066627.599,1801662.232,2881095.882,4838495.296,8203196.249,12907343.28)

Reju_93 = (7.693,17.541,33.092,62.464,106.855,171.258,294.407,488.906,789.307,1285.963,2071.441
,3555.104,5293.358,8208.857,14209.164,22414.774,36233.038,56430.483,98003.382,153175.075,230629.236
,367012.469,622374.14,972602.424,1578910.005,2518023.844,4014552.929,6619933.745)

Reju_94 = (7.359,17.567,33.858,58.347,105.648,174.43,264.345,422.985,696.915,1083.119,1666.509,2569.247
,3876.961,6108.492,10235.597,15611.102,22615.702,37507.976,56839.148,89790.959,138440.903,230799.046,339613.603
,525158.781,817140.238,1338740.135,2195829.3,3047693.068)

Reju_95 = (7.673,16.756,32.533,56.427,92.834,150.858,237.011,364.643,568.089,865.277,1339.429
,1985.877,2875.394,4364.342,7156.256,10535.552,16082.227,23682.935,34641.939,55221.141,85643.29
,117734.163,182188.101,281107.086,438521.61,643263.376,968217.019,1492042.493)

Reju_96 = (7.382,15.953,30.198,48.99,85.255,133.207,195.814,310.637,443.693,682.992,1031.828,1530.268,2230.995
,3150.607,4726.061,7083.099,9981.162,14526.687,21751.579,30504.528,45462.81,67914.384,103155.211,140341.177
,208997.932,292831.957,449922.126,648146.904)

Reju_97 = (7.274,16.509,30.537,49.527,82.182,123.985,173.064,265.1,382.245,518.803,814.577,1117.215
,1620.337,2238.515,3232.586,4588.088,5959.955,9286.9,12879.436,17143.478,23537.923
,34113.647,50306.611,68176.073,94469.403,135631.119,194206.215,284073.814)

Reju_98 = (7.1484, 15.4373, 27.8197, 46.6118, 71.8894, 106.9733, 155.4935,224.4175, 314.3871, 437.9404
, 598.7475, 821.0079, 1139.017, 1523.22, 2092.5409, 2748.8782,3784.153, 5154.7666
, 6830.4071, 9478.3713, 12200.0345, 16903.9505, 23120.748, 30792.335,41529.4048, 56638.8078, 74530.7701, 101177.7708)

Reju_99 = (7.02172, 14.99599, 26.89201, 43.66623, 66.55378, 97.36456, 137.53675,189.50498, 257.91195, 346.20698, 459.16799, 601.36478
, 786.17561, 1022.23828, 1334.42993, 1712.80689, 2204.77487,2843.89253, 3618.95423, 4637.45267, 5956.83116, 7672.50338,
9764.0386, 12491.01406,15879.92959, 20337.74713, 25936.01868, 33321.68415)


Reju_exp_9 = np.polyfit(x, np.log(Reju_9),1)
print(Reju_exp_9)
Reju_exp_91 = np.polyfit(x, np.log(Reju_91),1)
print(Reju_exp_91)
Reju_exp_92 = np.polyfit(x, np.log(Reju_92),1)
print(Reju_exp_92)
Reju_exp_93 = np.polyfit(x, np.log(Reju_93),1)
print(Reju_exp_93)
Reju_exp_94 = np.polyfit(x, np.log(Reju_94),1)
print(Reju_exp_94)
Reju_exp_95 = np.polyfit(x, np.log(Reju_95),1)
print(Reju_exp_95)
Reju_exp_96 = np.polyfit(x, np.log(Reju_96),1)
print(Reju_exp_96)
Reju_exp_97 = np.polyfit(x, np.log(Reju_97),1)
print(Reju_exp_97)
Reju_exp_98 = np.polyfit(x, np.log(Reju_98),1)
print(Reju_exp_98)
Reju_exp_99 = np.polyfit(x, np.log(Reju_99),1)
print(Reju_exp_99)


param_exp_9 = [math.e**Reju_exp_9[1], math.e**Reju_exp_9[0]]
print('parametros exp 90', param_exp_9)
param_exp_91 = [math.e**Reju_exp_91[1], math.e**Reju_exp_91[0]]
print('parametros exp 91', param_exp_91)
param_exp_92 = [math.e**Reju_exp_92[1], math.e**Reju_exp_92[0]]
print('parametros exp 92', param_exp_92)
param_exp_93 = [math.e**Reju_exp_93[1], math.e**Reju_exp_93[0]]
print('parametros exp 93', param_exp_93)
param_exp_94 = [math.e**Reju_exp_94[1], math.e**Reju_exp_94[0]]
print('parametros exp 94', param_exp_94)
param_exp_95 = [math.e**Reju_exp_95[1], math.e**Reju_exp_95[0]]
print('parametros exp 95', param_exp_95)
param_exp_96 = [math.e**Reju_exp_96[1], math.e**Reju_exp_96[0]]
print('parametros exp 96', param_exp_96)
param_exp_97 = [math.e**Reju_exp_97[1], math.e**Reju_exp_97[0]]
print('parametros exp 97', param_exp_97)
param_exp_98 = [math.e**Reju_exp_98[1], math.e**Reju_exp_98[0]]
print('parametros exp 98', param_exp_98)
param_exp_99 = [math.e**Reju_exp_99[1], math.e**Reju_exp_99[0]]
print('parametros exp 99', param_exp_99)



#plt.title('Reju_95')
#plt.xlabel('Passos até o estouro do Bucket')
#plt.ylabel('Frequencia')

'''''''''
#rX = np.linspace(0,29)
rP = np.exp(x*Reju_exp_95[0]+Reju_exp_95[1])
print('rP', rP)
#print(rP)
#print(rX)
#plt.hist(Reju_95, bins=30, density=True)
plt.title("Distribuição Exponencial")
plt.xlabel('Passos até o estouro do Bucket')
plt.ylabel('Frequencia')
plt.plot(x,rP, color='blue',label='Exp')
plt.plot(x,Reju_95, color='green',label='Reju_5%')
plt.legend()
#plt.show()
'''


depths = np.linspace(1, 28, 28)
analitico9 = 14.1693/(0.46*0.90)**(0.60977*depths)
analitico91 = 9.96429/(0.46*0.91)**(0.606878*depths)
analitico92 = 13.9233/(0.46*0.92)**(0.570972*depths)
analitico93 = 9.33475/(0.46*0.93)**(0.566466*depths)
analitico94 = 20.0881/(0.46*0.94)**(0.509257*depths)
analitico95 = 12.7366/(0.46*0.95)**(0.503363*depths)
analitico96 = 15.5557/(0.46*0.96)**(0.464842*depths)
analitico97 = 12.7911/(0.46*0.97)**(0.44252*depths)
analitico98 = 24.7297/(0.46*0.98)**(0.37278*depths)
analitico99 = 33.7894/(0.46*0.99)**(0.312959*depths)
#print('Analitico', analitico)


#plt.style.use('ggplot')
#fig1 = plt.figure(figsize=(13, 13))
#plt.axis([1, 29, 1, 1000000])
plt.plot(x,analitico9, 'g^')
plt.plot(x,Reju_9, color='k',label='10%', linewidth=3)
#plt.plot(x,analitico91, 'g^')
#plt.plot(x,Reju_91, color='grey',label='9%', linewidth=3)
plt.plot(x,analitico92, 'g^')
plt.plot(x,Reju_92, color='grey',label='8%', linewidth=3)
#plt.plot(x,analitico93, 'g^')
#plt.plot(x,Reju_93, color='y',label='7%', linewidth=3)
plt.plot(x,analitico94, 'g^')
plt.plot(x,Reju_94, color='m',label='6%', linewidth=3)
#plt.plot(x,analitico95, 'g^')
#plt.plot(x,Reju_95, color='k',label='Reju_5%')
plt.plot(x,analitico96, 'g^')
plt.plot(x,Reju_96, color='r',label='4%', linewidth=3)
#plt.plot(x,analitico97, 'g^')
#plt.plot(x,Reju_97, color='b',label='Reju_3%')
plt.plot(x,analitico98, 'g^')
plt.plot(x,Reju_98, color='y',label='2%', linewidth=3)
#plt.plot(x,analitico99, 'g^')
#plt.plot(x,Reju_99, color='brown',label='Reju_1%')

plt.plot(x, Simula, color='b', label='0%', linewidth=4)
plt.grid()
plt.yscale("log", basey=10)
plt.xscale("log", basex=2)
plt.xlabel("depth of bucket",fontsize=35)
plt.ylabel("mean steps until \n false alarm",fontsize=35)
plt.legend(fontsize=25, frameon=True)
plt.tick_params(labelsize=35)
plt.show()






