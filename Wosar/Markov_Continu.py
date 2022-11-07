import numpy as np
import pandas as pd
import scipy.stats as ss
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt



def get_transition_matrix(
        r1,
        r_1,
        r2,
        r_2,
        r3,
        r4,
        lambd
):

    d1 = -r2-r_1
    d2 = -r_2-r2-r4-lambd
    d3 = -r_2-r4-lambd
    d4 = -r3
    d5 = -r1

    Q = np.array([[d1,r2,0,0,r_1],
                  [r_2,d2,r2,r4,lambd],
                  [0, r_2, d3, r4, lambd],
                  [r3,0,0,d4,0],
                  [r1,0,0,0,d5]])
    return Q

def is_steady_state(state, Q):

    return np.allclose((state @ Q), 0)


def obtain_steady_state_with_matrix_exponential(Q, max_t=100):
    """
      Solve the defining differential equation until it converges.
      - Q: the transition matrix
      - max_t: the maximum time for which the differential equation is solved at each attempt.
      """
    dimension = Q.shape[0]
    state = np.ones(dimension) / dimension

    while not is_steady_state(state=state, Q=Q):
        state = state @ sp.linalg.expm(Q * max_t)

    return state

QQ = get_transition_matrix(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
#QQ1 = get_transition_matrix(0.1, 0.01, 0.5, 0.7, 0.1, 0.07, 0.07)
#QQ2 = get_transition_matrix(0.1, 0.01, 0.5, 0.7, 0.1, 0.05, 0.05)
#print(QQ)
v = obtain_steady_state_with_matrix_exponential(QQ)
#v1 = obtain_steady_state_with_matrix_exponential(QQ1)
#v2 = obtain_steady_state_with_matrix_exponential(QQ2)
print('Same rate',v)
#print('Same lambd/r4',v1)
#print('Diff lambd/r4',v2)
print(is_steady_state(v,QQ))
#print(is_steady_state(v1,QQ1))
#print(is_steady_state(v2,QQ2))

t = np.linspace(0,60)
#print(t)
#"=======================Estado Inicial=======================================")
v_inicial = np.transpose([1,0,0,0,0])
tot_inicial = []
inop_i =[]
for i in range(len(t)):
    p = v_inicial.dot(sp.linalg.expm(QQ * t[i]))
    tot_inicial.append(p)
np.array(tot_inicial)
df_inicial = pd.DataFrame(tot_inicial)

'''''''''
plt.plot(t, df_inicial[0], label='0')
plt.plot(t, df_inicial[1], label='1')
plt.plot(t, df_inicial[2], label='2')
plt.plot(t, df_inicial[3], label='Reju')
plt.plot(t, df_inicial[4], label='Ataque')
plt.xlabel('Time', fontsize= 24)
plt.ylabel('State Probability', fontsize= 24)
plt.legend(fontsize= 24)
plt.tick_params(labelsize=24)
plt.show()
'''
#"=======================Estado 1======================================="
tot_1 = []
v_1 = np.transpose([0,1,0,0,0])
for i in range(len(t)):
    p_1 = v_1.dot(sp.linalg.expm(QQ * t[i]))
    tot_1.append(p_1)
np.array(tot_1)
df_1 = pd.DataFrame(tot_1)
'''''''''
plt.plot(t, df_1[0], label='0')
plt.plot(t, df_1[1], label='1')
plt.plot(t, df_1[2], label='2')
plt.plot(t, df_1[3], label='Rejuv')
plt.plot(t, df_1[4], label='Ataque')
plt.xlabel('Time', fontsize= 24)
plt.ylabel('State Probability', fontsize= 24)
plt.legend(fontsize= 24)
plt.tick_params(labelsize=24)
plt.show()
'''
#"=======================Estado 2======================================="
tot_2 = []
v_2 = np.transpose([0,0,1,0,0])
for i in range(len(t)):
    p_2 = v_2.dot(sp.linalg.expm(QQ * t[i]))
    tot_2.append(p_2)
np.array(tot_2)
df_2 = pd.DataFrame(tot_2)
'''''''''
plt.plot(t, df_2[0], label='0')
plt.plot(t, df_2[1], label='1')
plt.plot(t, df_2[2], label='2')
plt.plot(t, df_2[3], label='Rejuv')
plt.plot(t, df_2[4], label='Ataque')
plt.xlabel('Time', fontsize= 24)
plt.ylabel('State Probability', fontsize= 24)
plt.legend(fontsize= 24)
plt.tick_params(labelsize=24)
plt.show()
'''

#"=======================Rejuv======================================="
v_R = np.transpose([0,0,0,1,0])
tot_rejuv = []
for i in range(len(t)):
    p_R = v_R.dot(sp.linalg.expm(QQ * t[i]))
    tot_rejuv.append(p_R)
np.array(tot_rejuv)
df_rejuv = pd.DataFrame(tot_rejuv)
'''''''''
plt.plot(t, df_rejuv[0], label='0')
plt.plot(t, df_rejuv[1], label='1')
plt.plot(t, df_rejuv[2], label='2')
plt.plot(t, df_rejuv[3], label='Rejuv')
plt.plot(t, df_rejuv[4], label='Ataque')
plt.xlabel('Time', fontsize= 24)
plt.ylabel('State Probability', fontsize= 24)
plt.legend(fontsize= 24)
plt.tick_params(labelsize=24)
plt.show()
'''''
#"=======================Ataque======================================="
v_A = np.transpose([0,0,0,0,1])
tot_Ataque = []
for i in range(len(t)):
    p_A = v_A.dot(sp.linalg.expm(QQ * t[i]))
    tot_Ataque.append(p_A)
np.array(tot_Ataque)
df_Ataque = pd.DataFrame(tot_Ataque)
'''''''''
plt.plot(t, df_Ataque[0], label='0')
plt.plot(t, df_Ataque[1], label='1')
plt.plot(t, df_Ataque[2], label='2')
plt.plot(t, df_Ataque[3], label='Rejuv')
plt.plot(t, df_Ataque[4], label='Ataque')
plt.xlabel('Time', fontsize=24)
plt.ylabel('State Probability', fontsize=24)
plt.legend(fontsize= 24)
plt.tick_params(labelsize=24)
plt.show()


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
#fig.suptitle('Horizontally stacked subplots')
ax1.plot(t, df_inicial[0])
ax1.plot(t, df_inicial[1])
ax1.plot(t, df_inicial[2])
ax1.plot(t, df_inicial[3])
ax1.plot(t, df_inicial[4])
ax2.plot(t, df_1[0])
ax2.plot(t, df_1[1])
ax2.plot(t, df_1[2])
ax2.plot(t, df_1[3])
ax2.plot(t, df_1[4])
ax3.plot(t, df_2[0])
ax3.plot(t, df_2[1])
ax3.plot(t, df_2[2])
ax3.plot(t, df_2[3])
ax3.plot(t, df_2[4])
ax4.plot(t, df_rejuv[0])
ax4.plot(t, df_rejuv[1])
ax4.plot(t, df_rejuv[2])
ax4.plot(t, df_rejuv[3])
ax4.plot(t, df_rejuv[4])
#ax5.plot(t, df_Ataque[0])
#ax5.plot(t, df_Ataque[1])
#ax5.plot(t, df_Ataque[2])
#ax5.plot(t, df_Ataque[3])
#ax5.plot(t, df_Ataque[4])
#plt.legend(fontsize= 24)
#plt.tick_params(ax1, ax2,ax3,ax4,labelsize=24)
plt.show()
'''''
r4 = []
Ur4 = []
piA_r4 = []
piR_r4 = []
for i in range(5000):
    Qr4 = get_transition_matrix(1, 0.01, 0.1, 0.1, 10, i*0.01, 0.05)
    v_eq= obtain_steady_state_with_matrix_exponential(Qr4)
    r4.append(i*0.01)
    Ur4.append((v_eq[3]) + (v_eq[4]))
    piR_r4.append(v_eq[3])
    piA_r4.append(v_eq[4])
#print(r4)
#print(Ur4)
print(piR_r4)
print(piA_r4)

plt.plot(r4, Ur4, label='πa+πr')
plt.plot(r4,piA_r4, label='πa')
plt.plot(r4,piR_r4, label='πr')
plt.xlabel('R4', fontsize=35)
plt.ylabel('Probability', fontsize=35)
plt.grid()
plt.legend(fontsize=30)
plt.tick_params(labelsize=35)
plt.show()
'''''''''
inop_i = []
ri_inop = []
inop_1 = []
inop_2 = []
inop_r = []
inop_a = []
for i in range(len(df_inicial)):
    inop_i.append((df_inicial[3][i]) + (df_inicial[4][i]))
    inop_1.append((df_1[3][i]) + (df_1[4][i]))
    inop_2.append((df_2[3][i]) + (df_2[4][i]))
    inop_r.append((df_rejuv[3][i]) + (df_rejuv[4][i]))
    inop_a.append((df_Ataque[3][i]) + (df_Ataque[4][i]))

plt.plot(t, inop_i, label='inop_i')
#plt.plot(t, inop_1, label='inop_1')
#plt.plot(t, inop_2, label='inop_2')
#plt.plot(t, inop_r, label='inop_r')
#plt.plot(t, inop_a, label='inop_a')
plt.legend()
plt.show()

plt.plot(t, df_inicial[3], label='inop_r')
plt.plot(t, df_inicial[4], label='inop_a')
plt.legend()
plt.show()
'''''

#(1, 0.01, 0.1, 0.01, 5, i*0.01, 0.02)
#(0.5, 0.01, 0.1, 0.01, 0.5, i*0.01, 0.05)
#(0.1, 0.1, 0.1, 0.1, 0.1, i*0.01, 0.1)
#(1, 0.05, 0.1, 0.05, 1, i*0.01, 0.05)