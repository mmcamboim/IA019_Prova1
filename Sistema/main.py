# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 18:31:44 2021

@author: mcamboim
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt


# Transition Matrices ========================================================
Ak = np.array([0.95,1e-1,-9.8e-2,0.80]).reshape(2,2)
Bk = np.array([2,1e-2]).reshape(2,1)
Ck = np.array([1,0,-0.5,1]).reshape(2,2)

# States and Outputs =========================================================
xk = np.array([np.pi/18,0]).reshape(2,1)
yk = np.zeros(2).reshape(2,1)

# Noises =====================================================================
W_STD = 1.0
V_STD = 0.1

# Storage Variables ==========================================================
ITER = 500
xk_t = np.zeros((ITER,xk.shape[0]))
yk_t = np.zeros((ITER,yk.shape[0]))

# Running =====================================================================
for i in range(ITER):
    wk = np.random.normal(0,W_STD,Bk.shape[1]).reshape(-1,1)
    vk = np.random.normal(0,V_STD,yk.shape[0]).reshape(-1,1)

    xk = Ak @ xk + Bk @ wk
    yk = Ck @ xk + vk

    xk_t[i,:] = xk.reshape(-1)
    yk_t[i,:] = yk.reshape(-1)

# Plot =======================================================================
plt.rcParams['axes.linewidth'] = 2.0
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}
plt.rc('font', **font)

plt.figure(figsize=(8,8),dpi=180)
plt.subplot(2,1,1)
plt.plot(xk_t[:,0],c='b',lw=2)
plt.plot(xk_t[:,1],c='r',lw=2)
plt.legend(['$xk_1$','$xk_2$'])
plt.xlim([0,ITER])
plt.grid(True,ls='dotted')
plt.xlabel('(a)\n')
plt.ylabel('xk []')

plt.subplot(2,1,2)
plt.plot(yk_t[:,0],c='b',lw=2)
plt.plot(yk_t[:,1],c='r',lw=2)
plt.legend(['$yk_1$','$yk_2$'])
plt.xlim([0,ITER])
plt.grid(True,ls='dotted')
plt.xlabel('(b)\n Iteração [N]\n')
plt.ylabel('yk []')

plt.tight_layout()

# Saving Generated Data ======================================================
with open(r'C:\Users\mcamboim\Documents\UNICAMP\IA019 - Realização e Predição de Séries Temporais Multivariáveis no Espaço de\Provas\Prova 1\Github\Kalman Filter\system.pickle','wb') as file:
    pickle.dump([Ak,Bk,Ck,xk_t,yk_t,W_STD,V_STD,ITER],file)
    
with open(r'C:\Users\mcamboim\Documents\UNICAMP\IA019 - Realização e Predição de Séries Temporais Multivariáveis no Espaço de\Provas\Prova 1\Github\Kalman Filter Refinement\system.pickle','wb') as file:
    pickle.dump([Ak,Bk,Ck,xk_t,yk_t,W_STD,V_STD,ITER],file)    
