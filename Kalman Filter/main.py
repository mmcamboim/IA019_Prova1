# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 19:15:23 2021

@author: mcamboim
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

from kalman_func import KalmanFilter

# Loading System Data ========================================================
with open('system.pickle','rb') as file:
    [Ak,Bk,Ck,xk_t,yk_t,W_STD,V_STD,ITER] = pickle.load(file)

# Initial Variables ==========================================================
Pk0 = 1.0
xk0 = np.array([np.pi/18,0]).reshape(2,1)
kf = KalmanFilter(Ak,Bk,Ck,W_STD,V_STD,ITER,Pk0,xk0)

# Running ====================================================================
for i in range(ITER):
    kf.RunKfOneStepAhead(yk_t[i].reshape(2,1))

# Plotting ===================================================================
plt.rcParams['axes.linewidth'] = 2.0
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}
plt.rc('font', **font)

# States ---------------------------------------------------------------------
plt.figure(figsize=(8,8),dpi=180)
plt.subplot(2,1,1)
plt.plot(range(0,500),xk_t[:,0],c='b',lw=2)
plt.plot(range(0,501),kf.xk[0,0,:],c='r',lw=2)
plt.legend(['$xk_{1real}$','$xk_{1est}$'])
plt.xlim([0,ITER])
plt.grid(True,ls='dotted')
plt.xlabel('(a)\n')
plt.ylabel('$xk_1$ []')

plt.subplot(2,1,2)
plt.plot(range(0,500),xk_t[:,1],c='b',lw=2)
plt.plot(range(0,501),kf.xk[1,0,:],c='r',lw=2)
plt.legend(['$xk_{2real}$','$xk_{2est}$'])
plt.xlim([0,ITER])
plt.grid(True,ls='dotted')
plt.xlabel('(b)\n Iteração [N]\n')
plt.ylabel('$xk_2$ []')

plt.tight_layout()


plt.figure(figsize=(8,8),dpi=180)
plt.subplot(2,1,1)
plt.plot(range(0,500),xk_t[:,0] - kf.xk[0,0,:-1],c='b',lw=2)
plt.xlim([0,ITER])
plt.grid(True,ls='dotted')
plt.xlabel('(a)\n')
plt.ylabel('$xk_{1real} - xk_{1est}$ []')

plt.subplot(2,1,2)
plt.plot(range(0,500),xk_t[:,1] - kf.xk[1,0,:-1],c='b',lw=2)
plt.xlim([0,ITER])
plt.grid(True,ls='dotted')
plt.xlabel('(b)\n Iteração [N]\n')
plt.ylabel('$xk_{2real} - xk_{2est}$ []')

plt.tight_layout()

# Outputs --------------------------------------------------------------------
plt.figure(figsize=(8,8),dpi=180)
plt.subplot(2,1,1)
plt.plot(range(0,500),yk_t[:,0],c='b',lw=2)
plt.plot(range(0,501),kf.yk[0,0,:],c='r',lw=2)
plt.legend(['$yk_{1real}$','$yk_{1est}$'])
plt.xlim([0,ITER])
plt.grid(True,ls='dotted')
plt.xlabel('(a)\n')
plt.ylabel('$yk_1$ []')

plt.subplot(2,1,2)
plt.plot(range(0,500),yk_t[:,1],c='b',lw=2)
plt.plot(range(0,501),kf.yk[1,0,:],c='r',lw=2)
plt.legend(['$yk_{2real}$','$yk_{2est}$'])
plt.xlim([0,ITER])
plt.grid(True,ls='dotted')
plt.xlabel('(b)\n Iteração [N]\n')
plt.ylabel('$yk_2$ []')

plt.tight_layout()

plt.figure(figsize=(8,8),dpi=180)
plt.subplot(2,1,1)
plt.plot(range(0,500),yk_t[:,0] - kf.yk[0,0,:-1],c='b',lw=2)
plt.xlim([0,ITER])
plt.grid(True,ls='dotted')
plt.xlabel('(a)\n')
plt.ylabel('$yk_{1real} - yk_{1est}$ []')

plt.subplot(2,1,2)
plt.plot(range(0,500),yk_t[:,1] - kf.yk[1,0,:-1],c='b',lw=2)
plt.xlim([0,ITER])
plt.grid(True,ls='dotted')
plt.xlabel('(b)\n Iteração [N]\n')
plt.ylabel('$yk_{2real} - yk_{2est}$ []')

plt.tight_layout()

# Covariância ------------------------------------------------
plt.figure(figsize=(8,8),dpi=180)
plt.subplot(2,1,1)
plt.plot(kf.Pk[0,0,:],c='b',lw=2)
plt.grid(True,ls='dotted')
plt.ylabel("$P_{0,0}(k|k-1)$ []")
plt.xlabel('(a)\n')

plt.subplot(2,1,2)
plt.plot(kf.Pk[1,1,:],c='b',lw=2)
plt.grid(True,ls='dotted')
plt.ylabel("$P_{1,1}(k|k-1)$ []")
plt.xlabel('(b)\n Iteração [N]\n')

plt.tight_layout()
