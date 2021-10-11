# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 19:15:23 2021

@author: mcamboim
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

from kalman_func import KalmanFilter

plt.close('all')

# Loading System Data ========================================================
with open('system.pickle','rb') as file:
    [Ak,Bk,Ck,xk_t,yk_t,W_STD,V_STD,ITER] = pickle.load(file)

# Initial Variables ==========================================================
Pk0 = 1000.0
xk0 = np.array([0,0]).reshape(2,1)
kf = KalmanFilter(Ak,Bk,Ck,W_STD,V_STD,ITER,Pk0,xk0)

# Running ====================================================================
for i in range(ITER):
    kf.RunKfOneStepAhead(yk_t[i].reshape(2,1))

    
# Plotting ===================================================================
plt.figure(figsize=(8,8),dpi=180)
plt.subplot(2,1,1)
plt.plot(range(0,500),xk_t[:,0],c='b',lw=2)
plt.plot(range(0,501),kf.xk[0,0,:],c='r',lw=2)
plt.legend(['$xk_{1real}$','$xk_{1est}$'])
plt.xlim([0,ITER])
plt.grid(True,ls='dotted')
plt.ylabel('xk_1 []')

plt.subplot(2,1,2)
plt.plot(range(0,500),xk_t[:,1],c='b',lw=2)
plt.plot(range(0,501),kf.xk[1,0,:],c='r',lw=2)
plt.legend(['$xk_{2real}$','$xk_{2est}$'])
plt.xlim([0,ITER])
plt.grid(True,ls='dotted')
plt.xlabel('Iteração [N]')
plt.ylabel('xk_2 []')


plt.figure()
plt.plot(kf.Pk[1,1,:])

