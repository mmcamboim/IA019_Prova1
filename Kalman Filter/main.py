# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 19:15:23 2021

@author: mcamboim
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

from kalman_func import KalmanFilter

with open('system.pickle','rb') as file:
    [Ak,Bk,Ck,xk_t,yk_t,W_STD,V_STD,ITER] = pickle.load(file)

Pk0 = 1.0
kf = KalmanFilter(Ak,Bk,Ck,W_STD,V_STD,ITER,Pk0)

for i in range(ITER):
    #kf.RunKf(yk_t[i].reshape(2,1))
    kf.RunKfStepAhead(yk_t[i].reshape(2,1))

# Ajustar gráficos para ficarem sobrepostos
plt.subplot(2,1,1)
plt.plot(xk_t[:,1])
plt.plot(kf.xk[1,0,0:])

plt.subplot(2,1,2)
plt.plot(xk_t[:,0])
plt.plot(kf.xk[0,0,1:])

plt.figure()
plt.plot(kf.Pk[1,1,:])