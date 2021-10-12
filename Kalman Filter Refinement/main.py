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
Pk0 = 1.0
xk0 = np.array([np.pi/18,0]).reshape(2,1)
kf = KalmanFilter(Ak,Bk,Ck,W_STD,V_STD,ITER,Pk0,xk0)

# Running ====================================================================
for i in range(ITER):
    kf.RunKfOneStepAhead(yk_t[i].reshape(2,1))
# Refinement =================================================================

kf.iter = ITER - 1
for i in range(ITER - 1):
    kf.RunKfRefinement()
    

# Plotting ===================================================================
plt.rcParams['axes.linewidth'] = 2.0
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}
plt.rc('font', **font)

# States ---------------------------------------------------------------------
fig = plt.figure(figsize=(8,8),dpi=180)
ax1 = fig.add_subplot(2,1,1)
plt.plot(range(0,500),xk_t[:,0],c='b',lw=2)
plt.plot(range(0,501),kf.xk[0,0,:],c='r',lw=2)
plt.plot(range(0,501),kf.xk_N[0,0,:],c='g',lw=2)
plt.legend(['$xk_{1real}$','$xk_{1est}$','$xk_{1ref}$'])
plt.xlim([0,ITER])
plt.grid(True,ls='dotted')
plt.xlabel('(a)\n')
plt.ylabel('$xk_1$ []')

left, bottom, width, height = [0.18, 0.68, 0.25, 0.2]
ax2 = fig.add_axes([left, bottom, width, height])
ax2.plot(range(100,151),xk_t[100:151,0],c='b',lw = 2)
ax2.plot(range(100,151),kf.xk[0,0,100:151],c='r',lw = 2)
ax2.plot(range(100,151),kf.xk_N[0,0,100:151],c='g',lw = 2)
plt.xlim([100,150])
plt.grid(True,ls='dotted')

ax1 = fig.add_subplot(2,1,2)
plt.plot(range(0,500),xk_t[:,1],c='b',lw=2)
plt.plot(range(0,501),kf.xk[1,0,:],c='r',lw=2)
plt.plot(range(0,501),kf.xk_N[1,0,:],c='g',lw=2)
plt.legend(['$xk_{1real}$','$xk_{1est}$','$xk_{1ref}$'])
plt.xlim([0,ITER])
plt.grid(True,ls='dotted')
plt.xlabel('(b)\n Iteração [N]\n')
plt.ylabel('$xk_2 []$')

plt.tight_layout()


plt.figure(figsize=(8,8),dpi=180)
plt.subplot(2,1,1)
plt.plot(range(0,500),xk_t[:,0] - kf.xk_N[0,0,:-1],c='b',lw=2)
plt.xlim([0,ITER])
plt.ylim([-1,1])
plt.grid(True,ls='dotted')
plt.ylabel('$xk_{1real} - xk_{1ref}$ []')
plt.xlabel('(a)\n')

plt.subplot(2,1,2)
plt.plot(range(0,500),xk_t[:,1] - kf.xk_N[1,0,:-1],c='b',lw=2)
plt.xlim([0,ITER])
plt.ylim([-1,1])
plt.grid(True,ls='dotted')
plt.xlabel('(b)\n Iteração [N]\n')
plt.ylabel('$xk_{2real} - xk_{2ref}$ []')

plt.tight_layout()

# Output ---------------------------------------------------------------------
fig = plt.figure(figsize=(8,8),dpi=180)
ax1 = fig.add_subplot(2,1,1)
plt.plot(range(0,500),yk_t[:,0],c='b',lw=2)
plt.plot(range(0,501),kf.yk[0,0,:],c='r',lw=2)
plt.plot(range(0,501),kf.yk_N[0,0,:],c='g',lw=2)
plt.legend(['$yk_{1real}$','$yk_{1est}$','$yk_{1ref}$'])
plt.xlim([0,ITER])
plt.grid(True,ls='dotted')
plt.ylabel('$yk_1 []$')
plt.xlabel('(a)\n')

left, bottom, width, height = [0.18, 0.68, 0.25, 0.2]
ax2 = fig.add_axes([left, bottom, width, height])
ax2.plot(range(100,151),yk_t[100:151,0],c='b',lw = 2)
ax2.plot(range(100,151),kf.yk[0,0,100:151],c='r',lw = 2)
ax2.plot(range(100,151),kf.yk_N[0,0,100:151],c='g',lw = 2)
plt.xlim([100,150])
plt.grid(True,ls='dotted')

ax1 = fig.add_subplot(2,1,2)
plt.plot(range(0,500),yk_t[:,1],c='b',lw=2)
plt.plot(range(0,501),kf.yk[1,0,:],c='r',lw=2)
plt.plot(range(0,501),kf.yk_N[1,0,:],c='g',lw=2)
plt.legend(['$yk_{1real}$','$yk_{1est}$','$yk_{1ref}$'])
plt.xlim([0,ITER])
plt.grid(True,ls='dotted')
plt.xlabel('(b)\n Iteração [N]\n')
plt.ylabel('$yk_2 []$')

plt.tight_layout()


plt.figure(figsize=(8,8),dpi=180)
plt.subplot(2,1,1)
plt.plot(range(0,500),yk_t[:,0] - kf.yk_N[0,0,:-1],c='b',lw=2)
plt.xlim([0,ITER])
plt.ylim([-1,1])
plt.xlabel('(a)\n')

plt.grid(True,ls='dotted')
plt.ylabel('$yk_{1real} - yk_{1ref}$ []')

plt.subplot(2,1,2)
plt.plot(range(0,500),yk_t[:,1] - kf.yk_N[1,0,:-1],c='b',lw=2)
plt.xlim([0,ITER])
plt.ylim([-1,1])
plt.grid(True,ls='dotted')
plt.xlabel('(b)\n Iteração [N]\n')
plt.ylabel('$yk_{2real} - yk_{2ref}$ []')

plt.tight_layout()

# Covariance -----------------------------------------------------------------

plt.figure(figsize=(8,8),dpi=180)
plt.subplot(2,1,1)
plt.plot(kf.Pk[0,0,:],c='b',lw=2)
plt.plot(kf.Pk_N[0,0,:],c='r',lw=2)
plt.grid(True,ls='dotted')
plt.ylabel("$P_{0,0}$ []")
plt.xlabel('(a)\n')
plt.legend(['P$_{0,0}(k|k-1)$','P$_{0,0}(k|N)$'])

plt.subplot(2,1,2)
plt.plot(kf.Pk[1,1,:],c='b',lw=2)
plt.plot(kf.Pk_N[1,1,:],c='r',lw=2)
plt.grid(True,ls='dotted')
plt.ylabel("$P_{1,1}$ []")
plt.xlabel('(b)\n Iteração [N]\n')
plt.legend(['P$_{1,1}(k|k-1)$','P$_{1,1}(k|N)$'])

plt.tight_layout()

