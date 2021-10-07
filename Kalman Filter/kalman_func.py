# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 19:17:35 2021

@author: mcamboim
"""
import numpy as np

class KalmanFilter:
    def __init__(self,Ak,Bk,Ck,W_STD,V_STD,ITER,Pk0):
        # Creating Variables
        self.Ak = Ak
        self.Bk = Bk
        self.Ck = Ck
        self.Qk = np.eye(Bk.shape[1]) * W_STD
        self.Rk = np.eye(np.min(Ck.shape)) * V_STD
        self.Ik = np.eye(Ak.shape[0])
        
        self.Lk = np.zeros((Ak.shape[0],Ak.shape[0],ITER + 1))
        self.Mk = np.zeros((Ak.shape[0],Ck.shape[0],ITER + 1))
        self.Fk = np.zeros((Ck.shape[0],Ck.shape[0],ITER + 1))
        self.Kk = np.zeros((Ck.shape[1],Ck.shape[0],ITER + 1))
        self.Pk = np.zeros((Ak.shape[0],Ak.shape[0],ITER + 1))
        self.xk = np.zeros((Ak.shape[0],1,ITER + 1))
        self.yk = np.zeros((Ck.shape[0],1,ITER + 1))
        self.ek = np.zeros((Ck.shape[0],1,ITER + 1))
        
        # Iniatializing Variables
        self.xk[:,:,0] = np.array([np.pi/18,0]).reshape(2,1)
        self.Pk[:,:,0] = np.eye(Ak.shape[0]) * Pk0
        self.iter = 0
      
    def RunKf(self,y_meas):
        self.xk[:,:,self.iter + 1] = self.Ak @ self.xk[:,:,self.iter]
        
        self.Pk[:,:,self.iter + 1] = self.Ak @ self.Pk[:,:,self.iter] @ self.Ak.T + self.Bk @ self.Qk @ self.Bk.T
        
        self.yk[:,:,self.iter + 1] = self.Ck @ self.xk[:,:,self.iter + 1]
        self.ek[:,:,self.iter + 1] = y_meas - self.yk[:,:,self.iter + 1]
        
        self.Mk[:,:,self.iter + 1] = self.Pk[:,:,self.iter + 1] @ self.Ck.T
        self.Fk[:,:,self.iter + 1] = self.Ck @ self.Pk[:,:,self.iter + 1] @ self.Ck.T + self.Rk
        self.Kk[:,:,self.iter + 1] = self.Mk[:,:,self.iter + 1] @ np.linalg.inv(self.Fk[:,:,self.iter + 1])
        
        self.xk[:,:,self.iter + 1] += self.Kk[:,:,self.iter + 1] @ self.ek[:,:,self.iter + 1]
        
        self.Pk[:,:,self.iter + 1] *= (self.Ik - self.Kk[:,:,self.iter + 1] @ self.Ck)
        
        self.iter += 1 
 
    def RunKfStepAhead(self,y_meas):
        self.Mk[:,:,self.iter + 1] = self.Ak @ self.Pk[:,:,self.iter] @ self.Ck.T
        self.Fk[:,:,self.iter + 1] = self.Ck @ self.Pk[:,:,self.iter] @ self.Ck.T + self.Rk
        self.Kk[:,:,self.iter + 1] = self.Mk[:,:,self.iter + 1] @ np.linalg.inv(self.Fk[:,:,self.iter + 1])
        
        self.yk[:,:,self.iter + 1] = self.Ck @ self.xk[:,:,self.iter]
        self.ek[:,:,self.iter + 1] = y_meas - self.yk[:,:,self.iter + 1]
        
        vt = self.Kk[:,:,self.iter + 1] @ self.ek[:,:,self.iter + 1]
        self.xk[:,:,self.iter + 1] = self.Ak @ self.xk[:,:,self.iter]  +  vt
        
        self.Lk[:,:,self.iter + 1] = self.Ak - self.Kk[:,:,self.iter + 1] @ self.Ck
        
        self.Pk[:,:,self.iter + 1] = self.Ak @ self.Pk[:,:,self.iter] @ self.Lk[:,:,self.iter + 1].T + self.Bk @ self.Qk @ self.Bk.T
        
        self.iter +=1 