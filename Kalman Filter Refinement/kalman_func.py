# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 19:17:35 2021

@author: mcamboim
"""
import numpy as np

class KalmanFilter:
    def __init__(self,Ak,Bk,Ck,W_STD,V_STD,ITER,Pk0,xk0):
        self.InitTIVar(Ak,Bk,Ck,W_STD,V_STD)
        self.InitTVVar(Ak,Bk,Ck,ITER)
        self.VarInit(Ak,Pk0,xk0)
        self.VarRefinementInit(Ak,ITER)

    def InitTIVar(self,Ak,Bk,Ck,W_STD,V_STD):
        self.Ak = Ak
        self.Bk = Bk
        self.Ck = Ck
        self.Qk = np.eye(Bk.shape[1]) * W_STD
        self.Rk = np.eye(Ck.shape[0]) * V_STD

    def InitTVVar(self,Ak,Bk,Ck,ITER):
        self.Lk = np.zeros((Ak.shape[0],Ak.shape[1],ITER + 1))
        self.Mk = np.zeros((Ak.shape[0],Ck.shape[0],ITER + 1))
        self.Fk = np.zeros((Ck.shape[0],Ck.shape[0],ITER + 1))
        self.Kk = np.zeros((Ak.shape[0],Ck.shape[0],ITER + 1))
        self.Pk = np.zeros((Ak.shape[0],Ak.shape[1],ITER + 1))
        self.xk = np.zeros((Ak.shape[0],1,ITER + 1))
        self.yk = np.zeros((Ck.shape[0],1,ITER + 1))
        self.ek = np.zeros((Ck.shape[0],1,ITER + 1))

    def VarInit(self,Ak,Pk0,xk0):
        self.xk[:,:,0] = xk0
        self.Pk[:,:,0] = np.eye(Ak.shape[0]) * Pk0
        self.iter = 0
        
    def VarRefinementInit(self,Ak,ITER):
        self.rk = np.zeros((Ak.shape[0],1,ITER))
        self.Nk = np.zeros((Ak.shape[0],Ak.shape[1],ITER))
        self.xk_N = np.zeros((Ak.shape[0],1,ITER + 1))
        self.Pk_N = np.zeros((Ak.shape[0],Ak.shape[1],ITER + 1))

    def RunKfOneStepAhead(self,y_meas):
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
        
    def RunKfRefinement(self):
        self.rk[:,:,self.iter - 1] = self.Ck.T @ np.linalg.inv(self.Fk[:,:,self.iter]) @ self.ek[:,:,self.iter] + self.Lk[:,:,self.iter].T @ self.rk[:,:,self.iter] 
        self.Nk[:,:,self.iter - 1] = self.Ck.T @ np.linalg.inv(self.Fk[:,:,self.iter]) @ self.Ck + self.Lk[:,:,self.iter].T @ self.Nk[:,:,self.iter] @ self.Lk[:,:,self.iter]
        self.xk_N[:,:,self.iter] = self.xk[:,:,self.iter] + self.Pk[:,:,self.iter] @ self.rk[:,:,self.iter - 1]
        self.Pk_N[:,:,self.iter] = self.Pk[:,:,self.iter] - self.Pk[:,:,self.iter] @ self.Nk[:,:,self.iter - 1] @ self.Pk[:,:,self.iter]
        
        self.iter -= 1
        
        