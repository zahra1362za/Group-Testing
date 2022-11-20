# import neccessary libraries
import numpy as np
import random
from datetime import datetime
from scipy.stats import beta
from sklearn.metrics import accuracy_score
import os
import Cod
import matplotlib.pyplot as plt
from scipy import spatial
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score


# initialize parameters for beta distributions:
a_alpha=1
b_alpha=900
a_beta=10
b_beta=8000
a_betaf=20
b_betaf=8000
a_gama=50
b_gama=200
a_teta0=10
b_teta0=1000
a_teta1=9000
b_teta1=300
P=1
U=1
K=2
J=1
n,T=64,128
epsln=0.001
hyper_params=np.array([a_alpha,b_alpha,a_beta,b_beta,a_betaf,b_betaf,a_gama,b_gama,a_teta0,b_teta0,a_teta1,b_teta1])

Data=np.load('data.npy',allow_pickle=True)
G,YF,X_true,F =Data[0],Data[1],Data[2],Data[3]


unique_rows = np.unique(F, axis=0)

def R_(G,X,params,F):
    T,n=G.shape[0],G.shape[1]
    alpha_,beta_,betaf,gama_,theta_0_,theta_1_=params[0],params[1],params[2],params[3],params[4],params[5]
    infected_neighbore=np.array(CNbr(G,X))
    R=np.zeros((n,T))+1
    for i in range(n):
        for t in range(T-1):

            if (X[i][t]==0)&(X[i][t+1]==1):
                c=int(infected_neighbore[i,t])
                cf=int(F.dot(X.T[t])[i])
                if cf==0:
                    cf=cf+1                
                pr_a=alpha_/(alpha_+beta_*c+betaf*cf)
                pr_b=beta_/(alpha_+beta_*c+betaf*cf)
                pr_bf=betaf/(alpha_+beta_*c+betaf*cf)
                v=np.random.multinomial(1, [pr_a]+[pr_b]*c+[pr_bf]*cf)
                print(v,cf)
                if v[0]==1:
                    R[i,t]=0
                if len(v)-np.where(v==1)[0][0]>cf:
                    R[i,t]=2
                else:    
                    R[i,t]=3
    return R
            
#***************************************************            
def algrthm(params,X,G,YF):
    
    param=[]
    n,T=X.shape[0],X.shape[1]
    pos_probs=np.zeros((n,T))
    for i in range(U):
        
        T=X.shape[1]
        
        param.append(params)
        
        cal=Cod.Calculate_X(pos_probs,K,X,G,F,YF,params,P)
        X=cal[0]
        pos_probs=cal[1]
        R=Cod.R_(G,X,params,F)
        if (i!=U-1):
            prm=Cod.Params(R,G,F,X,YF,hyper_params)
            
            if i>1 & Cod.epsilone(np.array(param[-1]),np.array(params)):
                param.append(params)
                params=prm[0]
                R=prm[1]
    with open('b.npy', 'wb') as f:

        np.save(f, params)

        f.close()
    return X,pos_probs
            

# functions for multiprocessing
def Step_Gibbs(params,X,G,Y):
    Trained=[]
    T=G.shape[0]
    for time_step in range(8,T,8):
        G_=G[:time_step]
        Y_=Y[:,:time_step]
        X_=X[:,:time_step]
        Train=algrthm(params,X_,G_,Y_)
        Trained.append(Train) 
    return Trained     

