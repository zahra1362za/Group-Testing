# import neccessary libraries
import numpy as np
import random
from datetime import datetime
from scipy.stats import beta
import Code_mu
import matplotlib.pyplot as plt
from scipy import spatial
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import os
# initialize parameters for beta distributions:
a_alpha=1
b_alpha=400
a_beta=1
b_beta=800
a_betaf=8
b_betaf=800
a_gama=50
b_gama=200
a_teta0=10
b_teta0=1000
a_teta1=9000
b_teta1=300
P=1
U=2
K=2
n,T=100,360
epsln=0.001
hyper_params=np.array([a_alpha,b_alpha,a_beta,b_beta,a_betaf,b_betaf,a_gama,b_gama,a_teta0,b_teta0,a_teta1,b_teta1])

if os.path.exists('G.npy'):
    G = np.load('G.npy')
if os.path.exists('YF.npy'):
    YF = np.load('YF.npy')
if os.path.exists('F.npy'):
    F = np.load('F.npy')
if os.path.exists('X.npy'):
    X_True = np.load('X.npy')
n,T=X_True.shape[0],X_True.shape[1]

unique_rows = np.unique(F, axis=0)    
#function to plot figure 2:    
def algrthm(params,X,Y):
    prob=[]
    param=[]
    n,T=X.shape[0],X.shape[1]
    pos_probs=np.zeros((n,T))
    for i in range(U):
       
        with open('algiteration.txt', 'w') as file:
            file.write(str(i))
            file.close()
        param.append(params)
        
        cal=Code_mu.Calculate_X(pos_probs,K,X,G,F,Y,params,P)
        X=cal[0]
        pos_probs=cal[1]
        R=Code_mu.R_(G,X,params,F)
        np.savez('paramalg_.npz',params)
        if (i!=U-1):
            prm=Code_mu.Params(R,G,F,X,Y,hyper_params)
            params=prm[0]
            if i>1 & Code_mu.epsilone(np.array(param[-1]),np.array(params)):
                param.append(params)
                
                R=prm[1]
    prob.append(pos_probs)
    param.append(params)
    with open('parameters.txt', 'w') as file:
        file.write(param)
        file.close()    
    return prob

