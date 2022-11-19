#!/usr/bin/env python
# coding: utf-8

# In[1]:


import multiprocessing
from multiprocessing import Pool
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from scipy.stats import beta
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import worker_Data

# In[2]:


def Sample_alpha(a_alpha, b_alpha):
    
    try:
        for i in beta.rvs(a_alpha, b_alpha, size=10000):
            if (i>0.001)&(i<0.2):
                alpha_=round(i,3)
                return alpha_ 
        if not(i>0.001):
            return -1.0
        else:
            return -2.0
    except  ValueError:
        print(a_alpha, b_alpha)


# In[3]:


def Sample_betaf(a_betaf, b_betaf):
    for i in beta.rvs(a_betaf, b_betaf, size=1000):
        if (i>0.002)&(i<0.5):
            betaf=round(i,4)
            return betaf
    if not(i>0.002):
        return -1.0
    else:
        return -2.0


# In[4]:


def Sample_gama(a_gama,b_gama):
    for i in beta.rvs(a_gama, b_gama, size=10000):
        if (i>0.1)&(i<0.5):
            gama_=round(i,3)
            return gama_  
    if not(i>0.1):
        return -1.0
    else:
        return -2.0        


# In[5]:


def Sample_theta0(a_teta0, b_teta0):
    for i in beta.rvs(a_teta0, b_teta0, size=10000):
        if (i>0.01)&(i<0.51):
            theta_0_=round(i,3)
            return theta_0_  
    if not(i>0.01):
        return -1.0
    else:
        return -2.0        


# In[6]:


def Sample_theta1(a_teta1, b_teta1):
    for i in beta.rvs(a_teta1, b_teta1, size=10000):
        if i>0.78:
            theta_1_=round(i,3)
            return theta_1_ 
    if not(i>0.78):
        return -1
               


# In[7]:


def Sample_beta(a_beta, b_beta):
    for i in beta.rvs(a_beta, b_beta, size=10000):
        if (i>0.001)&(i<0.0451):
            beta_=round(i,4)
            return beta_ 
    if not(i>0.001):
        return -1.0
    else:
        return -2.0        


# In[8]:


# function to sample infection and emission parameters(alpha,beta,betaf,gama,teta0,teta1)
def initialize_parameters(hyper_params):
    a_alpha=hyper_params[0]
    b_alpha=hyper_params[1]
    a_beta=hyper_params[2]
    b_beta=hyper_params[3]
    a_betaf=hyper_params[4]
    b_betaf=hyper_params[5]
    a_gama=hyper_params[6]
    b_gama=hyper_params[7]
    a_teta0=hyper_params[8]
    b_teta0=hyper_params[9]
    a_teta1=hyper_params[10]
    b_teta1=hyper_params[11]
    alpha_=Sample_alpha(a_alpha, b_alpha)
    beta_=Sample_beta(a_beta, b_beta)
    betaf=Sample_betaf(a_betaf, b_betaf)
    while alpha_>beta_:
        print("HR")
        alpha_=Sample_alpha(a_alpha, b_alpha)
        beta_=Sample_beta(a_beta, b_beta)
    while beta_>betaf:
        print("hrr")
        betaf=Sample_betaf(a_betaf, b_betaf)
    gama_=Sample_gama(a_gama,b_gama)
    theta_0_=Sample_theta0(a_teta0, b_teta0)
    theta_1_=Sample_theta1(a_teta1, b_teta1)
    params=np.array([alpha_,beta_,betaf,gama_,theta_0_,theta_1_])
    return params


# In[31]:


#function for spliting X,Y,G into desired timesteps, the output of this function is the result of Gibbs sampling algorithm:
# returns probability which is the model estimation of X:
# this code is parallelized on different timesteps:

def Step_Gibbs_parallel(params,X,Y,G):
    arg=[]
    pool_list=[]
    for time_step in range(8,129,8):
        G_=G[:time_step]
        # here by passing Y related to YF_missing for specified value of mu, we can apply desired amount of missing value:
        Y_=Y[:,:time_step]
        X_=X[:,:time_step]
        
        pool_list.append([params,X_,G_,Y_])
        
    if __name__ ==  '__main__': 
        with Pool(processes =16) as pool:

            parallel_output = pool.starmap(worker_Data.algrthm,pool_list )# use tqdm to show the progress
            pool.close()
            pool.join()
    return parallel_output


# In[10]:


# function for plotting figure 1,outputs the array of AUC 
def plot_figure_1(Trained):
    
    roc_=[]
    for i in range(4):
        for j in range(len(Trained[i])):
            Train=np.array(Trained[i][j][1][0])
            print(Train.shape,X_true[:,:(j+1)*8].shape)
            y_score=np.hstack(Train)
            y_test=np.hstack(X_true[:,:(j+1)*8])
            roc_.append(plot_ROC(y_score,y_test))
    return  roc_


# In[11]:


       


# In[12]:


# function to count the number of the infected neighbores of i at t:
def CNbr(G,X):
    n,T=X.shape[0],X.shape[1]
    C=np.zeros((T,n))
    for t in range(T):
        C[t]=G[t].dot(X.T[t])
    return C.T


# In[16]:


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
U=3
K=100
J=1
n,T=64,128
epsln=0.001
hyper_params=np.array([a_alpha,b_alpha,a_beta,b_beta,a_betaf,b_betaf,a_gama,b_gama,a_teta0,b_teta0,a_teta1,b_teta1])











# In[22]:
Data=np.load('data.npy',allow_pickle=True)
G,YF,X_true,F =Data[0],Data[1],Data[2],Data[3]


# function to generate missing data:
# mu is the average number of tests for a typical family during a year:
def missing_data(mu,YF):
    number_of_families,T=YF.shape[0],YF.shape[1]
    YF_missing=np.zeros((number_of_families,T))-1
    time=list(range(T))
    random.shuffle(time)
    for f in range(number_of_families):
        idx=np.sort(np.random.choice(time, mu,replace=False))
        for t in idx:
            YF_missing[f,t]=YF[f,t]
        
    return YF_missing 


# In[24]:
n,T=X_true.shape[0],X_true.shape[1]


# function to count the number of the infected neighbores of i at t:
# Function to obtain the very initial sample of X, using forwad sampling:
def Forward_Sampling(T,n,G,F,param):
    alpha_=param[0]
    beta_=param[1]
    betaf=param[2]
    gama_=param[3]
    p0=P
    p1=1-P
    x=int(np.round(((1-P)*n),0))
    X=np.zeros((n,T))  
    idx=np.random.choice(range(n), x)
    X[idx,0]=1
    for t in range(T-1):
        
        cf=F.dot(X.T[t])
        
        c=CNbr(G,X)[:,t]
        p1=(alpha_+beta_*c+betaf*cf)**(1-X[:,t])*(1-gama_)**(X[:,t])
               
        X[:,t+1]=np.random.binomial( 1, p1,size=None) 
    return X    

mu=[1,2,3,4,6,12]
MissingData=[]
for i in mu:
    MissingData.append([missing_data(i,YF),i])
#%store MissingData   


# In[25]:

unique_rows = np.unique(F, axis=0)

params= initialize_parameters(hyper_params)
X=Forward_Sampling(T,n,G,F,params)
    
Trained=Step_Gibbs_parallel(params,X,MissingData[-1][0],G)
for i in range(len(Trained)):
    a=str(i)+".npz"
    np.savez(a,Trained[i][0])   
       





