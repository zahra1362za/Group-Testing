import numpy as np
import random

from datetime import datetime
from scipy.stats import beta
from sklearn.metrics import accuracy_score

import os
epsln=0.001

if os.path.exists('G.npy'):
    G = np.load('G.npy')
if os.path.exists('YF.npy'):
    YF = np.load('YF.npy')
if os.path.exists('F.npy'):
    F = np.load('F.npy')
if os.path.exists('X.npy'):
    X_true = np.load('X.npy')
if os.path.exists('Y.npy'):
    Y = np.load('Y.npy')
n,T=X_true.shape[0],X_true.shape[1]
a_alpha=1
b_alpha=400
a_beta=1
b_beta=800
a_betaf=3
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
J=1
n,T=100,360
epsln=0.001
hyper_params=np.array([a_alpha,b_alpha,a_beta,b_beta,a_betaf,b_betaf,a_gama,b_gama,a_teta0,b_teta0,a_teta1,b_teta1])
def process(hyper_params):
    params= initialize_parameters(hyper_params)
    X=Forward_Sampling(T,n,G,F,params)
    return np.array([params,X]) 
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
    while alpha_==-1:
        a_alpha=a_alpha+5
        hyper_params[0]=a_alpha
        alpha_=Sample_alpha(a_alpha, b_alpha)
    while alpha_==-2:
        b_alpha=b_alpha+500
        hyper_params[1]=b_alpha
        alpha_=Sample_alpha(a_alpha, b_alpha)
    beta_=Sample_beta(a_beta, b_beta)
    while beta_==-1:
        a_beta=a_beta+5
        hyper_params[2]=a_beta
        beta_=Sample_beta(a_beta, b_beta)
    while beta_==-2:
        b_beta=b_beta+500
        hyper_params[3]=b_beta
        beta_=Sample_beta(a_beta, b_beta)    
    betaf=Sample_betaf(a_betaf, b_betaf)
    while alpha_>beta_:
        alpha_=Sample_alpha(a_alpha, b_alpha)
        beta_=Sample_beta(a_beta, b_beta)
    while beta_>betaf:
        betaf=Sample_betaf(a_betaf, b_betaf)
    gama_=Sample_gama(a_gama,b_gama)
    theta_0_=Sample_theta0(a_teta0, b_teta0)
    while theta_0_==-1:
        a_teta0=a_teta0+5
        hyper_params[8]=a_teta0
        theta_0_=Sample_theta0(a_teta0, b_teta0)
    while theta_0_==-2:
        b_teta0=b_teta0+500
        hyper_params[9]=b_teta0
        theta_0_=Sample_theta0(a_teta0, b_teta0)       
    theta_1_=Sample_theta1(a_teta1, b_teta1)
    while theta_1_==-1:
        a_teta1=a_teta1+5
        hyper_params[8]=a_teta1
        theta_1_=Sample_theta1(a_teta0, b_teta0)
   
    params=np.array([alpha_,beta_,betaf,gama_,theta_0_,theta_1_])
    return params
def process(hyper_params):
    params= initialize_parameters(hyper_params)
    X=Forward_Sampling(T,n,G,F,params)
    return np.array([params,X])        
# function to count the number of the infected neighbores of i at t:
def CNbr(G,X):
    n,T=X.shape[0],X.shape[1]
    C=np.zeros((T,n))
    for t in range(T):
        C[t]=G[t].dot(X.T[t])
    return C.T

def transition(X,t,G,F,j,param):
    alpha_=param[0]
    beta_=param[1]
    betaf=param[2]
    gama_=param[3]
    c=CNbr(G,X)[j][t]
    number_of_infected_members_in_family=F.dot(X.T[t])[j]
    k=X[j,t]-2*X[j,t+1]
    if k==0:
        return 1-alpha_-beta_*c-betaf*number_of_infected_members_in_family
    elif k==-2:
        return alpha_+beta_*c+betaf*number_of_infected_members_in_family
    elif k==1:
        return gama_
    else:
        return 1-gama_
    
def Sample_hidden_state(pos_probs,X,G,F,Y,param,P,t):
    unique_rows = np.unique(F, axis=0)
    alpha_=param[0]
    beta_=param[1]
    betaf=param[2]
    gama_=param[3]
    theta_0_=param[4]
    theta_1_=param[5]
    n,T=X.shape[0],X.shape[1]
    
    for i in range(n):
        if t==0:
            p_0,p_1=P,1-P
        else:
            p_0,p_1=1,1
        j=family_index(i,unique_rows)
        pow0=np.count_nonzero(Y[j,t]==0)
        pow1=np.count_nonzero(Y[j,t]==1)
        #pow1_=np.count_nonzero(Y[i,t]==-1)
        number_of_members_in_family=np.sum(unique_rows[family_index(i,unique_rows)])
        
        X[i,t]=0
        number_of_infected_members_in_family0=F.dot(X.T[t])[i]
        number_of_healthy_members_in_family0=number_of_members_in_family-number_of_infected_members_in_family0
        p_0=(1/number_of_members_in_family)*p_0*((1-theta_0_)*number_of_healthy_members_in_family0+(1-theta_1_)*number_of_infected_members_in_family0)**pow0*(theta_1_*number_of_infected_members_in_family0+theta_0_*number_of_healthy_members_in_family0)**pow1
        if (t==0):
            c=G[t].dot(X.T[t])[i]
        else:    
            c=G[t-1].dot(X.T[t-1])[i]
    
        if t!=0:
            if X[i,t-1]==0:
                p_0=p_0*(1-alpha_-beta_*c-betaf*number_of_infected_members_in_family0)
            else:
                p_0=p_0*gama_
        
        X[i,t]=1
        number_of_infected_members_in_family1=F.dot(X.T[t])[i]
        number_of_healthy_members_in_family1=number_of_members_in_family-number_of_infected_members_in_family1
        p_1=(1/number_of_members_in_family)*p_1*((1-theta_0_)*number_of_healthy_members_in_family1+(1-theta_1_)*number_of_infected_members_in_family1)**pow0*(theta_1_*number_of_infected_members_in_family1+theta_0_*number_of_healthy_members_in_family1)**pow1

        if (t==0):
            c=G[t].dot(X.T[t])[i]
        else:    
            c=G[t-1].dot(X.T[t-1])[i]
        if t!=0:
            if X[i,t-1]==0:
                p_1=p_1*(alpha_+beta_*c+betaf*number_of_infected_members_in_family1)
            else:
                p_1=p_1*(1-gama_)
        family_members=unique_rows[family_index(i,unique_rows)]
        
        if t!=T-1:        
            X[i,t]=0
            for j in np.where(family_members==1)[0]:
                if j!=i:
                    p_0=p_0*transition(X,t,G,F,j,param)
            for j in np.where(G[t][i]==1)[0]:
                p_0=p_0*transition(X,t,G,F,j,param)
            X[i,t]=1
            for j in np.where(family_members==1)[0]:
                if j!=i:
                    p_1=p_1*transition(X,t,G,F,j,param)
            for j in np.where(G[t][i]==1)[0]:
                p_1=p_1*transition(X,t,G,F,j,param)
        if t==T-1:
            if X[i,t-1]==0:
                X[i,t]=0
                c=G[t].dot(X.T[t])[i]
                number_of_infected_members_in_family=F.dot(X.T[t])[i]
                p_0=p_0*(1-alpha_-beta_*c-betaf*number_of_infected_members_in_family)
                X[i,t]=1
                c=G[t].dot(X.T[t])[i]
                number_of_infected_members_in_family=F.dot(X.T[t])[i]
                p_1=p_1*(alpha_+beta_*c+betaf*number_of_infected_members_in_family)
            else:
                p_0=p_0*gama_
                p_1=p_1*(1-gama_)
        if p_0+p_1==0:            
            l=0.5
        else:
            l=p_1/(p_0+p_1)
        if (l<0)|(l>1): 
            np.savetxt("lcod.txt",[p_0,p_1])
            np.savez('paramt.npz',param)
        X[i,t]=np.random.binomial( 1,  l,size=None)  
        
        pos_probs[i,t]=l
    return X ,pos_probs   
epsln=0.001
def epsilone(a,b):
    return np.abs(a-b).all()>epsln

# function to sample new parameters and update parameters:

# function to sample new parameters and update parameters:
def Params(R,G,F,X,YF,hyper_params):
    n,T=X.shape[0],X.shape[1]
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
    unique_rows=np.unique(F,axis=0)  
    TP=np.sum(np.multiply(unique_rows.dot(X),YF))
    FP=np.count_nonzero(unique_rows.dot(X)-YF==-1)
    
    infected_neighbore=np.array(CNbr(G,X))
    
    alpha_=Sample_alpha(a_alpha +  np.count_nonzero(R==0) , b_alpha +np.count_nonzero(X==0)- np.count_nonzero(R==0))
    while alpha_==-1:
       
        hyper_params[0]=hyper_params[0]+5
        a_alpha=hyper_params[0]
        alpha_=Sample_alpha(a_alpha +  np.count_nonzero(R==0) , b_alpha +np.count_nonzero(X==0)- np.count_nonzero(R==0))
        
    while alpha_==-2:
                  
        hyper_params[1]=hyper_params[1]+500
        b_alpha=hyper_params[1]
        alpha_=Sample_alpha(a_alpha +  np.count_nonzero(R==0) , b_alpha +np.count_nonzero(X==0)- np.count_nonzero(R==0))
        
    beta_=Sample_beta(a_beta + np.count_nonzero(R==2) , b_beta +np.sum(np.multiply((1-X),infected_neighbore))-np.count_nonzero(R==2))
    while beta_==-1:
                  
        hyper_params[2]=hyper_params[2]+1
        a_beta=hyper_params[2]
        beta_=Sample_beta(a_beta + np.count_nonzero(R==2) , b_beta +np.sum(np.multiply((1-X),infected_neighbore))-np.count_nonzero(R==2))
         
    while beta_==-2:
                   
        hyper_params[3]=hyper_params[3]+500
        b_beta=hyper_params[3]
        beta_=Sample_beta(a_beta + np.count_nonzero(R==2) , b_beta +np.sum(np.multiply((1-X),infected_neighbore))-np.count_nonzero(R==2))
        
    betaf=Sample_betaf(a_betaf + np.count_nonzero(R==3) , b_betaf +np.sum(np.multiply((1-X),F.dot(X)))-np.count_nonzero(R==3))
    while betaf==-1:
                  
        hyper_params[4]=hyper_params[4]+1
        a_betaf=hyper_params[4]
        betaf=Sample_betaf(a_betaf + np.count_nonzero(R==3) , b_betaf +np.sum(np.multiply((1-X),F.dot(X)))-np.count_nonzero(R==3))
         
    while betaf==-2:
                   
        hyper_params[5]=hyper_params[5]+5
        b_betaf=hyper_params[5]
        betaf=Sample_betaf(a_betaf + np.count_nonzero(R==3) , b_betaf +np.sum(np.multiply((1-X),F.dot(X)))-np.count_nonzero(R==3))
        
    while alpha_>beta_:
                   
        hyper_params[1]=hyper_params[1]+500
        b_alpha=hyper_params[1]
        alpha_=Sample_alpha(a_alpha +  np.count_nonzero(R==0) , b_alpha +np.count_nonzero(X==0)- np.count_nonzero(R==0))
        beta_=Sample_beta(a_beta + np.count_nonzero(R==2) , b_beta +np.sum(np.multiply((1-X),infected_neighbore))-np.count_nonzero(R==2))
        
        
    while beta_>betaf:
                 
        hyper_params[4]=hyper_params[4]+5
        a_betaf=hyper_params[4]        
        betaf=Sample_betaf(a_betaf + np.count_nonzero(R==3) , b_betaf +np.sum(np.multiply((1-X),F.dot(X)))-np.count_nonzero(R==3))
        
    gama_=Sample_gama(a_gama +np.count_nonzero((X[:,:-1]-X[:,1:])==1), b_gama+np.sum(X)-np.count_nonzero((X[:,:-1]-X[:,1:])==1))
    while (gama_==-1)|(gama_==-2):
        if gama_==-1:
             
            hyper_params[6]=hyper_params[6]+5
            a_gama=hyper_params[6]
            gama_=Sample_gama(a_gama +np.count_nonzero((X[:,:-1]-X[:,1:])==1), b_gama+np.sum(X)-np.count_nonzero((X[:,:-1]-X[:,1:])==1))
            
        if gama_==-2:
        
            hyper_params[7]=hyper_params[7]+1
            b_gama=hyper_params[7]
            gama_=Sample_gama(a_gama +np.count_nonzero((X[:,:-1]-X[:,1:])==1), b_gama+np.sum(X)-np.count_nonzero((X[:,:-1]-X[:,1:])==1))
            
    theta_1_=Sample_theta1( a_teta1+TP,b_teta1+np.sum(unique_rows.dot(X))-TP)
    while theta_1_==-1:
        hyper_params[10]=hyper_params[10]+500
        a_teta1=hyper_params[10]
        theta_1_=Sample_theta1( a_teta1+TP,b_teta1+np.sum(unique_rows.dot(X))-TP)
        
    theta_0_=Sample_theta0( a_teta0+FP,b_teta0+np.count_nonzero((unique_rows.dot(X))==0)-FP)
    while theta_0_==-1:
        hyper_params[8]=hyper_params[8]+500
        a_teta0=hyper_params[8]
        theta_0_=Sample_theta0( a_teta0+FP,b_teta0+np.count_nonzero((unique_rows.dot(X))==0)-FP)    
   
    param=[alpha_,beta_,betaf,gama_,theta_0_,theta_1_]
    R=R_(G,X,param,F)
         
    return param,R


def Sample_alpha(a_alpha, b_alpha):
    for i in beta.rvs(a_alpha, b_alpha, size=10000):
        if (i>0.001)&(i<0.00501):
            alpha_=round(i,3)
            return alpha_ 
    if not(i>0.001):
        return -1.0
    else:
        return -2.0
    
def Sample_beta(a_beta, b_beta):
    for i in beta.rvs(a_beta, b_beta, size=10000):
        if (i>0.001)&(i<0.00501):
            beta_=round(i,4)
            return beta_ 
    if not(i>0.001):
        return -1.0
    else:
        return -2.0                

def Sample_betaf(a_betaf, b_betaf):
    for i in beta.rvs(a_betaf, b_betaf, size=1000):
        if (i>0.002)&(i<0.00501):
            betaf=round(i,4)
            return betaf
    if not(i>0.002):
        return -1.0
    else:
        return -2.0
def Sample_gama(a_gama,b_gama):
    for i in beta.rvs(a_gama, b_gama, size=10000):
        if (i>0.1)&(i<0.5):
            gama_=round(i,3)
            return gama_  
    if not(i>0.07):
        return -1.0
    else:
        return -2.0           

def Sample_theta0(a_teta0, b_teta0):
    for i in beta.rvs(a_teta0, b_teta0, size=10000):
        if (i>0.01)&(i<0.03):
            theta_0_=round(i,3)
            return theta_0_  
    if not(i>0.01):
        return -1.0
    else:
        return -2.0  

def Sample_theta1(a_teta1, b_teta1):
    for i in beta.rvs(a_teta1, b_teta1, size=10000):
        if i>0.799:
            theta_1_=round(i,3)
            return theta_1_ 
    if not(i>0.78):
        return -1

    
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
    while alpha_==-1:
        a_alpha=a_alpha+5
        hyper_params[0]=a_alpha
        alpha_=Sample_alpha(a_alpha, b_alpha)
    while alpha_==-2:
        b_alpha=b_alpha+500
        hyper_params[1]=b_alpha
        alpha_=Sample_alpha(a_alpha, b_alpha)
    beta_=Sample_beta(a_beta, b_beta)
    while beta_==-1:
        a_beta=a_beta+5
        hyper_params[2]=a_beta
        beta_=Sample_beta(a_beta, b_beta)
    while beta_==-2:
        b_beta=b_beta+500
        hyper_params[3]=b_beta
        beta_=Sample_beta(a_beta, b_beta)    
    betaf=Sample_betaf(a_betaf, b_betaf)
    while alpha_>beta_:
        alpha_=Sample_alpha(a_alpha, b_alpha)
        beta_=Sample_beta(a_beta, b_beta)
    while beta_>betaf:
        betaf=Sample_betaf(a_betaf, b_betaf)
    gama_=Sample_gama(a_gama,b_gama)
    theta_0_=Sample_theta0(a_teta0, b_teta0)
    while theta_0_==-1:
        a_teta0=a_teta0+5
        hyper_params[8]=a_teta0
        theta_0_=Sample_theta0(a_teta0, b_teta0)
    while theta_0_==-2:
        b_teta0=b_teta0+500
        hyper_params[9]=b_teta0
        theta_0_=Sample_theta0(a_teta0, b_teta0)       
    theta_1_=Sample_theta1(a_teta1, b_teta1)
    while theta_1_==-1:
        a_teta1=a_teta1+5
        hyper_params[8]=a_teta1
        theta_1_=Sample_theta1(a_teta0, b_teta0)
   
    params=np.array([alpha_,beta_,betaf,gama_,theta_0_,theta_1_])
    return params

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

# Gibbs sampling to obtain X, as new sample of posterior distribution:
def Calculate_X(pos_probs,K,X,G,F,Y,params,P):
    n,T=X.shape[0],X.shape[1]
    
    for k in range(K):
        for t in range(T):
            hidden_states=Sample_hidden_state(pos_probs,X,G,F,Y,params,P,t)
            X=hidden_states[0]
            pos_probs=hidden_states[1]
    
    return X  ,pos_probs 

# funtion to retun related family index of individual i:
def family_index(i,unique_rows):
    n=unique_rows.shape[1]
    for j in range(n):
        if unique_rows[j,i]==1:
            return j








