# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 16:14:22 2015

@author: nandinishah

Variational Inference
HW 4 - Problem 2
Bayesian Modeling
Sem III, Columbia University

"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import math
from scipy import special as sp
from scipy.stats import dirichlet
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from scipy.special import gamma
#from scipy.stats import w

# importing the data
path = "/Users/nandinishah/Documents/Columbia/Sem3/BayesianModels/HW4/"
file = open(path+'data.txt')
x = np.genfromtxt(file,delimiter=",")

# initializing the parameters
K=[2,4,10,25]
#K=[4]
d = len(x[0])
n = len(x)

for k in K:
    #k=2
    alpha = 1
    alpha_dash = [1.0/k for i in range(0,k)]
    a = d
    a_dash = [copy.copy(d) for i in range(0,k)]
    s = 10
    A = np.cov(np.transpose(x))
    B = (d/10.0)*A
    B_dash = [copy.copy((d/10.0)*A) for i in range(0,k)]
    
    #m_dash = [copy.copy(x[i]) for i in range(0,k)]  # kmeans
    model = KMeans(n_clusters=k)
    model.fit(x)
    m_dash = model.cluster_centers_
    covar_dash = [np.identity(d) for i in range(0,k)]   # cov of data
    phi = np.zeros((n,k))
    n_j = np.zeros(k)
    
    maxiter = 100
    Likelihood = np.zeros(maxiter)
    Likelihood2 = np.zeros(maxiter)
    
    for iter in range(0,maxiter):
        # (a) Updating q(c_i)
        sum_phi=np.zeros(n)
        for cluster in range(0,k):
            term1 = sp.psi((a_dash[cluster])/2.0)+sp.psi((a_dash[cluster]-1)/2.0) - math.log(np.linalg.det(B_dash[cluster]))
            #print iter, cluster, term1      
            term = np.dot((x-m_dash[cluster]),(a_dash[cluster]*np.linalg.inv(B_dash[cluster])))
            term2 = np.diag(np.dot(term,np.transpose(x-m_dash[cluster])))
            term = np.dot((a_dash[cluster]*np.linalg.inv(B_dash[cluster])),covar_dash[cluster])
            term3 = np.trace(term)
            term4 = sp.psi(alpha_dash[cluster])-sp.psi(np.sum(alpha_dash))
            phi[:,cluster] = np.exp(0.5*term1-0.5*term2-0.5*term3+term4)
            sum_phi+=phi[:,cluster]
        for cluster in range(0,k):
            phi[:,cluster] = phi[:,cluster]/sum_phi[:]
            n_j[cluster] = np.sum(phi[:,cluster]) # (b) Setting n_j
            
        # (c) Updating q(pi)
        alpha_dash = alpha + n_j
        
        # (d) Updating q(mu)
        for cluster in range(0,k):
            term = (1.0/s)*np.identity(d) + n_j[cluster]*a_dash[cluster]*np.linalg.inv(B_dash[cluster])
            covar_dash[cluster] = np.linalg.inv(term)
            term = np.dot((a_dash[cluster]*np.linalg.inv(B_dash[cluster])),np.dot(np.transpose(x),phi[:,cluster]))        
            m_dash[cluster] = np.dot(covar_dash[cluster],term)        
            #term = np.dot(covar_dash[cluster],(a_dash[cluster]*np.linalg.inv(B_dash[cluster])))
            #m_dash[cluster] = np.dot(np.dot(np.transpose(phi[:,cluster]),x),term)  
        
        # (e) Updating q(lambda_j)
        for cluster in range(0,k):
            a_dash[cluster] = a + n_j[cluster]
            sum_terms = np.zeros((d,d))
            for point in range(0,n):
                term=np.matrix(x[point,:]-m_dash[cluster])
                sum_terms+= phi[point,cluster]*np.dot(np.transpose(term),term)+covar_dash[cluster]
            B_dash[cluster] = B + sum_terms
        
        # (f) Variational objective function
        Eq_lnlambda_j = np.zeros(k)
        Eq_lambda_j = [np.zeros((d,d)) for i in range(0,k)]
        Eq_lnpi_j = np.zeros(k)
        term1 = 0.0
        term2 = 0.0
        term2b = 0.0
        term3 = 0.0
        term3b = 0.0
        term4 = 0.0
        term5a = 0.0
        Entropy_c_i = 0.0
        
        for cluster in range(0,k):
            Eq_lambda_j[cluster] = a_dash[cluster]*np.linalg.inv(B_dash[cluster])
            Eq_lnlambda_j[cluster] = (d*(d-1)/4.0)*math.log(math.pi) + d*math.log(2) - math.log(np.linalg.det(B_dash[cluster])) + sp.psi(a_dash[cluster]*0.5 + cluster*0.5) + sp.psi(a_dash[cluster]*0.5) + sp.psi(a_dash[cluster]*0.5 - 0.5) 
            #term = (x-m_dash[cluster])
            #term1 = np.diag(np.dot(np.dot(term,Eq_lambda_j[cluster]),np.transpose(term)))
            #term1 += np.trace(np.dot(Eq_lambda_j[cluster],covar_dash[cluster]))
            term2b = term2b + np.trace(covar_dash[cluster] + np.dot(np.transpose(np.matrix(m_dash[cluster])),np.matrix(m_dash[cluster])))
            term3b = term3b + np.trace(np.dot(B,Eq_lambda_j[cluster]))
            Eq_lnpi_j[cluster] = (sp.psi(alpha_dash[cluster]) - sp.psi(np.sum(alpha_dash)))
            term5a = term5a + (n_j[cluster])*Eq_lnpi_j[cluster]
            term4 = term4 + (alpha_dash[cluster]-1)*Eq_lnpi_j[cluster]
            
        for element in range(0,n):
            for cluster in range(0,k):
                term = np.matrix(x[element]-m_dash[cluster])
                term1a = 0.5*Eq_lnlambda_j[cluster] - 0.5*(np.dot(np.dot(term,Eq_lambda_j[cluster]),np.transpose(term))) - 0.5*np.trace(np.dot(Eq_lambda_j[cluster],covar_dash[cluster]))
                term1 = term1 + phi[element,cluster]*term1a
            Entropy_c_i = Entropy_c_i + np.dot(phi[element,:],np.log(phi[element,:]))
        term1 = term1 -(n*d*0.5*math.log(2*math.pi)) 
        term2 = -k*d*0.5*math.log(math.pi*2*s) - 0.5/s*term2b
        term3 = (a-d-1)*0.5*np.sum(Eq_lnlambda_j) - 0.5*term3b
        #term5 = n*np.sum(Eq_lnpi_j)
        term5 = term5a
    
    
        # Entropies
        Entropy = 0.0
        Entropy_lambda_j = 0.0
        for cluster in range(0,k):
            Entropy+=multivariate_normal.entropy(m_dash[cluster],covar_dash[cluster])
            Entropy_lambda_j += ((d+1)*0.5*np.log(np.linalg.det(np.linalg.inv(B_dash[cluster]))) + 0.5*d*(d+1)*math.log(2) + math.log(gamma(a_dash[cluster]*0.5)) + math.log(gamma(a_dash[cluster]*0.5-0.5)) - ((a_dash[cluster]-d-1)*0.5)*sp.psi(a_dash[cluster]*0.5) + n*d*0.5)
        Entropy  = Entropy + dirichlet.entropy(alpha_dash) + Entropy_c_i +Entropy_lambda_j
        # to add entropy of c_i from 1 to n -- multinomial
        
        Likelihood[iter] = term1+term2+term3+term4+term5+Entropy

    
    # Plotting likelihood plot
    y = np.zeros(n)
    for i in range(0,n):
        y[i]=np.argmax(phi[i,:])
        
    # Problem 3b - Plotting log likelihood vs iterations
    plt.ylabel('log likelihood')
    plt.xlabel('# iterations')
    plt.title('# clusters: '+str(k))
    plt.plot(Likelihood)
    plt.savefig(path+'fig/Problem2/Problem2-LogLikelihood-'+str(k)+'.png')    
    plt.close()
  
    # Problem3c - Plotting scatterplots of clusters 
    plt.ylabel('X[1]')
    plt.xlabel('X[0]')
    plt.title('# clusters: '+str(k))
    plt.scatter(x[:,0],x[:,1],c=y)
    plt.savefig(path+'fig/Problem2/Problem2-ScatterPlot-'+str(k)+'.png')    
    plt.close() 
