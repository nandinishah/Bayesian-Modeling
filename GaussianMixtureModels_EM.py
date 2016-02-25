# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 20:15:52 2015

@author: nandinishah

Expectation Maximization
HW 4 - Problem 1
Bayesian Modeling
Sem III, Columbia University

Data: Given observations X = {x1, . . . , xn} where each x_i E Rd. 
Assumption: As being generated from a Gaussian mixture model.
Model: EM algorithm for learning maximum likelihood values of pi and each (mu, sigma) for j = 1,2,...K.

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import copy

# importing the data
path = ""
file = open(path+'data.txt')
x = np.genfromtxt(file,delimiter=",")
X = np.matrix(x)

# plotting the data
plt.scatter(X[:,0],X[:,1])

# initializing parameters
K = [2,4,8,10]
#K=[2]
for k in K:
    #k = 2
    d = len(x[0])
    n = len(x[:,0])
    
    mean = [copy.copy(x[i]) for i in range(0,k)]
    covar = [np.identity(d) for i in range(0,k)]
    pi = [1.0/k for i in range(0,k)]
    phi = np.zeros((n,k))
    sum = np.zeros(n)
    nj = np.zeros(k)
    maxiter = 100
    Likelihood=np.zeros(maxiter)
    
    for iter in range(0,maxiter):
        # E step
        sum = np.zeros(n)
        for cluster in range(0,k):
            normfn = multivariate_normal(mean[cluster],covar[cluster])
            phi[:,cluster]=normfn.pdf(x)*pi[cluster]
            sum = sum+phi[:,cluster]
        for cluster in range(0,k):    
            phi[:,cluster]=phi[:,cluster]/sum
        
        # M step
        for cluster in range(0,k):
            nj[cluster] = np.sum(phi[:,cluster])
            mean[cluster][0]=(np.dot(phi[:,cluster],x[:,0]))/nj[cluster]
            mean[cluster][1]=(np.dot(phi[:,cluster],x[:,1]))/nj[cluster]        
            covar[cluster] = np.transpose(x-mean[cluster]).dot(np.diag(phi[:,cluster]))   
            covar[cluster] = (covar[cluster].dot(x-mean[cluster]))/float(nj[cluster])   
            pi[cluster]=nj[cluster]/float(n)
            
        # Likelihood calculation
        L = 0
        for cluster in range(0,k):
            normfn = multivariate_normal(mean[cluster],covar[cluster])
            term1 =  np.log(normfn.pdf(x))+math.log(pi[cluster])       
            L = L + np.dot(phi[:,cluster],term1)
        Likelihood[iter]=L
    
    y = np.zeros(n)
    for i in range(0,n):
        y[i]=np.argmax(phi[i,:])
    
    # Problem 2b - Plotting log likelihood vs iterations
    plt.ylabel('log likelihood')
    plt.xlabel('# iterations')
    plt.title('# clusters: '+str(k))
    plt.plot(Likelihood)
    plt.savefig(path+'fig/Problem1-LogLikelihood1-'+str(k)+'.png')    
    plt.close() 
    
    # Problem2c - Plotting scatterplots of clusters 
    plt.ylabel('X[1]')
    plt.xlabel('X[0]')
    plt.title('# clusters: '+str(k))
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.savefig(path+'fig/Problem1-ScatterPlot1-'+str(k)+'.png')    
    plt.close()
