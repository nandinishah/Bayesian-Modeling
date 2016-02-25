# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:25:21 2015

@author: nandinishah

Gibbs Sampling
HW 4 - Problem 3
Bayesian Modeling
Sem III, Columbia University

Data: Given observations X = {x1, . . . , xn} where each x_i E Rd. 
Assumption: As being generated from a Gaussian mixture model. mu/lambda ~ Normal(m,c*lambda_inv). lambda ~ Wishart(a,B).
Model: Gibbs sampling algorithm.

"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import multivariate_normal
from scipy.special import gamma
from scipy.stats import wishart


# importing the data
#path = "/Users/nandinishah/Documents/Columbia/Sem3/BayesianModels/HW4/"
path = ""
file = open(path+'data.txt')
x = np.genfromtxt(file,delimiter=",")

n = len(x)
d = len(x[0])
m = np.mean(x,axis=0)
c = 0.1
a = d
A = np.cov(np.transpose(x))
B = c*d*A
alpha = 1.0
K = 1

maxiter = 500
lines = [np.zeros(maxiter) for i in range(0,6)]
cluster_count = np.zeros(maxiter)

class Theta(object):
    def __init__(self):
        self.m_dash = [0.0,0.0]
        self.mu = [0.0,0.0]        
        self.sigma = np.zeros((d,d))
        self.s = 0.0
        self.c_dash = 0.0
        self.a_dash = 0.0
        self.B_dash = np.zeros((d,d))
    def compute(self,index):
        sumterm = [0.0,0.0]
        self.s = 0
        Bterm = np.zeros((d,d))
        for i in range(0,len(c_i)):
            if c_i[i] == index:            
                sumterm += x[i]
                self.s += 1
        for i in range(0,len(c_i)):
            if c_i[i] == index: 
                term = np.matrix(x[i]-sumterm/self.s)
                Bterm = Bterm + np.dot(np.transpose(term),term)
        self.m_dash = (c/(self.s+c))*m + (1/(self.s+c))*sumterm
        self.c_dash = self.s + c
        self.a_dash = a + self.s
        term = np.matrix(sumterm/self.s - m)
        self.B_dash = B + Bterm + (self.s/(a*self.s+1))*np.dot(np.transpose(term),term)
        # random sampling from distribution        
        #self.sigma = a*np.linalg.inv(self.B_dash) ##### need to sample from wishart typically
        self.sigma = wishart.rvs(df=self.a_dash,scale=np.linalg.inv(self.B_dash))        
        self.mu = np.random.multivariate_normal(self.m_dash,self.sigma)


# assuming all parameters are right before the first iteration
c_i = np.zeros(n)
thetas = []
t1 = Theta()
t1.compute(0)
thetas.append(t1)
n_j = [n]
marginal = np.zeros(n)

def marginalComputation():
    term1 = (c/(math.pi*(1+c)))**(d/2.0)
    term3 = math.exp(math.log(gamma(a*0.5+0.5))-math.log(gamma(a*0.5-0.5)))
    for point_index in range(0,n):
        term = np.matrix(x[point_index]-m)
        term2 = B + (c/(1+c))*np.dot(np.transpose(term),term)   
        marginal[point_index] = term1 * ((np.linalg.det(term2)**(-(a+1)/2.0)) / (np.linalg.det(B)**(-a/2.0))) * term3
    
marginalComputation()

for iteration in range(0,maxiter):
    for point in range(0,n):
        phi = np.zeros(K+1)
        for cluster_index in range(0,K):
            # (1a)            
            phi[cluster_index] = multivariate_normal.pdf(x[point],thetas[cluster_index].mu,thetas[cluster_index].sigma) * n_j[cluster_index]/(alpha+n-1)
        # (1b)            
        phi[K]= (alpha/(alpha+n-1.0)) * marginal[point]
        # (1c)        
        normalizer = sum(phi)
        phi=phi/normalizer
        getindex = [i for i in range(0,K+1)]
        newcluster_index = np.dot(getindex,np.random.multinomial(1, phi))
        oldcluster_index = int(c_i[point])
        n_j[oldcluster_index] -= 1
        c_i[point] = newcluster_index
        
        # (1d) checking if assigned to the new cluster and re-calculating n_j
        if c_i[point] == K:
            K=K+1
            t1 = Theta()
            t1.compute(c_i[point])
            thetas.append(t1)
            n_j.append(1)
        else:
            n_j[newcluster_index] +=1

        # destroying clusters if empty, updating value of K, re-assigning c_i
        #for cluster_index in range(0,len(n_j)):
        cluster_index = 0        
        while cluster_index < len(n_j):        
            if n_j[cluster_index] == 0:
                thetas.pop(cluster_index)
                n_j.pop(cluster_index)
                K = K-1
                for element in range(0,n):
                    if c_i[element] >= cluster_index:
                        c_i[element] = c_i[element]-1
                cluster_index -= 1
            cluster_index += 1                    
    # (2) regenerating the thetas
    thetas = []
    for cluster_index in range(0,K):
        t1 = Theta()
        t1.compute(cluster_index)
        thetas.append(t1)
            
    # Question 2b
    ans = n_j
    ans = sorted(ans, reverse=True)
    for i in range(0,6):   
        lines[i][iteration] = ans[i]

    # Question 2c
    cluster_count[iteration]=K
    print iteration,K
   
# Plotting lines
x_axis = [i for i in range(0,maxiter)]
plt.title('Plotting lines')
plt.xlabel('# iteration')
plt.ylabel('Top 6 cluster contents')
plt.plot(x_axis,lines[0],x_axis,lines[1],x_axis,lines[2],x_axis,lines[3],x_axis,lines[4],x_axis,lines[5])  
plt.savefig(path+'Problem3-Lineplots.png')
plt.close()
 
           
# Plotting cluster counts
plt.xlabel('# iteration')
plt.ylabel('# clusters')
plt.title('Cluster counts')
plt.plot(x_axis,cluster_count)
plt.savefig(path+'Problem3-ClusterCounts.png')
plt.close()  
