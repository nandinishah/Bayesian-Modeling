# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 18:33:52 2015

@author: nandinishah

APPLICATION OF EXPECTATION MAXIMIZATION ALGORTHM TO OCR (mnist) DATASET.

Data: Post PCA, 15 dimensional mnist data set. Each 15-dimensional vector x has a label y with y = 0 indicating “4” and y = 1 indicating “9”.

Model: parameter setting σ = 1.5 and λ = 1


"""

import numpy as np
from scipy.stats import norm
import math

# Reading input data
folder = ''
fileXtest = folder+'Xtest.csv'
fileXtrain = folder+'Xtrain.csv'
fileYtest = folder+'Ytest.csv'
fileYtrain = folder+'Ytrain.csv'
fileQ = folder+'Q.csv'

Xtest=np.genfromtxt(fileXtest,delimiter=",")
Xtrain=np.genfromtxt(fileXtrain,delimiter=",")
Ytest=np.genfromtxt(fileYtest,delimiter=",")
Ytrain=np.genfromtxt(fileYtrain,delimiter=",")
Q=np.genfromtxt(fileQ,delimiter=",")


# initializing variables
d = len(Xtrain[0])
N = len(Xtrain)
sigma=1.5
lam=1.0
T=100
w = np.array([0 for i in range(0,d)])
one = np.array([1 for i in range(0,len(Ytrain))])
Ytrain2 = np.logical_xor(Ytrain,one)
loglikeAll = [0.0 for i in range(0,T)]
wt = []

# EM algorithm implementation
for t in range(0,T):
    # E step
    s = -1*(Xtrain.dot(w))/sigma
    probdf = norm.pdf(s)
    cumdf = norm.cdf(s)
    Eq1 = Xtrain.dot(w)+(sigma*probdf/(1-cumdf))
    Eq1 = np.multiply(Eq1,Ytrain)
    Eq0 = Xtrain.dot(w)+(-sigma*probdf/(cumdf))
    Eq0 = np.multiply(Eq0,Ytrain2)
    Eq=Eq1+Eq0
    
    # M step
    term1 = lam*np.eye(d)+Xtrain.transpose().dot(Xtrain)/(sigma**2)
    term1 = np.linalg.inv(term1)
    term2 = Xtrain.transpose().dot(Eq)/(sigma**2)
    w = term1.dot(term2)
    
    # Assessing convergence through likelihood
    cumdf = norm.cdf(-s)
    loglike=(d/2.0)*math.log(lam/(2*math.pi))-(lam/2.0)*w.dot(w)+Ytrain.dot(np.log(cumdf))+Ytrain2.dot(np.log(1-cumdf))
    loglikeAll[t]=loglike
    
    if t in [0,4,9,24,49,99]:
        wt.append(w)


# Plotting log likelihood as a function of t
import matplotlib.pyplot as plt
plt.plot(loglikeAll)
plt.ylabel('log likelihood')
plt.show()


# Predictions for test set
confusionMatrix = [[0,0],[0,0]]
s=(Xtest.dot(w))/sigma
cumdf=norm.cdf(s)
Ypredict = [0 for i in range(0,len(Ytest))]
misclassified = []
ambiguous = []
for i in range(0,len(Ytest)):
    if cumdf[i]<0.5: 
        Ypredict[i]=0
    else:
        Ypredict[i]=1

    if Ypredict[i]==Ytest[i] and Ytest[i]==0:
        confusionMatrix[0][0]+=1
    elif Ypredict[i]!=Ytest[i] and Ytest[i]==0:
        confusionMatrix[1][0]+=1
        misclassified.append(i)
    elif Ypredict[i]==Ytest[i] and Ytest[i]==1:
        confusionMatrix[1][1]+=1
    else:
        confusionMatrix[0][1]+=1
        misclassified.append(i)
    if abs(cumdf[i]-0.5)<0.0045:
        ambiguous.append(i)
            
# Confusion matrix
print "Confusion matrix:"
print "-"*20
print "Number of 4's classified as 4's:",confusionMatrix[0][0]
print "Number of 4's classified as 9's:",confusionMatrix[1][0]
print "Number of 9's classified as 4's:",confusionMatrix[0][1]
print "Number of 9's classified as 9's:",confusionMatrix[1][1]
err = (confusionMatrix[1][0]+confusionMatrix[0][1])/float(len(Xtest))*100
print "Test error:",err,"%"
print "Accuracy:",100-err,"%"


# Printing misclassified digits
print "\nMisclassified digits:"
print "-"*20
import scipy.misc as sm
for j in range(0,3):
    image = Q.dot(Xtest[misclassified[j]])
    i = np.array(image)
    i = i.reshape((28, 28))
    filename = '/Users/nandinishah/Documents/Columbia/Sem3/BayesianModels/HW2/'+'imageQ2d_'+str(j)+'.png'
    sm.imsave(filename,i)
    print "Predictive prob of image",j,":",cumdf[misclassified[j]]


# Printing ambiguous digits
print "\nAmbiguous digits:"
print "-"*20
for j in range(0,3):
    image = Q.dot(Xtest[ambiguous[j]])
    i = np.array(image)
    i = i.reshape((28, 28))
    filename = '/Users/nandinishah/Documents/Columbia/Sem3/BayesianModels/HW2/'+'imageQ2e_'+str(j)+'.png'
    sm.imsave(filename,i)
    print "Predictive prob of image",j,":",cumdf[ambiguous[j]]

count = 0
for j in wt:
    image = Q.dot(j)
    i = np.array(image)
    i = i.reshape((28, 28))
    filename = '/Users/nandinishah/Documents/Columbia/Sem3/BayesianModels/HW2/'+'imageQ2f_'+str(count)+'.png'
    sm.imsave(filename,i)
    count=count+1

    





