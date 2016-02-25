# -*- coding: utf-8 -*-
"""
Created on Fri Oct 2 20:15:52 2015

@author: nandinishah

APPLICATION OF HIERARCHICAL NAIVE BAYES ALGORTHM TO OCR (mnist) DATASET.

Data: Post PCA, 15 dimensional mnist data set. Each 15-dimensional vector x has a label y with y = 0 indicating “4” and y = 1 indicating “9”.

Model: labels as y ~ Bernoulli(p), assume each x_i dimension with independent normal-gamma priors (mu,lambda).


"""

from pandas import DataFrame as df
import pandas as pd
from scipy.stats import t
import math


##### reading the data #####
Xtest = pd.read_csv('./Xtest.csv',header=None,sep=',',index_col=False)
Ytest = df.from_csv('./Ytest.csv',header=None,sep=',',index_col=False)
Xtrain = df.from_csv('./Xtrain.csv',header=None,sep=',',index_col=False)
Ytrain = df.from_csv('./Ytrain.csv',header=None,sep=',',index_col=False)
Ytrain.columns=['label']
Xtrain.columns=[range(0,15)]

##### initializing variables #####
a=1
b=1
c=1
e=1
f=1
N=len(Ytrain)

##### calculating prior on Y using training labels #####
def calcPriorY():
    countY1 = Ytrain.sum()
    countY0 = len(Ytrain)-Ytrain.sum()
    priorY=[0,0]
    priorY[0]=(e+countY0)/(N+e+f)
    priorY[1]=(e+countY1)/(N+e+f)
    return priorY

priorY=calcPriorY()


##### calculating params of the t-distribution on each dimension of X #####
trainData = pd.concat([Xtrain,Ytrain],axis=1)

trainData.sort(columns=['label'],inplace=True)

def calcParams(thislabel):
    data = trainData.loc[trainData.label==thislabel]
    n=len(data)
    m = data.mean()
    s = data.std()  
    mu0=[0 for i in xrange(len(m)-1)]
    lambda0=[0 for i in xrange(len(m)-1)]
    b0=[0 for i in xrange(len(m)-1)]
    c0=[0 for i in xrange(len(m)-1)]
    tdof=[0 for i in xrange(len(m)-1)]
    tlambda=[0 for i in xrange(len(m)-1)]
    tmu=[0 for i in xrange(len(m)-1)]
    for i in xrange(len(m)-1):
        mu0[i] = float(a*n)/float(a*n+1)*m[i]    
        lambda0[i] = n+(1.0/a)    
        b0[i] = n/2.0+b-0.5
        c0[i] = c+(float((n-1)*s[i])/2.0)+(m[i]*m[i]*n/2.0)/(a*n+1.0)    
        tdof[i] = 2*b0[i]      
        tlambda[i] = float(b0[i])*float(lambda0[i])/float(c0[i]*(1+lambda0[i]))         
        tmu[i]=mu0[i]
    return [tdof,tlambda,tmu]


##### predicting Y values for test set based on params #####
params = calcParams(0)
params0DF = df(params,index=['tdof','tlambda','tmu'])
params = calcParams(1)
params1DF = df(params,index=['tdof','tlambda','tmu'])
Ypredict = [[0 for i in xrange(len(Ytest))],[0 for i in xrange(len(Ytest))]]
for element in xrange(len(Xtest)):
    prob0 = 1
    prob1 = 1
    for dim in xrange(len(Xtest.iloc[element])):
        scaledX = float(Xtest.iloc[element][dim]-params0DF.loc['tmu'][dim])*math.sqrt(params0DF.loc['tlambda'][dim])   
        prob0 = prob0*t.pdf(scaledX,params0DF.loc['tdof'][dim])
        scaledX = float(Xtest.iloc[element][dim]-params1DF.loc['tmu'][dim])*math.sqrt(params1DF.loc['tlambda'][dim])   
        prob1 = prob1*t.pdf(scaledX,params1DF.loc['tdof'][dim])
    prob0 = float(prob0)*float(priorY[0])
    prob1 = float(prob1)*float(priorY[1])
    finalprob0 = float(prob0)/float(prob0+prob1)
    finalprob1 = float(prob1)/float(prob0+prob1)

    if finalprob1>finalprob0: 
        Ypredict[0][element]=1
        Ypredict[1][element]=float(finalprob1)
    else: 
        Ypredict[0][element]=0
        Ypredict[1][element]=float(finalprob0)


##### Confusion Matrix Calculation #####
confusionMatrix = [[0,0],[0,0]]
misclassified = [] # for part 3c
for i in xrange(len(Ytest)):
    if Ytest[0][i]!=Ypredict[0][i] and Ytest[0][i]==1:
        confusionMatrix[1][0]=confusionMatrix[1][0]+1
        misclassified.append(i)
    elif Ytest[0][i]!=Ypredict[0][i] and Ytest[0][i]==0:
        confusionMatrix[0][1] = confusionMatrix[0][1]+1
        misclassified.append(i)
    elif Ytest[0][i]==Ypredict[0][i] and Ytest[0][i]==1:
        confusionMatrix[1][1]=confusionMatrix[1][1]+1
    else:
        confusionMatrix[0][0] = confusionMatrix[0][0]+1
print "CONFUSION MATRIX"
print "----------------"
print "# 4's classified as a 4:",confusionMatrix[0][0]
print "# 4's classified as a 9:",confusionMatrix[0][1]
print "# 9's classified as a 4:",confusionMatrix[1][0]
print "# 9's classified as a 9:",confusionMatrix[1][1]

error = float(confusionMatrix[1][0]+confusionMatrix[0][1])/float(len(Ytest[0]))*100.0
accuracy = float(confusionMatrix[1][1]+confusionMatrix[0][0])/float(len(Ytest[0]))*100.0

print "Test error:", (error),"%"
print "Accuracy:",(accuracy),"%"



##### Reconstructing three misclassified digits #####
import numpy as np
import scipy.misc as sm

# Q.csv contains the transformation vector from 15 to 784 dimensions #
Q = pd.read_csv('./Q.csv',header=None,sep=',',index_col=False)
predictiveProb3c = [[0,0,0],[0,0,0],[0.0,0.0,0.0]]
for element in xrange(3):
    image = [0 for i in xrange(len(Q))]
    for row in xrange(len(Q)):
        result = Q.iloc[row].multiply(Xtest.iloc[misclassified[element]])
        image[row] = result.sum()
    i = np.array(image)
    i = i.reshape((28, 28))
    filename = './'+'imageQ3c_'+str(element)+'.png'
    sm.imsave(filename,i)
    predictiveProb3c[0][element]=Ytest[0][misclassified[element]]
    predictiveProb3c[1][element]=Ypredict[0][misclassified[element]]
    predictiveProb3c[2][element]=Ypredict[1][misclassified[element]]
predictiveProb3cDF = df(predictiveProb3c,index=['Ytest','Ypredicted','predictiveprobability'])
print "Reconstructing three misclassified digits:"
print "------------" 
print predictiveProb3cDF



##### Some ambiguous predictions #####
Ypreddf = df(Ypredict)
Ypreddf = Ypreddf.transpose()
Ypreddf.columns=['label','prob']
YpredAmbiguous = Ypreddf.loc[Ypreddf.prob>0.495]
YpredAmbiguous = Ypreddf.loc[Ypreddf.prob<0.505]
YpredAmbiguous.sort(columns=['prob'],inplace=True)
predictiveProb3d = [[0,0,0],[0,0,0],[0.0,0.0,0.0]]
for element in xrange(3):
    image = [0 for i in xrange(len(Q))]
    for row in xrange(len(Q)):
        result = Q.iloc[row].multiply(Xtest.iloc[YpredAmbiguous.index[element]])
        image[row] = result.sum()
    i = np.array(image)
    i = i.reshape((28, 28))
    filename = './'+'imageQ3d_'+str(element)+'.png'
    sm.imsave(filename,i)
    predictiveProb3d[0][element]=Ytest[0][YpredAmbiguous.index[element]]
    predictiveProb3d[1][element]=Ypredict[0][YpredAmbiguous.index[element]]
    predictiveProb3d[2][element]=Ypredict[1][YpredAmbiguous.index[element]]
predictiveProb3dDF = df(predictiveProb3d,index=['Ytest','Ypredicted','predictiveprobability'])
#predictiveProb3dDF.columns=[YpredAmbiguous.index[0],YpredAmbiguous.index[1],YpredAmbiguous.index[2]]
print "Some ambiguous predictions:"
print "------------" 
print predictiveProb3dDF