#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 13:10:06 2022

@author: magnusaxen
"""


from labfuns import genBlobs,plotGaussian
from labfuns import testClassifier,plotBoundary
import numpy as np
#Assignment 1
def mlParams(X,y):
    
    
    #This says how many unique labels we have
    classes=np.unique(y)
    d=np.shape(X)[1] #How many "sets" of points
    #We wish now to categorize them
    mu = np.zeros((len(classes),d))
    sigma = np.zeros((len(classes),d,d))
    #Mean calculation
    for jdx,Class in enumerate(classes):
        idx = y==Class # Returns a true or false with the length of y
        # Or more compactly extract the indices for which y==class is true,
        # analogous to MATLAB’s find
        idx = np.where(y==Class)[0]
        xlc = X[idx,:] # Get the x for the class labels. Vectors are rows.

        for i in range(d):
            mean=np.mean(xlc[:,i])
            mu[jdx,i]=mean
    
    #Sigma calculation
    for jdx,Class in enumerate(classes):
        idx = y==Class # Returns a true or false with the length of y
        # Or more compactly extract the indices for which y==class is true,
        # analogous to MATLAB’s find
        idx = np.where(y==Class)[0]
        xlc = X[idx,:] # Get the x for the class labels. Vectors are rows.
        for i in range(d):
            
            diff=(xlc[:,i]-mu[jdx,i])*(xlc[:,i]-mu[jdx,i])
            summ=sum(diff)/len(xlc)
            
            sigma[jdx,i,i]=summ
    
    
    return mu,sigma


X,y=genBlobs()
mu,sigma=mlParams(X,y)
plotGaussian(X,y,mu,sigma)



#Assignment 2
def computePrior(y):
    classes=np.unique(y)
    
    priors=np.zeros((len(y)))
    for jdx,Class in enumerate(classes):
        idx = y==Class # Returns a true or false with the length of y
        # Or more compactly extract the indices for which y==class is true,
        # analogous to MATLAB’s find
        idx = np.where(y==Class)[0]
        xlc = X[idx,:] # Get the x for the class labels. Vectors are rows.
        priors[jdx]=len(xlc)/len(y)
        
    return priors

prior=computePrior(y)




def classifyBayes(X,prior,mu,sigma):
    
    #
    r=np.shape(X)[0]
    classes=np.shape(mu)[0]
    delta_xs=np.zeros((classes,r))
    classes=np.unique(y)
    
    for jdx,Class in enumerate(classes):
        idx = y==Class # Returns a true or false with the length of y
        # Or more compactly extract the indices for which y==class is true,
        # analogous to MATLAB’s find
        #idx = np.where(y==Class)[0]
        print(idx)
        print(X[idx,:])
        xlc = X[idx,:] # Get the x for the class labels. Vectors are rows.
        
        Counter=0
        difference= X-mu[jdx]
        for i in range(r):
            delta_x=-1/2*np.log(np.linalg.det(sigma[jdx]))
            print(delta_x)
            print("3")
            print(np.shape(delta_x))
            print(jdx)
            print(Counter)
            delta_xs[jdx,Counter]=delta_x
            Counter+=1
        
    #Max?
    return delta_xs


#Assignment 3
# NOTE: no need to touch this
class BayesClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, X, labels, W=None):
        rtn = BayesClassifier()
        rtn.prior = computePrior(labels, W)
        rtn.mu, rtn.sigma = mlParams(X, labels, W)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBayes(X, self.prior, self.mu, self.sigma)
    
    


testClassifier(classifyBayes(X,prior,mu,sigma), dataset='iris', split=0.7)

testClassifier(classifyBayes(X,prior,mu,sigma), dataset='vowel', split=0.7)

plotBoundary(classifyBayes(X,prior,mu,sigma), dataset='vowel',split=0.7)


