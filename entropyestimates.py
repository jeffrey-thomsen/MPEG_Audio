# -*- coding: utf-8 -*-
"""
Created on Tue May 19 01:02:12 2020

@author: Jeffrey
"""
import numpy as np

def idealadaptivecodelength(x):
    # x - data sequence
    
    # returns the dictionary A of unique values occuring in x
    A = np.unique(x)
    
    lenA = len(A) # dictionary length
    lenX = len(x) # data sequence length
    
    # initialize a counter array to contain the number of occurances for each
    # dictionary entry
    counter = np.zeros(lenA,dtype=int)
    
    # initialize a p_hat array to subsequently store the probability estimates
    p_hat=np.zeros(lenX)
    
    for i in range(lenX):
        
        ind = A==x[i] # return which dictionary index the current sample represents
        
        counter[ind] += 1 # increment the counter at that index
    
        p_hat[i] = (counter[ind]+1) / (i+lenA*1) # Bayes' estimate of prob.
            
    
    
    iacl = - np.sum(np.log2(p_hat)) / lenX

    
    return iacl



def empiricalselfentropy(x):
    # x - data sequence
    
    # return the dictionary A of unique values occuring in x and a 
    # corresponding array counts with the number of occurances for each 
    # dictionary entry
    A, counts = np.unique(x,return_counts=True)
    
    # calculate probability estimates
    normcounts = counts/len(x)
    
    # calculate estimated self-entropy
    ie = -np.mean(np.log2(normcounts))
    
    return ie