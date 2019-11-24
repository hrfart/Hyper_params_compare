# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:58:58 2019

@author: hmnor
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 21:13:44 2019

@author: hmnor

https://machinelearningmastery.com/what-is-bayesian-optimization/


"""
	
# example of bayesian optimization for a 1d function from scratch
#from math import sin
#from math import pi
#from numpy import arange
#from numpy import vstack
#from numpy import argmax
from numpy import argmin
#from numpy import asarray
#from numpy.random import normal
#from numpy.random import random
from scipy.stats import norm
#from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
#from matplotlib import pyplot
#import numpy.matlib

from sklearn.gaussian_process import GaussianProcessRegressor

from model import runmodel

import numpy as np

#TODO: update to support harry's design
from config import *

 

""" Surrogate or approximation for the objective function
ESTIMATES COSE OF ONE OR MORE SAMPLES 

#will need to rewrite this to return prediction from GP model???
#think about architecture 
# """
#TODO: update with from scratch GP model
def surrogate(model, X):
    
    #Inputs:
    #model - GP scikit learn 
    # X - values to generate predictions for 
    
    #Returns 
    #pred - predictions from model for X values (Nx1)?
    
	# catch any warning generated when making a prediction
    with catch_warnings():
		# ignore generated warnings
        simplefilter("ignore")
        pred = model.predict(X, return_std=True)
        
        return pred #model.predict(X, return_std=True)
 
    
# probability of improvement acquisition function
def acquisition(X, Xsamples, model):
	# calculate the best surrogate score found so far
    yhat, _ = surrogate(model, X)
    best = max(yhat)
	# calculate mean and stdev via surrogate function
    mu, std = surrogate(model, Xsamples)
    mu = mu[:, 0]
    
	# calculate the probability of improvement
    """ Can update this to expected improvement - more commonly used """ 
    #TODO: update to be Expected Improvement instead of Probability of Improvement
    probs = norm.cdf((mu - best) / (std+1E-9))
    return probs
 
# optimize the acquisition function
def opt_acquisition(X, y, model):
	# random search, generate random samples
    #Xsamples = random(100)
    #Xsamples = Xsamples.reshape(len(Xsamples), 1)
    #Xsamples = np.matlib.repmat(Xsamples, 1, 7)

    Xsamples = get_hyperparameter_configuration(100)
    Xsamples = Xsamples.reshape(100, NUM_HYPERPARAMS)
    
	# calculate the acquisition function for each sample
    scores = acquisition(X, Xsamples, model)
       
	# locate the index of the smallest scores 
    ix = argmin(scores)
    
    #Get sample of best score 
    newX  = Xsamples[ix, :]
    newX = newX.reshape(1, np.shape(newX)[0])
    
    return newX
 
    

""" STOLEN FROM CONRAD - hyperband.py """ 
# Function to return a matrix T of hyperparameter configurations
# n = number of hyperparameter configurations to return, i.e. cardinality of T, where each element of T
#   is a different column/hyperparameter configuration
#
# NOTE: THIS FUNCTION READS GLOBALS FROM model.py FOR HYPERPARAM CONFIGS
def get_hyperparameter_configuration(n):
    # Create matrix/set of hyperparam configurations to investigate
    T = np.zeros((NUM_HYPERPARAMS, n)) 

    # Uniformly choose hyperparameters among different ranges
    for hpidx in range(n):
        T[0, hpidx] = layer_opts[np.random.randint(0,len(layer_opts))]
        T[1, hpidx] = node_opts[np.random.randint(0, len(node_opts))]
        T[2, hpidx] = learnrate_opts[np.random.randint(0, len(learnrate_opts))]
        T[3, hpidx] = beta1_opts[np.random.randint(0, len(beta1_opts))]
        T[4, hpidx] = beta2_opts[np.random.randint(0, len(beta2_opts))]
        T[5, hpidx] = eps_opts[np.random.randint(0, len(eps_opts))]
        T[6, hpidx] = decay_opts[np.random.randint(0, len(decay_opts))]
        

    return T


###########################  BAYESIAN OPTIMIZATION SECTION   ######################################################
#inputs are data set to use, and the loaded data
#outputs final val and test loss,best parameters chosen
def bayesian_optimization(dataset,trainx,trainy,valx,valy,testx,testy):

    #initialize best and current params
    initial_hyper_vals = np.zeros(NUM_HYPERPARAMS) 
    initial_hyper_vals[0]= 3 #layer_opts[np.random.randint(0,6)] #layer_opts
    initial_hyper_vals[1]= 128 # node_opts[np.random.randint(0,5)]
    initial_hyper_vals[2]= 0.001 #learnrate_opts[np.random.randint(0,100)]
    initial_hyper_vals[3]= 0.9 #beta1_opts[np.random.randint(0,100)]
    initial_hyper_vals[4]= 0.999 #beta2_opts[np.random.randint(0,100)]
    initial_hyper_vals[5]= 10**-7 #eps_opts[np.random.randint(0,100)]
    initial_hyper_vals[6]= 0 #decay_opts[np.random.randint(0,100)]
    
    loss, _=runmodel(dataset,trainx,trainy,valx,valy,testx,testy,iterations,pat,initial_hyper_vals)
    
    #Initial Values
    X = initial_hyper_vals
    y = loss 
    
    #reshape 
    X = X.reshape(1, len(X))
    y = y.reshape(1, 1)
    
    #initial fit of the model 
    model = GaussianProcessRegressor()
    model.fit(X, y)
    
    #Tracking Progress variables
    best=np.zeros(7) #best - best hyperparameters
    current=np.zeros(7) #all - validation loss at each iteration
    lowestval=9999 #lowestval - lowest validation loss 
    lowesttest=9999 #lowesttest - lowest test loss
    
    #to look at all
    all=np.zeros(100)

    #for f in range(iters):
    for f in range(100):
        
        #Select next hyperparameter values 
        current = opt_acquisition(X, y, model)
        
        #calculate loss
        curr = np.reshape(current, NUM_HYPERPARAMS, 1)
        loss, test =runmodel(dataset,trainx,trainy,valx,valy,testx,testy,iterations,pat,curr)
        
        """ Update surrogate """
        # add the data to the dataset
        X = np.vstack((X, current)) #hyperparameter values 
        y = np.vstack((y, [[loss]])) #loss values
        
        # update the model
        model.fit(X, y)
        
        all[f]=loss
        #if this is the best so far save parameters
        if(loss<lowestval):
            lowestval=loss 
            lowesttest=test
            best=np.copy(current)
        
    return lowestval,lowesttest,best,all






