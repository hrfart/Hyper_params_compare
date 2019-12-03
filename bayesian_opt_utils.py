# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 21:13:44 2019
@author: hmnor
https://machinelearningmastery.com/what-is-bayesian-optimization/
"""

import model as mod
from global_utils_cfg import *

import numpy as np
from numpy import argmin
from scipy.stats import norm
from warnings import catch_warnings
from warnings import simplefilter
from sklearn.gaussian_process import GaussianProcessRegressor



""" Surrogate or approximation for the objective function
ESTIMATES COST OF ONE OR MORE SAMPLES 
# """
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
        
        return pred 
 
    
# probability of improvement acquisition function
def acquisition(X, Xsamples, model):
    
	# calculate the best surrogate score found so far
    yhat, _ = surrogate(model, X)

    best = min(yhat)     #best = max(yhat)
	# calculate mean and stdev via surrogate function
    mu, std = surrogate(model, Xsamples)
    mu = mu[:, 0]
    
	# calculate the probability of improvement
    probs = norm.cdf((mu - best) / (std+1E-9))
    
    
    return probs
 
# optimize the acquisition function
def opt_acquisition(X, y, model, num_models):
    
    # random search, generate random samples
    Xsamples = get_hyperparameter_configuration(num_models)
    Xsamples = np.transpose(Xsamples) 
    
	# calculate the acquisition function for each sample
    scores = acquisition(X, Xsamples, model)
       
	# locate the index of the smallest scores 
    ix = argmin(scores)
    
    #Get sample of best score 
    newX  = Xsamples[ix, :]
    newX = newX.reshape(1, np.shape(newX)[0])
    
    return newX
 
###########################  BAYESIAN OPTIMIZATION SECTION   ######################################################
#inputs are data set to use, and the loaded data
#outputs final val and test loss,best parameters chosen
def bayesian_optimization(max_gd_iters_per_cfg,num_models,dataset,trainx,trainy,valx,valy,testx,testy):

    #initialize best and current params
    initial_hyper_vals = np.zeros(NUM_HYPERPARAMS) 
    initial_hyper_vals[0]= 3 #layer_opts[np.random.randint(0,6)] #layer_opts
    initial_hyper_vals[1]= 128 # node_opts[np.random.randint(0,5)]
    initial_hyper_vals[2]= 0.001 #learnrate_opts[np.random.randint(0,100)]
    initial_hyper_vals[3]= 0.9 #beta1_opts[np.random.randint(0,100)]
    initial_hyper_vals[4]= 0.999 #beta2_opts[np.random.randint(0,100)]
    initial_hyper_vals[5]= 10**-7 #eps_opts[np.random.randint(0,100)]
    initial_hyper_vals[6]= 0 #decay_opts[np.random.randint(0,100)]
    
    loss, b, c= mod.runmodel(dataset,trainx,trainy,valx,valy,testx,testy,max_gd_iters_per_cfg,pat,initial_hyper_vals)
    loss = np.array(loss)

    #Initial Values
    X = initial_hyper_vals
    y = loss 
    
    #reshape 
    X = X.reshape(1, len(X))
    y = y.reshape(1, 1)
    
    #Generate More Training Examples 
    for r in range(0, 1):
        
        Xt = get_hyperparameter_configuration(1)
        
        #print(np.shape(Xt[:, 0]))
        loss, b, c =mod.runmodel(dataset,trainx,trainy,valx,valy,testx,testy,max_gd_iters_per_cfg,pat,Xt[:, 0])
        
        
        Xt_r = np.reshape(Xt, (1, NUM_HYPERPARAMS)) #reshape 
        X = np.vstack((X, Xt_r)) #hyperparameter values 
        y = np.vstack((y, [[loss]])) #loss values
        
        
        
    #initial fit of the model 
    model = GaussianProcessRegressor()
    model.fit(X, y)
    
    #Tracking Progress variables
    best=np.zeros(NUM_HYPERPARAMS) #best - best hyperparameters
    current=np.zeros(NUM_HYPERPARAMS) #all - validation loss at each iteration
    #best = initial_hyper_vals 
    #current = initial_hyper_vals

    lowestval=9999 #lowestval - lowest validation loss 
    lowesttest=9999 #lowesttest - lowest test loss
    
    #to look at all
    all=np.zeros(num_models)

    for f in range(num_models):
        
        #Select next hyperparameter values 
        current = opt_acquisition(X, y, model, num_models)
        curr = np.reshape(current, NUM_HYPERPARAMS, 1)
        
        #calculate loss
        loss, test, c =mod.runmodel(dataset,trainx,trainy,valx,valy,testx,testy,max_gd_iters_per_cfg,pat,curr)
        
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
            best=np.copy(curr)
        
    return lowestval,lowesttest,best,all