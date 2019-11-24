# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 18:56:51 2019

@author: hmnor
"""

import numpy as np 

#ranges to search through
layer_opts=[1,2,3,4,5,6]
node_opts=[32,64,128,256,532]
learnrate_opts=np.linspace(1e-5,1e-2,100)
beta1_opts=np.linspace(.85,.95,100)
beta2_opts=np.linspace(.9,.99999,100)
eps_opts=np.linspace(1e-9,1e-7,100)
decay_opts=np.linspace(0,.1,100)


#iterations for random grid search
iters=100

#for each run of the model
iterations=10 #100
pat=10 #patience



data_dir='data_sets'
output_file='results.csv'

#0-mimic,1-MNIST,2-housing,3-brains
dataset=2

#0-random grid search
#1-bayesian optimization
optmethod=1

#datasets
sets=['mimic','MNIST','housing','NKI']
methods=['random grid','Bayes','HYPERBAND','ORGD']


NUM_HYPERPARAMS = 7