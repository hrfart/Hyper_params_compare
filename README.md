# Hyper_params_compare

This repository contains a comparison of Hyper-paramater optimization methods.
Each method is implemented from scratch.

The Hyperparameter optimization methods to be compared are:

1) Random Grid search
2) Bayesian Optimization [1] 
3) Hyperband [2]
4) Population Based Training [3]

Four Data sets are used for evaluation
(located in /data_sets; although MIMIC is not available because it was too large to be hosted here but is available online):

1) The Boston Housing data set: Predicting house price (https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html)
2) MNIST: Hand-digit recognition [4]
3) MIMIC: Predicting in-hospital Mortality from EHR data [5]
4) NKI Brain Data: Predicting age from T1 MRI brain images [6]


All code is located in the file model.py. 
It is set up to loop through every Hyperparameter optimization model and dataset
and outputs plots showing the data fit for the best hyperparameters found, and loss at each algorithm iteration.
Also outputted are loss curves for each iteration and algorithm, and the optimal loss and hyperparameters found in a csv file.

The model being optimized is a fully connected Neural network implemented in Pytorch with an Adam optimizer.
Number of layers and nodes are treated as hyperparameters, along with learning rate, weight decay, and adam parameters beta1, beta2, and epsilon.

Dependencies include:

numpy (https://numpy.org)

sklearn (https://scikit-learn.org)

matplotlib (https://matplotlib.org)

pytorch (https://pytorch.org)

mlxtend.data (http://rasbt.github.io/mlxtend/)

joblib (https://joblib.readthedocs.io/en/latest/)

Note: set up to run on a Cuda GPU. If one is unavailable, set use_cuda to false in line 13.

References:

[1] Snoek, J., Larochelle, H, & Adams, R.P. (2012) Practical Bayesian optimization of machine learning
algorithms. NIPS’12 Proceedings of the 25th International Conference on Neural Information Processing
Systems 2:2951-2959.

[2] Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A, & Talwalkar, A. (2018) Hyperband: A Novel BanditBased 
Approach to Hyperparameter Optimization. Journal of Machine Learning Research 18:1-52.
4

[3] Jaderberg, M., Dalibard, V., Osindero, S., Mojcich, M.C., Donahue, J., Razavi, A., Vinyals, O., Green,  
T., Dunning, I, Karen, S., Fernadno, C. & Kavukcuoglu, K. (2017). Population Based Training of Neural Networks. 
Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining 1791-1799


[4] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998) Gradient-based learning applied to document
recognition. Proceedings of the IEEE 86:2278–2324.

[5] Pollard T.J., Shen L., Lehman L., Feng M., Ghassemi M., Moody B., Szolovits P., Celi L.A., & Mark, R.G. 
(2016) MIMIC-III, a freely accessible critical care database. Scientific Data 10

[6] The NKI-Rockland Sample: A model for accelerating the pace of discovery
science in psychiatry. Frontiers in neuroscience 6 152