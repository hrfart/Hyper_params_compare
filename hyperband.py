import numpy as np


####################################  HYPERBAND SECTION   ######################################################
# INPUT:
#   R = Maximum amount of resource that can be allocated to a single configuration
#   h = Larger h means more aggressive elimination schedule.  h = 3 or 4 is recommended.
#   dataset,trainx,trainy,valx,valy,testx,testy
#
# OUTPUT:
#   lowestval,lowesttest,best,all
#
#   lowestval = loss over validation set of final configuration (algorithm attempts to minimize this)
#   finaltest = loss over test set of final configuration (generalization error, which does not control algorithm)
#   best = best hyperparameter configuration chosen (corresponds to lowestval and finaltest)
#   all = ....
from global_utils import *

def hyperband(R, h, dataset, trainx, trainy, valx, valy, testx, testy):
    # Initialize smax and budget
    smax = np.floor(np.log(R) / np.log(h))
    B = (smax + 1) * R

    # Create lists of hyperparameter configs and another list of associated validation losses for the best configs found
    # in each bracket
    best_configs = []
    lowest_val_losses = []
    test_losses = []

    # For each bracket, run successive halving
    for s in np.arange(smax, -1, -1):
        # n is the number of configurations to run successive halving over
        n = np.ceil((B * h**s) / (R * (s + 1)))
        # r is the minimum resource allocated to all configurations
        r = R * h**(-s)

        # Begin SuccessiveHalving(n,r) inner loop
        T = get_hyperparameter_configuration(n)  # Each column of T is a set of hyperparameters
        val_loss = np.zeros((1, n))  # Initialize associated row vector for validation loss
        test_loss = np.zeros((1, n))  # Initialize associated row vector for test loss
        for i in np.arange(0, s+1):
            ni = np.floor(n * h**(-i))
            ri = r * h**i

            # For each hyperparameter configuration in T, record a loss
            # Note: np.size(T,1) is the number of columns in T, i.e. the number of hparam configs left to consider
            for t in range(0, np.size(T,1)):
                val_loss[t], test_loss[t] = runmodel(dataset, trainx, trainy, valx, valy, testx, ri, pat, T[t])

            # Get the top k configurations and use that as the new set of hyperparameters to consider
            T, val_loss, test_loss = top_k_configurations(T, val_loss, test_loss, np.floor(ni/h))

        # Save the best configuration and associated loss
        best_configs.append(T)
        lowest_val_losses.append(val_loss)
        test_losses.append(test_loss)

    # Now, return the best overall config and associated losses ("best" is based on lowest validation loss)
    best_idx = np.argmin(lowest_val_losses)
    return lowest_val_losses[best_idx], test_losses[best_idx], best_configs[best_idx], lowest_val_losses


# Function to return top k performing hyperparameter configurations in a set, based on loss L
# Here, L should be a row vector of losses corresponding to each column of T (each column is a hparam cfg)
def top_k_configurations(T, ValLoss, TestLoss, k):

    topk_idx = np.argpartition(ValLoss, k)[:k]
    T_topk = T[:, topk_idx]
    val_loss_topk = ValLoss[0, topk_idx]
    test_loss_topk = TestLoss[0, TestLoss]
    return T_topk, val_loss_topk, test_loss_topk
