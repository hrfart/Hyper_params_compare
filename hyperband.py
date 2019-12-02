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
from global_utils_cfg import *
import model as mod

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
        n = np.floor((smax + 1)/(s + 1)) * h**s
        # r is the minimum resource allocated to all configurations
        r = R * h**(-s)

        # Begin SuccessiveHalving(n,r) inner loop
        T = get_hyperparameter_configuration(int(n))  # Each column of T is a set of hyperparameters
        val_loss = np.zeros((1, int(n)))  # Initialize associated row vector for validation loss
        test_loss = np.zeros((1, int(n)))  # Initialize associated row vector for test loss
        for i in np.arange(0, s+1):
            ni = np.floor(n * h**(-i))
            ri = r * h**i

            # For each hyperparameter configuration in T, record a loss
            # Note: np.size(T,1) is the number of columns in T, i.e. the number of hparam configs left to consider
            for t in range(0, np.size(T,1)):
                val_loss[0,t], test_loss[0,t], _ = mod.runmodel(dataset, trainx, trainy, valx, valy, testx, testy, int(ri), pat, T[:,t])

            # Get the top k configurations and use that as the new set of hyperparameters to consider
            k = int(np.floor(ni/h))
            # Ensure that we converge to a nonempty config
            if k <= 1:
                T, val_loss, test_loss = top_k_configurations(T, val_loss, test_loss, 1)
                break
            else:
                T, val_loss, test_loss = top_k_configurations(T, val_loss, test_loss, k)

        # Save the best configuration and associated loss
        best_configs.append(T[:,0])
        lowest_val_losses.append(val_loss[0,0])
        test_losses.append(test_loss[0,0])

    # Now, return the best overall config and associated losses ("best" is based on lowest validation loss)
    best_idx = np.argmin(lowest_val_losses)
    return lowest_val_losses[best_idx], test_losses[best_idx], best_configs[best_idx], lowest_val_losses


# Function to return top k performing hyperparameter configurations in a set, based on loss L
# Here, L should be a row vector of losses corresponding to each column of T (each column is a hparam cfg)
def top_k_configurations(T, ValLoss, TestLoss, k):

    topk_idx = np.argpartition(ValLoss[0,:], k)[:k]
    T_topk = T[:, topk_idx]
    val_loss_topk = ValLoss[:, topk_idx]
    test_loss_topk = TestLoss[:, topk_idx]
    return T_topk, val_loss_topk, test_loss_topk


# Auxiliary function to print out the Hyperband brackets to be used for a given R and h
def hyperband_brackets(R, h):

    running_total_GD_iterations = 0
    running_total_num_cfgs = 0

    # Initialize smax and budget
    smax = np.floor(np.log(R) / np.log(h))
    B = (smax + 1) * R
    print("smax={}, B={}".format(smax, B))

    # For each bracket, run successive halving
    for s in np.arange(smax, -1, -1):
        # n is the number of configurations to run successive halving over
        n = np.floor((smax + 1)/(s+1)) * h**s
        # r is the minimum resource allocated to all configurations
        r = R * h**(-s)
        print("NEW BRACKET: s = {}, n = {}, r = {}".format(int(s), int(n), int(r)))

        running_total_num_cfgs += int(n)

        for i in np.arange(0, s+1):
            ni = np.floor(n * h**(-i))
            ri = r * h**i
            print("inner loop ni = {}, ri = {}".format(int(ni), int(ri)))
            running_total_GD_iterations += int(ni * ri)

    print("Total GD iterations = {}".format(running_total_GD_iterations))
    print("Total number of configurations explored = {}".format(running_total_num_cfgs))

    return True
