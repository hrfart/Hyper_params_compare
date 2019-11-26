# File for all global configuration variables
# e.g. algorithm and hyperparameter search space

use_cuda=False

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"

data_dir = 'data_sets'
output_file = 'results.csv'

save_loss_curves=False

#0-mimic,1-MNIST,2-housing,3-brains
dataset=0

# ranges to search through
NUM_HYPERPARAMS = 7 # Number of different ranges to investigate
layer_opts = [1, 2, 3, 4, 5, 6]
node_opts = [32, 64, 128, 256, 532]
learnrate_opts = np.linspace(1e-5, 1e-2, 100)
beta1_opts = np.linspace(.85, .95, 100)
beta2_opts = np.linspace(.9, .99999, 100)
eps_opts = np.linspace(1e-9, 1e-7, 100)
decay_opts = np.linspace(0, .1, 100)

#iterations FOR RANDOM GRIDSEARCH
grid_iters=200

#for each run of the model
iterations=300
pat=15

#datasets
sets=['mimic','MNIST','housing','NKI']
methods=['random grid','Bayes','HYPERBAND','PBT']


# Function to return a matrix T of hyperparameter configurations
# n = number of hyperparameter configurations to return, i.e. cardinality of T, where each element of T
#   is a different column/hyperparameter configuration
#
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