import joblib
import numpy as np
from mlxtend.data import loadlocal_mnist
import os, datetime
import tensorflow as tf
from tensorflow_addons import optimizers
import matplotlib.pyplot as plt

data_dir = 'data_sets'
output_file = 'results.csv'

# 0-mimic,1-MNIST,2-housing,3-brains
dataset = 2

# 0-random grid search
optmethod = 0

# ranges to search through
NUM_HYPERPARAMS = 7 # Number of different ranges to investigate
layer_opts = [1, 2, 3, 4, 5, 6]
node_opts = [32, 64, 128, 256, 532]
learnrate_opts = np.linspace(1e-5, 1e-2, 100)
beta1_opts = np.linspace(.85, .95, 100)
beta2_opts = np.linspace(.9, .99999, 100)
eps_opts = np.linspace(1e-9, 1e-7, 100)
decay_opts = np.linspace(0, .1, 100)

# iterations for random grid search
iters = 100

# for each run of the model
iterations = 100
pat = 10

# datasets
sets = ['mimic', 'MNIST', 'housing', 'NKI']
methods = ['random grid', 'Bayes', 'HYPERBAND', 'ORGD']


##########################################    MAIN SECTION: RUNS ANALYSES   ##########################################################

def main():
    # load data
    trainx, trainy, valx, valy, testx, testy = loaddata(dataset)

    # do random grid search
    if optmethod == 0:
        lowestval, lowesttest, best, all = randomgridsearch(dataset, trainx, trainy, valx, valy, testx, testy)

    # write outputs
    out = sets[dataset] + "," + methods[optmethod] + ',' + str(lowesttest) + ',' + str(lowestval)
    for f in range(7):
        out = out + ',' + str(best[f])
    t = open(output_file, "a+")
    t.write(out + ',\n')
    t.write(out + '\n')
    t.close()

    # plot all iterations validation
    plt.plot(np.arange(iters), all, 'o')
    plt.xlabel('algorithm iteration')
    plt.ylabel('loss')
    plt.title(sets[dataset] + ' ' + methods[optmethod])
    plt.savefig(
        sets[dataset] + '.' + methods[optmethod] + '.' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".png")
    plt.clf()

    # probably also a good idea to save the data so we can make more involved plots later.
    joblib.dump(all, sets[dataset] + '.' + methods[optmethod] + '.' + datetime.datetime.now().strftime(
        '%Y-%m-%d_%H-%M-%S') + ".pkl")


##########################################    BASE MODEL SECTION   ##########################################################


# builds a model with data trainx,trainy,valx,valy,testx,testy,      limits iters and max iters,
# and hyper parameters in opts.
# returns loss for validation set and test set
# dataset input is so the correct loss is used.
def runmodel(dataset, trainx, trainy, valx, valy, testx, testy, maxiters, pat, opts):
    layers = int(opts[0])
    nodes = int(opts[1])
    learnrate = opts[2]
    beta1 = opts[3]
    beta2 = opts[4]
    eps = opts[5]
    decay = opts[6]

    # set up optimizer
    adam = optimizers.AdamW(decay, learning_rate=learnrate, beta_1=beta1, beta_2=beta2, epsilon=eps)

    # get shape of output data
    if testy.ndim > 1:
        outputs = testy.shape[1]
    else:
        outputs = 1

    # build model
    model = tf.keras.models.Sequential()
    # input layer
    model.add(tf.keras.layers.Dense(nodes, activation='relu', bias_initializer='glorot_normal',
                                    kernel_initializer='glorot_normal'))
    # additional hiddenlayers
    for f in range(layers - 1):
        model.add(tf.keras.layers.Dense(nodes, activation='relu', bias_initializer='glorot_normal',
                                        kernel_initializer='glorot_normal'))

    # path to save wights to
    weightspath = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.hdf5'
    # outputlayer and compile.
    if dataset > 1:
        model.add(tf.keras.layers.Dense(1, bias_initializer='glorot_normal', kernel_initializer='glorot_normal'))
        model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error', 'mean_absolute_error'])
        checkpoint = tf.keras.callbacks.ModelCheckpoint(weightspath, monitor='val_loss', patience=pat, verbose=1,
                                                        save_best_only=True, mode='min')
    elif dataset == 1:
        model.add(tf.keras.layers.Dense(outputs, activation='softmax', bias_initializer='glorot_normal',
                                        kernel_initializer='glorot_normal'))
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
        checkpoint = tf.keras.callbacks.ModelCheckpoint(weightspath, monitor='val_loss', patience=pat, verbose=1,
                                                        save_best_only=True, mode='min')
    else:
        model.add(tf.keras.layers.Dense(outputs, activation='sigmoid', bias_initializer='glorot_normal',
                                        kernel_initializer='glorot_normal'))
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])
        checkpoint = tf.keras.callbacks.ModelCheckpoint(weightspath, monitor='val_loss', patience=pat, verbose=1,
                                                        save_best_only=True, mode='min')

    callbacks_list = [checkpoint]

    print('fitting model')
    model.fit(trainx, trainy, epochs=iters, validation_data=(valx, valy), callbacks=callbacks_list)

    print('loading best model')
    # load best model
    model.load_weights(weightspath)
    if dataset > 1:
        model.compile(optimizer=adam, loss='mean_squared_error')
    elif dataset == 1:
        model.compile(optimizer=adam, loss='categorical_crossentropy')
    else:
        model.compile(optimizer=adam, loss='binary_crossentropy')

    # delete weights
    os.remove(weightspath)
    # get test and validation outputs
    print('evaluating on test and val data')
    return model.evaluate(valx, valy), model.evaluate(testx, testy)


##########################################    DATA LOADER Section   ##########################################################


# Accepts dataset as input: 0 is mimic, 2 is MNIST, 3 housing, 4 is brains
# outputs data in order trainx,trainy,valx,valy,testx,testy.
def loaddata(dataset):
    # load Mimic.
    # Mimic is already split 70,15,15
    if dataset == 0:
        a = joblib.load(data_dir + '/' + 'mimic_train.pkl')
        trainx = a[0]
        trainy = a[1]
        a = joblib.load(data_dir + '/' + 'mimic_validation.pkl')
        valx = a[0]
        valy = a[1]
        a = joblib.load(data_dir + '/' + 'mimic_test.pkl')
        testx = a[0]
        testy = a[1]
        del a

    # load MNIST.
    if dataset == 1:
        x, y = loadlocal_mnist(images_path=data_dir + '/train-images-idx3-ubyte',
                               labels_path=data_dir + '/train-labels-idx1-ubyte')
        testx, testy = loadlocal_mnist(images_path=data_dir + '/t10k-images-idx3-ubyte',
                                       labels_path=data_dir + '/t10k-labels-idx1-ubyte')

        # MNIST is split ~85/15, get remaining 15 (20 of train) of val from train.
        n = y.shape[0]

        newy = np.zeros((n, 10))
        # onehotify
        for f in range(10):
            newy[:, f] = (y == f)
        y = newy
        del newy

        newty = np.zeros((testy.shape[0], 10))
        # onehotify
        for f in range(10):
            newty[:, f] = (testy == f)
        testy = newty
        del newty

        trainx = x[0:int(np.floor(n * .8)), :]
        trainy = y[0:int(np.floor(n * .8)), :]
        valx = x[int(np.floor(n * .8)):-1, :]
        valy = y[int(np.floor(n * .8)):-1, :]

        del x, y

    # NOTE: Will do 70,15,15 split for housing and brains

    # load housing
    if dataset == 2:
        data = np.genfromtxt(data_dir + '/boston-corrected', delimiter=',')

        # split into x and y
        y = data[1:, 0]
        x = data[1:, 1:]

        # split into train and test and val
        n = y.shape[0]
        trainx = x[0:int(np.floor(n * .7)), :]
        trainy = y[0:int(np.floor(n * .7))]
        valx = x[int(np.floor(n * .7)):int(np.floor(n * .85)), :]
        valy = y[int(np.floor(n * .7)):int(np.floor(n * .85))]
        testx = x[int(np.floor(n * .85)):-1, :]
        testy = y[int(np.floor(n * .85)):-1]

        del data, x, y

    # load brains
    if dataset == 3:
        x = joblib.load(data_dir + '/nki_warped_pooled.pkl')
        y = np.genfromtxt(data_dir + '/nki_age.csv', delimiter=',')

        # split into train and test and val
        n = y.shape[0]
        x = np.reshape(x, [n, -1])
        trainx = x[0:int(np.floor(n * .7)), :]
        trainy = y[0:int(np.floor(n * .7))]
        valx = x[int(np.floor(n * .7)):int(np.floor(n * .85)), :]
        valy = y[int(np.floor(n * .7)):int(np.floor(n * .85))]
        testx = x[int(np.floor(n * .85)):-1, :]
        testy = y[int(np.floor(n * .85)):-1]

        del x, y

    return trainx, trainy, valx, valy, testx, testy


###########################  RANDOM GRID SEARCH SECTION   ######################################################
# inputs are data set to use, and the loaded data
# outputs final val and test loss,best parameters chosen
def randomgridsearch(dataset, trainx, trainy, valx, valy, testx, testy):
    # initialize best and current params
    best = np.zeros(7)
    current = np.zeros(7)
    lowestval = 9999
    lowesttest = 9999

    # to look at all
    all = np.zeros(100)

    for f in range(iters):
        # do randomization
        current[0] = layer_opts[np.random.randint(0, 6)]
        current[1] = node_opts[np.random.randint(0, 5)]
        current[2] = learnrate_opts[np.random.randint(0, 100)]
        current[3] = beta1_opts[np.random.randint(0, 100)]
        current[4] = beta2_opts[np.random.randint(0, 100)]
        current[5] = eps_opts[np.random.randint(0, 100)]
        current[6] = decay_opts[np.random.randint(0, 100)]

        # calculate loss
        loss, test = runmodel(dataset, trainx, trainy, valx, valy, testx, testy, iterations, pat, current)
        all[f] = loss
        # if this is the best so far save parameters
        if (loss < lowestval):
            lowestval = loss
            lowesttest = test
            best = np.copy(current)

    return lowestval, lowesttest, best, all


if __name__ == "__main__":
    main()