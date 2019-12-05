from model import *
import os, shutil

'''
This is an implementation of Population Based Training (https://arxiv.org/pdf/1711.09846.pdf)
It takes in the dataset, training data, validation data, and test data, and outputs the validation
and test performance of the best model, as well as data to plot loss curves.

'''
def pbt(dataset,trainx,trainy,valx,valy,testx,testy):
    #initialize best and current params
    num_iters = 300
    # number of initial configurations of hyperparameters we are starting with
    num_models = 70
    methodnum = 3
    
    # number of epochs we wait before we exploit/explore
    patcheck = 30
   
    # data to plot loss curves
    data = np.zeros((num_models, num_iters))

    # tracks number of iterations it has been since the model improved
    timesinceimproves = np.zeros(num_models)
    timesinceimproves = timesinceimproves.astype(int)

    # if a model has just been through the exploit step, then its corresponding index in
    # toexplore is set to 1, in which case we will explore 
    toexplores = np.zeros(num_models)
    # the best validation loss of each model
    bestlosses = np.zeros(num_models) + np.inf
    
    # the hyperparameters of each model
    hyps = []

    # variable used to print performance if the user desires
    perfs = [ [] for i in range(num_models)]

    # i*pat is the number of epochs that each model is trained for overall
    for i in range(10):
        for j in range(num_models):
            if min(timesinceimproves) > 10:
                break
            print('iter', i, '| model number ', j)
            run=j #which run out of 10 it is.
            run_to_load=j
            newparams = []
            # initialize models with random hyperparameters 
            if i == 0:
                current = np.zeros(7)
                current[0]=layer_opts[np.random.randint(0,6)]
                current[1]=node_opts[np.random.randint(0,5)]
                current[2]=learnrate_opts[np.random.randint(0,100)]
                current[3]=beta1_opts[np.random.randint(0,100)]
                current[4]=beta2_opts[np.random.randint(0,100)]
                current[5]=eps_opts[np.random.randint(0,100)]
                current[6]=decay_opts[np.random.randint(0,100)]
                hyps.append(current)
                # run each model for 30 epochs
                loss,newparams=runmodel(dataset,trainx,trainy,valx,valy,testx,testy,30,pat,current,methodnum,run,timesinceimproves[run])
                data[j, i] = loss
                timesinceimproves[run] += 1
            else:
                print(hyps[run])
                # continue to train each model (some may have been through the exploit phase)
                loss,newparams=runmodel(dataset,trainx,trainy,valx,valy,testx,testy,30,pat,hyps[run],methodnum,run,1,run_to_load,np.min([timesinceimproves[run]-1,1]), toexplores[run])
                data[j, i] = loss
                # update model parameters if they changed in the explore phase (which occurs in runmodel)
                if len(newparams) > 0:
                    hyps[run] = [hyps[run][0], hyps[run][1]]
                    hyps[run].extend(newparams)
                toexplores[run] = 0
            perfs[run].append(loss)
            # print(perfs[run])
            # if the loss decreases, we update the best performing model (indexed 0)
            if loss < bestlosses[run]:
                bestlosses[run] = loss
                if i > 0:
                    copyfile(str(run)+'.'+str(1)+'.opt.pt',str(run)+'.'+'0.opt.pt')
                    copyfile(str(run)+'.'+str(1)+'.pt',str(run)+'.'+'0.pt')
                    timesinceimproves[run] = 1
            else:
                # otherwise, we increment our patience variable
                timesinceimproves[run]+=1
 
            # if our model is not the best and it has improved in the last 15 iterations
            if run != np.argmin(bestlosses) and timesinceimprove[run] > 15:
                # exploit: set model "run"'s hyperparameters to be the best model's hyperparameters
                bestrun = np.argmin(bestlosses)
                hyps[run] = hyps[bestrun]
                copyfile(str(bestrun)+'.'+'0.opt.pt',str(run)+'.'+'0.opt.pt')
                copyfile(str(bestrun)+'.'+'0.pt',str(run)+'.'+'0.pt')
                timesinceimproves[run]=1
                # explore: set the variable to explore in run model
                toexplores[run] = 1

    #do evaluation
    bestrun = np.argmin(bestlosses)
    best = hyps[bestrun]
    lowestval,lowesttest,testeval=runmodel(dataset,trainx,trainy,valx,valy,testx,testy,1,pat,hyps[bestrun],methodnum,bestrun,0,bestrun,0,evaluate=True)
    evaluation_plot(dataset,testy,testeval)
    print(lowestval, lowesttest, best)

    dir_name = "."
    test = os.listdir(dir_name)

    for item in test:
        if item.endswith(".pt"):
            os.remove(os.path.join(dir_name, item))
    return lowestval,lowesttest,best,data


