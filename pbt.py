from model import *
import os, shutil

def pbt(max_gd_iters_per_cfg,num_models,dataset,trainx,trainy,valx,valy,testx,testy):
    #initialize best and current params
    num_iters = max_gd_iters_per_cfg
    methodnum = 3
    patcheck = 30
    
    data = np.zeros((num_models, num_iters))
    timesinceimproves = np.zeros(num_models)
    timesinceimproves = timesinceimproves.astype(int)

    toexplores = np.zeros(num_models)
    bestlosses = np.zeros(num_models) + np.inf

    hyps = []
    perfs = [ [] for i in range(num_models)]
    for i in range(int(np.ceil(num_iters/30))):
        for j in range(num_models):
            if min(timesinceimproves) > 10:
                break
            print('iter', i, '| model number ', j)
            run=j #which run out of 10 it is.
            run_to_load=j
            newparams = []
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
                loss,newparams=runmodel(dataset,trainx,trainy,valx,valy,testx,testy,1,pat,current,methodnum,run,timesinceimproves[run])
                data[j, i] = loss
                timesinceimproves[run] += 1
            else:
                print(hyps[run])
                loss,newparams=runmodel(dataset,trainx,trainy,valx,valy,testx,testy,1,pat,hyps[run],methodnum,run,timesinceimproves[run],run_to_load,timesinceimproves[run]-1, toexplores[run])
                data[j, i] = loss
                if len(newparams) > 0:
                    hyps[run] = [hyps[run][0], hyps[run][1]]
                    hyps[run].extend(newparams)
                toexplores[run] = 0
            perfs[run].append(loss)
            # print(perfs[run])
            if loss < bestlosses[run]:
                bestlosses[run] = loss
                if i > 0:
                    copyfile(str(run)+'.'+str(timesinceimproves[run])+'.opt.pt',str(run)+'.'+'0.opt.pt')
                    copyfile(str(run)+'.'+str(timesinceimproves[run])+'.pt',str(run)+'.'+'0.pt')
                    timesinceimproves[run] = 1
            else:
                timesinceimproves[run]+=1
 
            # sufficient improvement
            if i > 0 and i % patcheck == 0:
                if run != np.argmin(bestlosses):
                    patcheck = patcheck+30
                    # exploit
                    bestrun = np.argmin(bestlosses)
                    hyps[run] = hyps[bestrun]
                    copyfile(str(bestrun)+'.'+'0.opt.pt',str(run)+'.'+'0.opt.pt')
                    copyfile(str(bestrun)+'.'+'0.pt',str(run)+'.'+'0.pt')
                    timesinceimproves[run]=1
                    # explore
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


