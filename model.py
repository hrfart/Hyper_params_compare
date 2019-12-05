import joblib
from pbt import *
import numpy as np
from mlxtend.data import loadlocal_mnist
import os, datetime,sys, errno
import matplotlib.pyplot as plt
import sklearn.metrics
from shutil import copyfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from hyperband import *
from bayesian_opt_utils import *
from global_utils_cfg import *
import time
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Create output directory if it doesn't exist
try:
    os.mkdir(create_file_path([output_dir]))
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

if len(sys.argv)<2:
    print('Please enter an integer to select the hyperparameter opimizer.')
    print('0-random grid search')
    print('1-Bayesian')
    print('2-HYPERBAND')
    print('3-population based')
    quit()
else:
    try:
        optmethod=int(sys.argv[1])
    except:
        print('Please enter an integer to select the hyperparameter opimizer.')
        print('0-random grid search')
        print('1-Bayesian')
        print('2-HYPERBAND')
        print('3-population based')
        quit()
    if optmethod<0 or optmethod>3:
        print('Please enter an integer to select the hyperparameter opimizer.')
        print('0-random grid search')
        print('1-Bayesian')
        print('2-HYPERBAND')
        print('3-population based')
        quit()


##########################################    MAIN SECTION: RUNS ANALYSES   ##########################################################

def main():
    for dataset in configured_datasets:#,1,0,3]:
        #load data
        trainx,trainy,valx,valy,testx,testy=loaddata(dataset)

        #start timers
        start = time.time()
        pstart = time.process_time()
        #do random grid search
        if optmethod==0:
            lowestval,lowesttest,best,all=randomgridsearch(dataset,trainx,trainy,valx,valy,testx,testy)

        if optmethod==1:
            lowestval,lowesttest,best,all=bayesian_optimization(dataset,trainx,trainy,valx,valy,testx,testy)

        if optmethod==2:
            lowestval,lowesttest,best,all=hyperband(iterations,HBAND_H,dataset,trainx,trainy,valx,valy,testx,testy)

        if optmethod==3:
            lowestval,lowesttest,best,all=pbt(dataset,trainx,trainy,valx,valy,testx,testy)




        #write outputs
        out=sets[dataset]+","+methods[optmethod]+','+str(lowesttest)+','+str(lowestval)
        for f in range(7):
            out=out+','+str(best[f])
        out=out+','+str(time.time()-start)+','+str(time.process_time()-pstart)
        t=open(output_file,"a+")
        t.write(out+',\n')
        t.close()

        #plot all iterations validation
        plt.plot(np.arange(len(all)),all)
        plt.xlabel('algorithm iteration')
        plt.ylabel('loss')
        plt.title(sets[dataset]+' '+methods[optmethod])
        plt.savefig(create_file_path([output_dir,sets[dataset]+'.'+methods[optmethod]+'.'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+".png"]))
        plt.clf()

        #probably also a good idea to save the data so we can make more involved plots later.
        joblib.dump(all,create_file_path([output_dir,sets[dataset]+'.'+methods[optmethod]+'.'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+".pkl"]))


##########################################    BASE MODEL SECTION   ##########################################################

#network class
class Net(nn.Module):
    def __init__(self,datashape,dataset,numlayers,numnodes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(datashape,numnodes)
        if numlayers>1:
            self.fc2=nn.Linear(numnodes,numnodes)
        if numlayers>2:
            self.fc3=nn.Linear(numnodes,numnodes)
        if numlayers>3:
            self.fc4=nn.Linear(numnodes,numnodes)
        if numlayers>4:
            self.fc5=nn.Linear(numnodes,numnodes)
        if numlayers>5:
            self.fc6=nn.Linear(numnodes,numnodes)
        if dataset==1:
             self.outfc=nn.Linear(numnodes,10)
        else:
            self.outfc=nn.Linear(numnodes,1)

    def forward(self, x,dataset,numlayers):
        x = self.fc1(x)
        x = F.relu(x)
        if numlayers>1:
             x = self.fc2(x)
             x = F.relu(x)
        if numlayers>2:
             x = self.fc3(x)
             x = F.relu(x)
        if numlayers>3:
             x = self.fc4(x)
             x = F.relu(x)
        if numlayers>4:
             x = self.fc5(x)
             x = F.relu(x)
        if numlayers>5:
             x = self.fc6(x)
             x = F.relu(x)
        out = self.outfc(x)
        if dataset==0:
            out=F.sigmoid(out)
        if dataset==1:
            out=F.log_softmax(out,dim=1)
        return out



#data set class
class torch_dataset(torch.utils.data.Dataset):
    def __init__(self,datax,datay):
        self.x=datax
        self.y=datay

    def __len__(self):
        return len(self.y)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.x[idx,:],self.y[idx]


def longCEL(x,y):
    return torch.nn.CrossEntropyLoss(x,y)



#builds a model with data trainx,trainy,valx,valy,testx,testy,      limits iters and max iters,
#and hyper parameters in opts. Also pass in which optimization method is calling it.
#returns loss for validation set and test set
#dataset input is so the correct loss is used.
#the last four optional parameters are for the PBT method only.
def runmodel(dataset,trainx,trainy,valx,valy,testx,testy,    maxiters,pat,   opts, method=optmethod, run=None, path=None, loadrun=None, load=None, explore = 0, evaluate=False):

    #set hyperparameters.
    layers=int(opts[0])
    nodes=int(opts[1])
    learnrate=opts[2]
    beta1=opts[3]
    beta2=opts[4]
    eps=opts[5]
    decay=opts[6]


    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    #set up model
    model = Net(trainx.shape[1],dataset,layers,nodes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learnrate, betas=(beta1, beta2), eps=eps, weight_decay=decay, amsgrad=False)
    #load in previous model for PBT
    if load!=None:
        model=torch.load(str(loadrun)+'.'+str(load)+'.pt')
        optimizer = optim.Adam(model.parameters(), lr=learnrate, betas=(beta1, beta2), eps=eps, weight_decay=decay, amsgrad=False)
        optimizer.load_state_dict(torch.load(str(loadrun)+'.'+str(load)+'.opt.pt'))

    # explore step of PBT
    newparams = []
    if explore:
        plusminus = [-1, 1]
        for param_group in optimizer.param_groups:
            # small perturbation to add to each optimization hyperparameter
            lrchange = param_group['lr']*np.random.rand()*plusminus[np.random.randint(0, 2)]/10.0
            b0change = param_group['betas'][0]*np.random.rand()*plusminus[np.random.randint(0, 2)]/10.0
            b1change = param_group['betas'][1]*np.random.rand()*plusminus[np.random.randint(0, 2)]/10.0
            epschange = param_group['eps']*np.random.rand()*plusminus[np.random.randint(0, 2)]/10.0
            wdchange = param_group['weight_decay']*np.random.rand()*plusminus[np.random.randint(0, 2)]/10.0


            # update the hyperparameters
            lr = min(max(param_group['lr'] + lrchange, 1e-7), 1e-2)
            bparam1 = min(max(0.85, param_group['betas'][0] + b0change), 0.95)
            bparam2 = min(max(0.9, param_group['betas'][1] + b1change), 0.99999)
            b0 = min(bparam1, bparam2)
            b1 = max(bparam1, bparam2)
            eps = min(max(param_group['eps'] + epschange, 1e-9), 1e-7)
            wd = min(max(param_group['weight_decay'] + wdchange, 0), 0.1)

            param_group['lr'] = lr
            param_group['betas'] = (b0, b1)
            param_group['eps'] = eps
            param_group['weight_decay'] = wd
            newparams = [lr, b0, b1, eps, wd]

    #select loss function
    loss_func=torch.nn.MSELoss()
    if dataset==0:
        loss_func=torch.nn.BCELoss()
    if dataset==1:
        loss_func= F.nll_loss



    #set up weightspath
    if path==None:
        weightspath=create_file_path([output_dir,methods[method]+'.'+sets[dataset]+'.pt'])
    else:
        weightspath=create_file_path([str(run)+'.'+str(path)+'.pt'], False)




    #set up data loaders
    train_loader=torch.utils.data.DataLoader(torch_dataset(torch.tensor(trainx),torch.tensor(trainy)),batch_size=512, shuffle=True, **kwargs)
    val_loader=torch.utils.data.DataLoader(torch_dataset(torch.tensor(valx),torch.tensor(valy)),batch_size=512, shuffle=False, **kwargs)
    test_loader=torch.utils.data.DataLoader(torch_dataset(torch.tensor(testx),torch.tensor(testy)),batch_size=512, shuffle=False, **kwargs)


    #initialize best validation loss and time since improvement
    bestvalloss=np.inf
    timesinceimprove=0
    #train model
    model = model.float()
    for it in range(maxiters):
        print('iteration: '+str(it))
        model.train()
        for num, (data,target) in enumerate(train_loader):
            data, target = data.to(device).float(), target.to(device).float()
            if dataset==1:
                target=target.squeeze().long()
            optimizer.zero_grad()
            output = model.forward(data,dataset,layers)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()

        #check validation data
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data,target in val_loader:
                data, target = data.to(device).float(), target.to(device).float()
                if dataset==1:
                    target=target.squeeze().long()
                output = model.forward(data,dataset,layers)
                val_loss += loss_func(output, target).item()

        if dataset<2:
            val_loss /= len(val_loader) #take mean value if metric isn't mse.
        print(val_loss)



            


        #keep track of best loss
        if val_loss<bestvalloss:
            bestvalloss=val_loss
            timesinceimprove=0
            #save the model if there was improvement
            torch.save(model,weightspath)
            if path!=None:
                #also save optimizer state for PBT.
                torch.save(optimizer.state_dict(),create_file_path([str(run)+'.'+str(path)+'.opt.pt'], False))
        else: #increment time since last improvement.            
            timesinceimprove+=1
        #quit if there hasn't been improvement for sufficiently many iterations
        if timesinceimprove>pat:
            break
    
    
    #if this is a PBT non-evaluation run, only return the validation loss and the paramater
    if path!=None and not evaluate:
        return bestvalloss,newparams
    else:
        #evaluate test data and return full predicted values.
        testout=[]
        test_loss=0
        #load best model
        model=torch.load(weightspath)
        with torch.no_grad():
            for data,target in test_loader:
                data, target = data.to(device).float(), target.to(device).float()
                if dataset==1:
                    target=target.squeeze().long()
                output = model.forward(data,dataset,layers)
                test_loss += loss_func(output, target).item()  # sum loss
                for t in output:
                    if use_cuda:
                        t=t.cpu()
                    testout.append(t.numpy())
        if dataset<2:
            test_loss /= len(test_loader)
        return bestvalloss,test_loss,np.asarray(testout)



##########################################    DATA LOADER Section   ##########################################################


# Accepts dataset as input: 0 is mimic, 2 is MNIST, 3 housing, 4 is brains
# outputs data in order trainx,trainy,valx,valy,testx,testy.
def loaddata(dataset):
    #load Mimic.
    #Mimic is already split 70,15,15
    if dataset==0:
        a=joblib.load(data_dir+'/'+'mimic_train.pkl')
        trainx=a[0]
        trainy=a[1]
        a=joblib.load(data_dir+'/'+'mimic_validation.pkl')
        valx=a[0]
        valy=a[1]
        a=joblib.load(data_dir+'/'+'mimic_test.pkl')
        testx=a[0]
        testy=a[1]
        trainx=np.reshape(trainx,[trainx.shape[0],-1])
        valx=np.reshape(valx,[valx.shape[0],-1])
        testx=np.reshape(testx,[testx.shape[0],-1])

        del a






    #load MNIST.
    if dataset==1:
        x, y = loadlocal_mnist(images_path=data_dir+'/train-images-idx3-ubyte', labels_path=data_dir+'/train-labels-idx1-ubyte')
        testx, testy = loadlocal_mnist(images_path=data_dir+'/t10k-images-idx3-ubyte', labels_path=data_dir+'/t10k-labels-idx1-ubyte')

        #MNIST is split ~85/15, get remaining 15 (20 of train) of val from train.
        n=y.shape[0]


        testx=testx.astype(float)
        testy=testy.astype(float)


        trainx=x[0:int(np.floor(n*.8)),:].astype(float)
        trainy=y[0:int(np.floor(n*.8))].astype(float)
        valx=x[int(np.floor(n*.8)):-1,:].astype(float)
        valy=y[int(np.floor(n*.8)):-1].astype(float)

        del x,y



    #NOTE: Will do 70,15,15 split for housing and brains



    #load housing
    if dataset==2:
        data=np.genfromtxt(data_dir+'/boston-corrected',delimiter=',')

        #split into x and y
        y=data[1:,0]
        x=data[1:,1:]

        #split into train and test and val
        n=y.shape[0]
        trainx=x[0:int(np.floor(n*.7)),:]
        trainy=y[0:int(np.floor(n*.7))]
        valx=x[int(np.floor(n*.7)):int(np.floor(n*.85)),:]
        valy=y[int(np.floor(n*.7)):int(np.floor(n*.85))]
        testx=x[int(np.floor(n*.85)):-1,:]
        testy=y[int(np.floor(n*.85)):-1]

        del data,x,y



    #load brains
    if dataset==3:
        x=joblib.load(data_dir+'/nki_warped_pooled.pkl')
        y=np.genfromtxt(data_dir+'/nki_age.csv',delimiter=',')
        x=x[:,4:-4,4:-4,6:-6]
        #split into train and test and val
        n=y.shape[0]
        x=np.reshape(x,[n,-1])
        trainx=x[0:int(np.floor(n*.7)),:]
        trainy=y[0:int(np.floor(n*.7))]
        valx=x[int(np.floor(n*.7)):int(np.floor(n*.85)),:]
        valy=y[int(np.floor(n*.7)):int(np.floor(n*.85))]
        testx=x[int(np.floor(n*.85)):-1,:]
        testy=y[int(np.floor(n*.85)):-1]

        del x,y

    return trainx,np.expand_dims(trainy,axis=1),valx,np.expand_dims(valy,axis=1),testx,np.expand_dims(testy,axis=1)




##########################################      Output plots #####################################################################
#convert one-hot vector to a number
def dehotify(a):

    out=np.zeros(a.shape[0])
    for f in range(a.shape[0]):
        out[f]=np.argmax(a[f,:])
    return out

def evaluation_plot(dataset,a,b):
    if dataset==1:
         b=dehotify(b)

    if dataset==0:
        a=a[:,0]
        b=b[:]
        plt.plot(a-.25+np.linspace(0,.5,len(a)),b,'o')
        plt.xlabel('Actual Mortality')
        plt.ylabel('Predicted Mortality')
        plt.title('Accuracy= '+str(sklearn.metrics.accuracy_score(a,np.round(b)))+', AUROC= '+str(sklearn.metrics.roc_auc_score(a,np.round(b))))
        print(str(sklearn.metrics.roc_auc_score(a,np.round(b))))
    else:

        plt.plot([np.min(a),np.max(a)],[np.min(a),np.max(a)],'m')
    if dataset==1:
        a=a[:,0]
        plt.plot(a-.25+np.linspace(0,.5,len(a)),b-.25+np.linspace(0,.5,len(a)),'*')
        plt.xlabel('Actual Number (Jittered)')
        plt.ylabel('Predicted Number (Jittered)')
        plt.title('Accuracy= '+str(sklearn.metrics.accuracy_score(a,np.round(b))))
    if dataset==2:
        plt.plot(a,b,'o')
        plt.xlabel('Actual House Price (Thousands of Dollars)')
        plt.ylabel('Predicted House Price (Thousands of Dollars)')
    if dataset==3:
        plt.plot(a,b,'o')
        plt.xlabel('Actual Age (Months)')
        plt.ylabel('Predicted Age (Months)')
    plt.savefig(create_file_path([output_dir,'randomgrid-'+sets[dataset]+'.png']))
    plt.clf()



###########################  RANDOM GRID SEARCH SECTION   ######################################################
#inputs are data set to use, and the loaded data
#outputs final val and test loss,best parameters chosen
def randomgridsearch(dataset,trainx,trainy,valx,valy,testx,testy):

    #initialize best and current params
    best=np.zeros(7)
    current=np.zeros(7)
    lowestval=np.inf
    lowesttest=np.inf

    #to look at all
    all=np.zeros(grid_iters)

    for f in range(grid_iters):
        #do randomization
        current[0]=layer_opts[np.random.randint(0,6)]
        current[1]=node_opts[np.random.randint(0,5)]
        current[2]=learnrate_opts[np.random.randint(0,100)]
        current[3]=beta1_opts[np.random.randint(0,100)]
        current[4]=beta2_opts[np.random.randint(0,100)]
        current[5]=eps_opts[np.random.randint(0,100)]
        current[6]=decay_opts[np.random.randint(0,100)]

        #calculate loss
        loss, test,testeval=runmodel(dataset,trainx,trainy,valx,valy,testx,testy,iterations,pat,current,0)
        all[f]=loss



        #if this is the best so far save parameters
        if(loss<lowestval):
            lowestval=loss
            lowesttest=test
            best=np.copy(current)
            #plot best test results
            evaluation_plot(dataset,testy,testeval)


    return lowestval,lowesttest,best,all






if __name__== "__main__":
  main()
