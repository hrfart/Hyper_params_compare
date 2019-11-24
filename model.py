import joblib
import numpy as np
from mlxtend.data import loadlocal_mnist
import os, datetime
import matplotlib.pyplot as plt
import sklearn.metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

use_cuda=True

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
# from tensorflow.python.compiler.xla import jit
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


data_dir = 'data_sets'
output_file = 'results.csv'


save_loss_curves=False

#0-mimic,1-MNIST,2-housing,3-brains
dataset=2

#0-random grid search
optmethod=0

#ranges to search through
layer_opts=[1,2,3,4,5,6]
node_opts=[32,64,128,256,532]
learnrate_opts=np.linspace(1e-7,1e-2,100)
beta1_opts=np.linspace(.85,.95,100)
beta2_opts=np.linspace(.9,.99999,100)
eps_opts=np.linspace(1e-9,1e-7,100)
decay_opts=np.linspace(0,.05,100)

#iterations FOR RANDOM GRIDSEARCH
grid_iters=200

#for each run of the model
iterations=300
pat=15

#datasets
sets=['mimic','MNIST','housing','NKI']
methods=['random grid','Bayes','HYPERBAND','PBT']

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
iters = 200

# for each run of the model
iterations = 300
pat = 15

# datasets
sets = ['mimic', 'MNIST', 'housing', 'NKI']
methods = ['random grid', 'Bayes', 'HYPERBAND', 'PBT']


##########################################    MAIN SECTION: RUNS ANALYSES   ##########################################################

def main():
    for dataset in [2,1,0,3]:
        #load data
        trainx,trainy,valx,valy,testx,testy=loaddata(dataset)
    
    
        #do random grid search
        if optmethod==0:    
            lowestval,lowesttest,best,all=randomgridsearch(dataset,trainx,trainy,valx,valy,testx,testy)
        
    
    
    
        #write outputs
        out=sets[dataset]+","+methods[optmethod]+','+str(lowesttest)+','+str(lowestval)
        for f in range(7):
            out=out+','+str(best[f])
        t=open(output_file,"a+")
        t.write(out+',\n')
        t.close()
    
        #plot all iterations validation
        plt.plot(np.arange(len(all)),all)
        plt.xlabel('algorithm iteration')
        plt.ylabel('loss')
        plt.title(sets[dataset]+' '+methods[optmethod])
        plt.savefig(sets[dataset]+'.'+methods[optmethod]+'.'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+".png")
        plt.clf()
    
        #probably also a good idea to save the data so we can make more involved plots later.
        joblib.dump(all,sets[dataset]+'.'+methods[optmethod]+'.'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+".pkl")


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
#the three optional parameters are for the PBT method only.
def runmodel(dataset,trainx,trainy,valx,valy,testx,testy,    maxiters,pat,   opts, method, path=None, load=None, evaluate=False):
    
    
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
    
    if load!=None:
        model=torch.load(str(load)+'.pt')
        #MODIFY OPTIMZER HERE
    else:
        #set up optimizer
        optimizer = optim.Adam(model.parameters(), lr=learnrate, betas=(beta1, beta2), eps=eps, weight_decay=decay, amsgrad=False)
    
    #loss function
    loss_func=torch.nn.MSELoss()
    if dataset==0:
        loss_func=torch.nn.BCELoss()
    if dataset==1:
        loss_func= F.nll_loss
    
    
    
    #set up weightspath
    if path==None:
        weightspath=methods[method]+'.'+sets[dataset]+'.pt'
    else:
        path=str(path)+'.pt'
    

    
    
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

        
        print(val_loss)
        
        if val_loss<bestvalloss:
            bestvalloss=val_loss
            timesinceimprove=0
            torch.save(model,weightspath)
        else:
            timesinceimprove+=1
        if timesinceimprove>pat:
            break

    if path!=None and not evaluate:
        return bestvalloss
    else:
        #evaluate test data
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
                test_loss += loss_func(output, target).item()  # sum up batch loss
                for t in output:
                    testout.append(t.numpy())
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
    plt.savefig('randomgrid-'+sets[dataset]+'.png')
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
