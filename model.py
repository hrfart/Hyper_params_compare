import joblib
import numpy as np
from mlxtend.data import loadlocal_mnist
import os,datetime
import tensorflow as tf
from tensorflow_addons import optimizers
import matplotlib.pyplot as plt
import sklearn.metrics


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="4"
from tensorflow.python.compiler.xla import jit
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


data_dir='data_sets'
output_file='results.csv'

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
grid_iters=100

#for each run of the model
iterations=300
pat=15


#datasets
sets=['mimic','MNIST','housing','NKI']
methods=['random grid','Bayes','HYPERBAND','ORGD']






##########################################    MAIN SECTION: RUNS ANALYSES   ##########################################################

def main():
    for dataset in [0,1,2,3]:
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



    
#builds a model with data trainx,trainy,valx,valy,testx,testy,      limits iters and max iters,
#and hyper parameters in opts. Also pass in which optimization method is calling it.
#returns loss for validation set and test set
#dataset input is so the correct loss is used.
def runmodel(dataset,trainx,trainy,valx,valy,testx,testy,    maxiters,pat,   opts, method):
    layers=int(opts[0])
    nodes=int(opts[1])
    learnrate=opts[2]
    beta1=opts[3]
    beta2=opts[4]
    eps=opts[5]
    decay=opts[6]
    
    #set up optimizer
    adam=optimizers.AdamW(decay,learning_rate=learnrate,beta_1=beta1,beta_2=beta2,epsilon=eps)
    
    #get shape of output data
    if testy.ndim>1:
        outputs=testy.shape[1]
    else:
        outputs=1
        
    #build model
    model = tf.keras.models.Sequential()
    
    #input layer
    if dataset==3 and nodes==532: #brain image data is too big to allocate the memory for this, so use second largest size
        nodes=264
    
    model.add(tf.keras.layers.Dense(nodes, activation='relu',bias_initializer='glorot_normal',kernel_initializer='glorot_normal'))
    
    
    #additional hiddenlayers
    for f in range(layers-1):
        model.add(tf.keras.layers.Dense(nodes, activation='relu',bias_initializer='glorot_normal',kernel_initializer='glorot_normal'))
    
    outputs=float(outputs)
    #path to save wights to
    weightspath=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'.hdf5'
    #outputlayer and compile.
    if dataset>1:
        model.add(tf.keras.layers.Dense(1,bias_initializer='glorot_normal',kernel_initializer='glorot_normal'))
        model.compile(optimizer=adam, loss='mean_squared_error',  metrics=['mean_squared_error','mean_absolute_error'])
        checkpoint = tf.keras.callbacks.ModelCheckpoint(weightspath, monitor='val_loss',patience=pat, verbose=1, save_best_only=True, mode='min')
        checkpoint2=tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=pat, verbose=0, mode='min', baseline=None)
    elif dataset==1:
        model.add(tf.keras.layers.Dense(outputs, activation='softmax',bias_initializer='glorot_normal',kernel_initializer='glorot_normal'))
        model.compile(optimizer=adam, loss='categorical_crossentropy',  metrics=['acc'])
        checkpoint = tf.keras.callbacks.ModelCheckpoint(weightspath, monitor='val_loss',patience=pat, verbose=1, save_best_only=True, mode='min')
        checkpoint2=tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=pat, verbose=0, mode='min', baseline=None)
    else:
        model.add(tf.keras.layers.Dense(outputs, activation='sigmoid',bias_initializer='glorot_normal',kernel_initializer='glorot_normal'))
        model.compile(optimizer=adam, loss='binary_crossentropy',  metrics=['acc'])
        checkpoint = tf.keras.callbacks.ModelCheckpoint(weightspath, monitor='val_loss',patience=pat, verbose=1, save_best_only=True, mode='min')
        checkpoint2=tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=pat, verbose=0, mode='min', baseline=None)




    callbacks_list = [checkpoint,checkpoint2]

    print('fitting model')
    history_callback = model.fit(trainx,trainy, epochs=iterations,validation_data=(valx,valy), callbacks=callbacks_list,batch_size=512)
    
    
    print('loading best model')
    #load best model
    model.load_weights(weightspath)
    if dataset>1:
        model.compile(optimizer=adam, loss='mean_squared_error')
    elif dataset==1:
        model.compile(optimizer=adam, loss='categorical_crossentropy')
    else:
        model.compile(optimizer=adam, loss='binary_crossentropy')
    
    #save loss curves
    a={}
    out=sets[dataset]+'.'+methods[method]+'_losses.pkl'
    if os.path.exists(out):
        a=joblib.load(out)
    time=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    a[time+'train']=np.array(history_callback.history["loss"])
    a[time+'val']=np.array(history_callback.history["val_loss"])
    joblib.dump(a,out)
    if save_loss_curves:
        loss_history = np.array(history_callback.history["loss"])
        valloss_history = np.array(history_callback.history["val_loss"])
        train, = plt.plot(range(len(loss_history)),loss_history,'k--', label='train')
        val, = plt.plot(range(len(loss_history)),valloss_history,'r--', label='val')
        plt.legend([train,val], ['Train Loss', 'Validation Loss'])
        #plt.axis([0,len(loss_history),0,100])
        plt.savefig('loss_curves/'+sets[dataset]+'-'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'.png')
        plt.clf()
            
    #delete weights 
    os.remove(weightspath)
    #get test and validation outputs
    print('evaluating on test and val data')
    
    return model.evaluate(valx,valy),model.evaluate(testx,testy),model.predict(testx)

    
    
        
        





##########################################    DATA LOADER Section   ##########################################################


#Accepts dataset as input: 0 is mimic, 2 is MNIST, 3 housing, 4 is brains
#outputs data in order trainx,trainy,valx,valy,testx,testy.
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
        
        newy=np.zeros((n,10))
        #onehotify
        for f in range(10):
            newy[:,f]=(y==f)
        y=newy
        del newy
        testx=testx.astype(float)
        testy=testy.astype(float)
        
        newty=np.zeros((testy.shape[0],10))
        #onehotify
        for f in range(10):
            newty[:,f]=(testy==f)
        testy=newty
        del newty
            
        trainx=x[0:int(np.floor(n*.8)),:].astype(float)
        trainy=y[0:int(np.floor(n*.8)),:].astype(float)
        valx=x[int(np.floor(n*.8)):-1,:].astype(float)
        valy=y[int(np.floor(n*.8)):-1,:].astype(float)

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
    
    return trainx,trainy,valx,valy,testx,testy




##########################################      Output plots #####################################################################
#convert one-hot vector to a number
def dehotify(a):

    out=np.zeros(a.shape[0])
    for f in range(a.shape[0]):
        out[f]=np.argmax(a[f,:])
    return out
    
def evaluation_plot(dataset,a,b):
    if dataset==1:
        a=dehotify(a)
        b=dehotify(b)
    
    if dataset==0:
        plt.plot(a-.25+np.linspace(0,.5,len(a)),b,'o')
        plt.xlabel('Actual Mortality')
        plt.ylabel('Predicted Mortality')
        plt.title('Accuracy= '+str(sklearn.metrics.accuracy_score(a,np.round(b)))+', AUROC= '+str(sklearn.metrics.roc_auc_score(a,np.round(b))))
        print(str(sklearn.metrics.roc_auc_score(a,np.round(b))))
    else:
        
        plt.plot([np.min(a),np.max(a)],[np.min(a),np.max(a)],'m')
    if dataset==1:
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
    all=np.zeros(100)
    
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