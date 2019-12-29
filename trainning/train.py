import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from scipy import stats
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,  Reshape, Activation, Flatten, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras.backend import clear_session
import random
import os
from termcolor import colored, cprint 
import datetime

# global var

version_counter = 1
best = 92.0
save_layers_info_times = 1
checkpoints = False
cvt_to_tflite = True
write_to_log = True
save_plot_ornot = True

now = datetime.datetime.now()
edit_info = "{}_{}_{}_{}_{}_{}".format(now.year,now.month,now.day,now.hour,now.minute,now.second)
thisis = 'HAR'

log_file = "{}_{}_log_.txt".format(thisis,edit_info)

CTL_eval_data_path      = 'data/CTL_data/eval_dataset/*'
CTL_train_data_path     = 'data/CTL_data/total.csv'
CTL_tflite_model_path   = 'CTL_tflites/*'

# HAR_eval_data_path    = '/Users/acewood/code_base/research/training/try_3/HAR_data/eval_dataset/a1.csv'
HAR_train_data_path     = 'data/HAR_data/WISDM_ar_v1.1_raw.txt'
HAR_tflite_model_path   = 'HAR_tflites/*'

def myprint(s):
    with open('log.txt','a+') as f:
        print(s, file=f)

def write2log(content):
    global log_file
    with open(log_file,"a+") as f:
        f.write(content)

print_red_on_cyan = lambda x: cprint(x, 'red', 'on_cyan') 


def log_message(mode, *messages):
    if(mode==1):
        str="*******-----{message}-----********".format(message = messages)
        print("$" * 70)
        print(str)
        print("$" * 70)
    elif(mode==2):
        length = len(max(messages,key=len))
        print("-"*length)
        print("-"*length)
        for mes in messages:
            print(mes)
        print("-"*length)
        print("-"*length)
    else:
        print(messages)

def save_plot(history,subpath):
    # summarize history for acc
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(subpath+'/acc.png')
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(subpath+'/loss.png')
    plt.clf()

def maxidx(list):
    return list.index(max(list))+1


def self_eval_single_act(model,x_tests, single_activity):
    log_message(1,"Evaluating........")
    count =0
    for i in range(len(x_tests)):
        temp = x_tests[i]
        pre = model.predict(temp)[0].tolist()
        p = maxidx(pre)
        # print("{}  :  {}".format(p,act))
        if p == single_activity:
            count = count +1
    print("self_eval_single_act ACC :   {}".format(count/len(x_tests)))


def self_eval(model,x_tests,y_tests,n_time_step):  
    log_message(1,"Evaluating........")
    count =0
    for i in range(len(x_tests)):
        temp=x_tests[i]
        temp = temp.reshape([1,n_time_step,3])
        pre = model.predict(temp)[0].tolist()
        p = maxidx(pre)
        lab = y_tests[i].tolist()
        l = maxidx(lab)
        if p == l:
            count = count +1
    print("self_eval ACC :   {}".format(count/len(x_tests)))


def process_trainning_data(df,n_time_step, n_step ,n_features,random_seed):
    print("Preparing dataset for trainning......")
    segments = []
    labels = []
    for i in range(0, len(df) - n_time_step,n_step):
        sample = df[i:i+n_time_step][['x-axis','y-axis','z-axis']]
        segments.append(np.asarray(sample))
        label = stats.mode(df['activity'][i : i + n_time_step])[0][0]
        labels.append(label)
    labels= np.asarray(pd.get_dummies(labels),dtype=np.float32)
    reshaped_segments = np.asarray(segments,dtype=np.float32)
    x_train, x_test, y_train, y_test = train_test_split(reshaped_segments,labels,test_size=0.3,random_state=random_seed)
    print("Prepared !!")
    return (x_train,y_train,x_test,y_test)

# def processEvalDataWithStep(df,n_time_step, n_step ,n_features,single_activity):
#     print("Preparing dataset for evaluating......")
#     if single_activity:
#         print('not support yet')
#         return
#     segments = []
#     labels = []
#     for i in range(0, len(df) - n_time_step,n_step):
#         xs = df['x-axis'].values[i : i + n_time_step]
#         ys = df['y-axis'].values[i : i + n_time_step]
#         zs = df['z-axis'].values[i : i + n_time_step]
#         segments.append([xs,ys,zs])
#         labels.append([1.0,0.0,0.0,0.0,0.0])

#     labels= np.asarray(labels,dtype=np.float32)
#     reshaped_segments = np.asarray(segments,dtype=np.float32).reshape(-1,n_time_step,n_features)
#     #x_train, x_test, y_train, y_test = train_test_split(reshaped_segments,labels,test_size=0.01,random_state=random_seed)

#     print("df: {}".format(df.shape))
#     print("total: {}".format(reshaped_segments.shape))
#     print("labels: {}".format(labels.shape))
#     return (reshaped_segments,labels)

def process_evalData_simpleSplit(df,n_time_step ,n_features, single_activity,n_total_activity):    
    print("In processEvalDataSimpleSplit")
    segments = []
    labels = []

    if single_activity > 0 & single_activity < 7:
        label = [0.0] * n_total_activity
        label[single_activity-1] = 1.0

        for i in range(0,len(df)-n_time_step,n_time_step):
            sample = df[i:i+n_time_step][['x-axis','y-axis','z-axis']]
            segments.append(np.asarray(sample))
            labels.append(label)
        labels= np.asarray(labels,dtype=np.float32)

    else:
        for i in range(0,len(df)-n_time_step,n_time_step):
            sample = df[i:i+n_time_step][['x-axis','y-axis','z-axis']]
            segments.append(np.asarray(sample))
            label = stats.mode(df['activity'][i : i + n_time_step])[0][0]
            labels.append(label)
        labels= np.asarray(pd.get_dummies(labels),dtype=np.float32)
    reshaped_segments = np.asarray(segments,dtype=np.float32)
    return (reshaped_segments,labels)

def cvt2tflite(h5model,version):
    cvt = tf.lite.TFLiteConverter.from_keras_model(h5model)
    tflite_model = cvt.convert()
    path = "{}_tflites/model_v{}.tflite".format(thisis,version)
    open(path,"wb").write(tflite_model)
    log_message(1," MODEL 2 TFLITE ")

def read_raw_data_har(filePath):
    columnNames = ['user_id','activity','timestamp','x-axis','y-axis','z-axis']
    data = pd.read_csv(filePath,header = None, names=columnNames,na_values=';')
    data['z-axis']=data['z-axis'].str.rstrip(';').astype(float)
    data.dropna(axis=0,how='any',inplace=True)
    return data

def read_data_with_type(path):
    dtypes = pd.read_csv(path, nrows=1).iloc[0].to_dict()
    return pd.read_csv(path,dtype=dtypes, skiprows=[1])

def read_raw_data_ctl(filePath):
    columnNames = ['activity','timestamp','x-axis','y-axis','z-axis']
    data = pd.read_csv(filePath,header = None, names=columnNames,na_values=';')
    data.dropna(axis=0,how='any',inplace=True)
    return data


def plot_data(data):
    axis = data.plot(subplots=True, figsize=(13, 8))
    for ax in axis:
        ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))

def predict_tflite_model(path,x,y):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    count = 0 
    s1 = 1
    s2 = x[0].shape[0]
    s3 = x[0].shape[1]
    for i in range(len(x)):
        print(i)
        temp = x[i]
        temp = temp.reshape([s1,s2,s3])
        interpreter.set_tensor(input_details[0]['index'], temp)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0].tolist()
        p = maxidx(output_data)
        lab = y[i].tolist()
        l = maxidx(lab)
        if p == l:
            count = count +1
    print("self_eval TFLITE ACC :   {}".format(count/len(x)))



def plot_activity(activity, df, num,head):
    if head == 1:
        data = df[df['activity'] == activity][['x-axis', 'y-axis', 'z-axis']].head(num) 
    elif head == 0:
        data = df[df['activity'] == activity]
        data = data[['x-axis', 'y-axis', 'z-axis']]
    else:
        data = df[df['activity'] == activity][['x-axis', 'y-axis', 'z-axis']].tail(num)
    axis = data.plot(subplots=True, figsize=(10, 8), title=activity)
    for ax in axis:
        ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))



def gen_inputd(df,step):
    ret = []
    for i in range(0,len(df)-150,step):
        temp = df[i:i+150].as_matrix()
        temp = temp.reshape([1,150,3])
        ret.append(temp)
    return ret



def to_csv(df, path):
    df.loc[-1] = df.dtypes
    df.index = df.index + 1
    df.sort_index(inplace=True)
    df.to_csv(path, index=False)

# def featureNormalize(dataset):
#     mu = np.mean(dataset,axis=0)
#     sigma = np.std(dataset,axis=0)
#     return (dataset-mu)/sigma

def plotAxis(axis,x,y,title):
    axis.plot(x,y)
    axis.set_title(title)
    axis.xaxis.set_visible(False)
    axis.set_ylim([min(y)-np.std(y),max(y)+np.std(y)])
    axis.set_xlim([min(x),max(x)])
    axis.grid(True)


def eval_loss_acc(acc,loss):
    first_half = acc / 1.0 * 65
    second_half = (1.0 - loss) * 35
    return (first_half + second_half)


def plotActivity(activity,data):
    fig,(ax0,ax1,ax2) = plt.subplots(nrows=3, figsize=(15,10),sharex=True)
    plotAxis(ax0,data['timestamp'],data['x-axis'],'x-axis')
    plotAxis(ax1,data['timestamp'],data['y-axis'],'y-axis')
    plotAxis(ax2,data['timestamp'],data['z-axis'],'z-axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.9)
    plt.show()

# def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
#     from tensorflow.python.framework.graph_util import convert_variables_to_constants
#     graph = session.graph
#     with graph.as_default():
#         freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
#         output_names = output_names or []
#         output_names += [v.op.name for v in tf.global_variables()]
#         # Graph -> GraphDef ProtoBuf
#         input_graph_def = graph.as_graph_def()
#         if clear_devices:
#             for node in input_graph_def.node:
#                 node.device = ""
#         frozen_graph = convert_variables_to_constants(session, input_graph_def,output_names, freeze_var_names)
#         return frozen_graph


def one_model(N_FEATURES,N_TIME_STEPS,N_CLASSES,LEARNING_RATE,
            L2_LOSS,N_HIDDEN_UNITS,N_EPOCHS,BATCH_SIZE,RANDOM_SEED,
            STEP,LOSS,OPTIMIZER,VALIDATION_SPLIT,MONITOR,
            X_TRAIN, Y_TRAIN, X_TEST, Y_TEST):
    
    #model 1.  acc:0.89
    # model = Sequential()
    # model.add(Dense(N_HIDDEN_UNITS,  activation = 'relu' , input_shape= (N_TIME_STEPS,N_FEATURES), use_bias=True, kernel_initializer='random_normal', bias_initializer='random_normal', kernel_regularizer=regularizers.l2(L2_LOSS), bias_regularizer= regularizers.l2(L2_LOSS),name="input_"))
    # model.add(Dense(N_HIDDEN_UNITS,  activation = 'relu' ,  use_bias=True, kernel_initializer='random_normal', bias_initializer='random_normal', kernel_regularizer=regularizers.l2(L2_LOSS), bias_regularizer= regularizers.l2(L2_LOSS)))
    # model.add(Dense(N_HIDDEN_UNITS,  activation = 'relu' ,  use_bias=True, kernel_initializer='random_normal', bias_initializer='random_normal', kernel_regularizer=regularizers.l2(L2_LOSS), bias_regularizer= regularizers.l2(L2_LOSS)))
    # model.add(Flatten())
    # model.add(Dense(N_CLASSES, activation = 'softmax', name = 'output_' ))

    #model 2.  could not convert to tflite
    # model = Sequential()
    # model.add(Dense(N_HIDDEN_UNITS,  activation = 'relu' , input_shape= (N_TIME_STEPS,N_FEATURES), use_bias=True, kernel_initializer='random_normal', bias_initializer='random_normal', kernel_regularizer=regularizers.l2(L2_LOSS), bias_regularizer= regularizers.l2(L2_LOSS),name="input_"))
    # model.add(SimpleRNN(N_HIDDEN_UNITS,activation='tanh',use_bias=True))
    # model.add(Dense(N_CLASSES, activation='softmax', name='output_'))


    #model 3
    model = Sequential()
    model.add(Conv1D(filters=N_HIDDEN_UNITS, kernel_size=3, activation='relu', input_shape=(N_TIME_STEPS,N_FEATURES),name="input_"))
    model.add(Conv1D(filters=N_HIDDEN_UNITS, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(N_CLASSES, activation='softmax',name="output_"))

    create_model(N_FEATURES,
                N_TIME_STEPS,
                N_CLASSES,
                LEARNING_RATE,
                L2_LOSS,
                N_HIDDEN_UNITS,
                N_EPOCHS,
                BATCH_SIZE,
                RANDOM_SEED,
                STEP,
                model,
                LOSS,
                OPTIMIZER,
                VALIDATION_SPLIT,
                MONITOR,
                X_TRAIN,
                Y_TRAIN,
                X_TEST,
                Y_TEST
            ).run()
    clear_session()
    del model





class create_model:

    def __init__(self,n_features,n_time_steps,
        n_classes,learning_rate,l2_loss,n_hidden_units,
        n_epochs,batch_size,random_seed,step, model,loss, 
        optimizer, validation_split,monitor,
        x_train,y_train,x_test,y_test):
        global version_counter
        self.N_FEATURES = n_features
        self.N_TIME_STEPS = n_time_steps
        self.N_CLASSES = n_classes
        self.LEARNING_RATE = learning_rate
        self.L2_LOSS = l2_loss
        self.N_HIDDEN_UNITS = n_hidden_units
        self.N_EPOCHS = n_epochs
        self.BATCH_SIZE = batch_size
        self.RANDOM_SEED = random_seed
        self.STEP = step
        self.MODEL_VERSION = version_counter
        self.start_time = time.time()
        version_counter+=1
        self.model = model
        self.LOSS = loss
        self.OPTIMIZER = optimizer
        self.VALIDATION_SPLIT = validation_split
        self.MONITOR = monitor
        self.X_TRAIN = x_train
        self.Y_TRAIN = y_train
        self.X_TEST = x_test
        self.Y_TEST = y_test


    def print_vars(self):
        print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("      MODEL_VERSION   : ",self.MODEL_VERSION)
        print("      N_TIME_STEPS    : ",self.N_TIME_STEPS)
        print("      N_HIDDEN_UNITS  : ",self.N_HIDDEN_UNITS)
        print("      N_EPOCHS        : ",self.N_EPOCHS)
        print("      BATCH_SIZE      : ",self.BATCH_SIZE)
        print("      STEP            : ",self.STEP)
        print("------------------------------------------")
        print("      LEARNING_RATE   : ",self.LEARNING_RATE)
        print("      L2_LOSS         : ",self.L2_LOSS)
        print("      N_CLASSES.      : ",self.N_CLASSES)
        print("      RANDOM_SEED     : ",self.RANDOM_SEED)
        print("      N_FEATURES      : ",self.N_FEATURES)
        print("      LOSS            : ",self.LOSS)
        print("      OPTIMIZER       : ",self.OPTIMIZER)
        print("      VALIDATION_SPLIT: ",self.VALIDATION_SPLIT)
        print("      MONITOR         : ",self.MONITOR)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")

    def run(self):

        global save_layers_info_times , version_counter
        
        global total_run
        log_message(1," Processing: V{} -- ({}/{}) ".format(self.MODEL_VERSION,self.MODEL_VERSION,total_run))


        self.model.compile(
            loss        =   self.LOSS,
            optimizer   =   self.OPTIMIZER,
            metrics     =   ['accuracy']
        )

        ## -- checkpoints ---
        history = None
        global checkpoints
        if checkpoints:
            filepath = "{}_checkpoints/model_v{}".format(thisis,self.MODEL_VERSION)
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            filepath+="/ckpt_{epoch}.hdf5"
            checkpoint = ModelCheckpoint(filepath, monitor=self.MONITOR, verbose=1, save_best_only=True, mode='auto')
            history = self.model.fit(
                self.X_TRAIN,
                self.Y_TRAIN,
                epochs = self.N_EPOCHS,
                batch_size = self.BATCH_SIZE,
                callbacks = [checkpoint],
                verbose = 1,
                validation_split = self.VALIDATION_SPLIT
            )
        else:
            history = self.model.fit(
                self.X_TRAIN,
                self.Y_TRAIN,
                epochs = self.N_EPOCHS,
                batch_size = self.BATCH_SIZE,
                verbose = 1,
                validation_split = self.VALIDATION_SPLIT
            )        

        test_loss, test_acc = self.model.evaluate(self.X_TEST,self.Y_TEST,verbose=0)
        eval_score = eval_loss_acc(test_acc,test_loss)
        
        
        # self_eval(self.model,self.X_TEST,self.Y_TEST,self.N_TIME_STEPS)

        log_message(2,"Test Loss: {} -- Test Acc: {}  -- Score: {}".format(test_loss,test_acc,eval_score))
        
        global best
        if  eval_score > best:
            best = eval_score
            path = "{}_models/model_v{}__ACC_{}__LOS_{}_".format(thisis,self.MODEL_VERSION,int(test_acc*1000),int(test_loss*1000))
            if not os.path.exists(path):
                os.makedirs(path)
            
            model_path = path+"/model.h5"
            self.model.save(model_path)

            if save_layers_info_times:
                model_plot_path = "{}_models/layers.png".format(thisis)
                plot_model(self.model,to_file=model_plot_path)
                save_layers_info_times = 0

            global cvt_to_tflite
            if cvt_to_tflite:
                cvt2tflite(self.model,self.MODEL_VERSION)
            global save_plot_ornot
            if save_plot_ornot:
                save_plot(history,path)

            run_time = time.time() - self.start_time            

            content = """
                            MODEL_VERSION           : {} 
                            N_TIME_STEPS            : {}
                            N_HIDDEN_UNITS          : {}
                            N_EPOCHS                : {}
                            BATCH_SIZE              : {}
                            STEP                    : {}
                            Test Loss               : {}
                            Test Acc                : {}
                            RUNTIME                 : {}
                            SCORE                   : {}




                      """.format(self.MODEL_VERSION , self.N_TIME_STEPS , self.N_HIDDEN_UNITS , self.N_EPOCHS , self.BATCH_SIZE , self.STEP , test_loss , test_acc , run_time,eval_score) 
            if write_to_log:
                write2log(content)
        del self.model




def main():

    N_FEATURES      = 3
    N_TIME_STEPS    = 150
    N_CLASSES       = 6
    LEARNING_RATE   = 0.0025
    L2_LOSS         = 0.0015
    N_HIDDEN_UNITS  = 64
    N_EPOCHS        = 25
    BATCH_SIZE      = 256
    RANDOM_SEED     = 382
    STEP_HAR        = 5
    STEP_CTL        = 5
    LOSS            = 'categorical_crossentropy'
    OPTIMIZER       = 'adam'
    VALIDATION_SPLIT= 0.26
    MONITORS        = ['val_loss','val_accuracy']

    N_TIME_STEPS_LIST   = [i for i in range(90,270,40)]
    N_HIDDEN_UNITS_LIST = [32,64,128]
    BATCH_SIZE_LIST     = [32,256,512,1024]
    N_EPOCHS_LIST       = [i for i in range(26,65,19)]

    global thisis
    thisis = "HAR"

    log_message(1,"Processing HAR....")
    train_df_har = read_raw_data_har(HAR_train_data_path)
    print(train_df_har.info())

    global total_run
    total_run = len(BATCH_SIZE_LIST) * len(N_EPOCHS_LIST) * len(N_TIME_STEPS_LIST)*len(N_HIDDEN_UNITS_LIST)
    

    for n_time_steps in N_TIME_STEPS_LIST:
        x_train , y_train , x_test , y_test = process_trainning_data(train_df_har,n_time_steps,STEP_HAR, N_FEATURES, RANDOM_SEED)
        print_red_on_cyan('************************\n'*2) 
        print("x_train1: {}".format(x_train.shape))
        print("y_train1: {}".format(y_train.shape)) 
        print("x_test1: {}".format(x_test.shape))
        print("y_test1: {}".format(y_test.shape))
        print_red_on_cyan('************************\n'*2) 
        for n_hidden_units in N_HIDDEN_UNITS_LIST:
            for n_epochs in N_EPOCHS_LIST:
                for n_batch_size in BATCH_SIZE_LIST:
                    one_model(N_FEATURES,
                            n_time_steps,
                            N_CLASSES,
                            LEARNING_RATE,
                            L2_LOSS,
                            n_hidden_units,
                            n_epochs,
                            n_batch_size,
                            RANDOM_SEED,
                            STEP_HAR,
                            LOSS,
                            OPTIMIZER,
                            VALIDATION_SPLIT,
                            MONITORS[1],
                            x_train,
                            y_train,
                            x_test,
                            y_test
                            )



    log_message(1,"Processing CTL....")
    global thisis
    thisis = "CTL"
    train_df_ctl = read_raw_data_ctl(CTL_train_data_path)
    print(train_df_ctl.info())

    global total_run
    total_run = len(BATCH_SIZE_LIST) * len(N_EPOCHS_LIST) * len(N_TIME_STEPS_LIST)*len(N_HIDDEN_UNITS_LIST)
    

    for n_time_steps in N_TIME_STEPS_LIST:
        x_train , y_train , x_test , y_test = process_trainning_data(train_df_har,n_time_steps,STEP_CTL, N_FEATURES, RANDOM_SEED)
        print_red_on_cyan('************************\n'*2) 
        print("x_train1: {}".format(x_train.shape))
        print("y_train1: {}".format(y_train.shape)) 
        print("x_test1: {}".format(x_test.shape))
        print("y_test1: {}".format(y_test.shape))
        print_red_on_cyan('************************\n'*2) 
        for n_hidden_units in N_HIDDEN_UNITS_LIST:
            for n_epochs in N_EPOCHS_LIST:
                for n_batch_size in BATCH_SIZE_LIST:
                    one_model(N_FEATURES,
                            n_time_steps,
                            N_CLASSES,
                            LEARNING_RATE,
                            L2_LOSS,
                            n_hidden_units,
                            n_epochs,
                            n_batch_size,
                            RANDOM_SEED,
                            STEP_CTL,
                            LOSS,
                            OPTIMIZER,
                            VALIDATION_SPLIT,
                            MONITORS[1],
                            x_train,
                            y_train,
                            x_test,
                            y_test
                            )



    # predict_tflite_model(tflite_model_path,eval_x,eval_y)

if __name__ == '__main__':
    main()


