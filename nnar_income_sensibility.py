#%% imports
from __future__ import absolute_import, division, print_function
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, LabelBinarizer
from sklearn.compose import ColumnTransformer
from pprint import pprint
import getopt, sys
import persistence
import logging
import gc

# default gpu to 0
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

# importing tf and keras
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.backend import dot, transpose, categorical_crossentropy, stack

#%% seeding random
np.random.seed(77)
tf.random.set_seed(77)

#%% parallel
import psutil
num_cpus = psutil.cpu_count(logical=False)
from joblib import parallel_backend
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects

#%% specifying model

def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(108,)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.softmax)
    ])

    return model

def weighted_categorical_crossentropy(T_male, T_female):

    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true,y_pred):
        T = K.stack([T_female.T, T_male.T])
        y_sensitive = y_true[:,0:2]
        y_true_target = y_true[:,-2:]
        T_volume = tf.tensordot(y_sensitive, T, axes=1)
       
        y_pred_target_corrected = K.batch_dot(T_volume, y_pred)
        #y_pred_corrected = K.concatenate([y_pred_sensitive,y_pred_target_corrected],axis=1)
        
        return categorical_crossentropy(y_true_target, y_pred_target_corrected)
   
    # Return a function
    return loss

def forward_categorical_crossentropy(T):

    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true,y_pred):
        pred = transpose(dot(transpose(T), transpose(y_pred)))
        return categorical_crossentropy(y_true, pred)
   
    # Return a function
    return loss



#%% pollute
def pollute(sensitive, data, T_male, T_female):
    polluted_data = data.copy()
    for i in range(data.shape[0]):
        r = np.random.rand(1)
        # if class is 0
        if sensitive[i,0] == 1:
            T = T_female
        elif sensitive[i,1] == 1:
            T = T_male
        if data[i,0] == 1:       
            # probability of false positive
            if r < T[0,1]:
                polluted_data[i,0] = 0
                polluted_data[i,1] = 1
        # if class is 1
        elif data[i,1] == 1:
            # probability of false negative
            if r < T[1,0]:
                polluted_data[i,0] = 1
                polluted_data[i,1] = 0
    return polluted_data

def worker(db_path,\
            X_train, y_train, polluted_labels,\
            X_test, y_test,\
            fp_male, fn_male, fp_female, fn_female, epochs):
    evaluater = Evaluater(db_path,\
                            X_train, y_train, polluted_labels,\
                            X_test, y_test,\
                            fp_male, fn_male, fp_female, fn_female, epochs)
    del evaluater
    _ = gc.collect()

#%% evaluater

class Evaluater(object):

    def __init__(self, directory, X_train, y_train, polluted_labels,\
                            X_test, y_test,\
                            fp_male, fn_male, fp_female, fn_female,\
                            training_epochs):
        logging.info("[%.1f,%.1f,%.1f,%.1f] I am alive! " % (fp_male, fn_male, fp_female, fn_female) )    
                             
        self.directory = directory
        self.training_epochs = training_epochs
        self.fp_male = fp_male
        self.fn_male = fn_male
        self.fp_female = fp_female
        self.fn_female = fn_female
        
        self.error_rates = [self.fp_male, self.fn_male, self.fp_female, self.fn_female]
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        self.T_male = np.array([[1-fp_male, fp_male],
                        [ fn_male , 1-fn_male]]).astype(np.float32)

        self.T_female = np.array([[1-fp_female, fp_female],
                            [ fn_female , 1-fn_female]]).astype(np.float32)

        self.polluted_labels = polluted_labels

        weighted_loss = weighted_categorical_crossentropy(self.T_male, self.T_female)
        
        # testing without correction
        test_loss, test_acc, test_pred = self.baseline_evaluate()
        self.baseline_result = [test_loss, test_acc]
        self.baseline_pred = test_pred

        # polluted data with alternating forward
        
        test_loss, test_acc, test_pred = self.weighted_evaluate(weighted_loss)
        self.weighted_forward_result = [test_loss, test_acc]
        self.weighted_forward_pred = test_pred
        #logging.info("[%.1f,%.1f,%.1f,%.1f] Alternating forward: %f" % (self.fp_male, self.fn_male, self.fp_female, self.fn_female, test_acc) )
        
        self.persist_results()
        
        logging.info("[%.1f,%.1f,%.1f,%.1f] I am done! " % (fp_male, fn_male, fp_female, fn_female) )
        del self
    
    def persist_results(self):
        db_path = os.path.join(self.directory, "baseline.db")
        conn = persistence.create_connection(db_path)
        with conn:
            persistence.persist_nnar(conn, "result", self.error_rates + self.baseline_result, self.baseline_pred)
        
        db_path = os.path.join(self.directory, "weighted_forward.db")
        conn = persistence.create_connection(db_path)
        with conn:
            persistence.persist_nnar(conn, "result", self.error_rates + self.weighted_forward_result, self.weighted_forward_pred)
        

    def weighted_evaluate(self, weighted_loss):

        # initializing model
        model = create_model()

        # specifying optmizer, loss and metrics
        model.compile(optimizer='adam', 
                    loss=weighted_loss,
                    metrics=['accuracy'])

        sensitive_and_labels = np.hstack([self.X_train[:,-2:],self.polluted_labels])
        model.fit(self.X_train, sensitive_and_labels, epochs=self.training_epochs, verbose=0)

        # testing with non polluted data
        loss, acc = model.evaluate(self.X_test, self.y_test, verbose=0)
        pred = np.argmax(model.predict(self.X_test), axis=-1)
        #pred = model.predict_classes(self.X_test)
        
        return float(loss), float(acc), pred


    def baseline_evaluate(self):

        # initializing model
        model = create_model()

        # specifying optmizer, loss and metrics
        model.compile(optimizer='adam', 
                    loss=categorical_crossentropy,
                    metrics=['accuracy'])

        model.fit(self.X_train, self.polluted_labels, epochs=self.training_epochs, verbose=0)

        # testing with non polluted data
        loss, acc = model.evaluate(self.X_test, self.y_test, verbose=0)
        pred = np.argmax(model.predict(self.X_test), axis=-1)
        #pred = model.predict_classes(self.X_test)

        
        return float(loss), float(acc), pred
      
#%%
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

#%%
def load_data():
    train = pd.read_csv("income/adult.data")
    test = pd.read_csv("income/adult.test")
    # encoding data

    categorical_features = [1,3,5,6,7,8,13]
    numerical_features = [0,2,4,10,11,12]
    sensitive_class = [9]
    target_class = [14]

    ct = ColumnTransformer([
        ("categorical_onehot", OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ("numerical", MinMaxScaler(), numerical_features),
        ("sensitive_onehot_target", OneHotEncoder(handle_unknown='ignore'), sensitive_class),
        ("categorical_target", OneHotEncoder(handle_unknown='ignore'), target_class),
        ])

    ct.fit(train)
    #pprint(train.sex)
    #pprint(train)
    parsed_test = ct.transform(test)

    # X_test, y_test splitting
    X_test = parsed_test[:,:-2].todense()
    y_test = parsed_test[:,-2:].todense()

    parsed_train = ct.transform(train).todense()

    X_train = parsed_train[:,:-2]
    y_train = parsed_train[:,-2:]

    pprint(X_train)
    pprint(X_train)
    return X_train, y_train, X_test, y_test

#%% main
def main():

    try:
        opts, _ = getopt.getopt(sys.argv[1:], "hd", ["help","directory="])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)
    directory=None
    pprint(opts)
    for o, a in opts:

        if o in ("-h", "--help"):
            sys.exit()
        elif o in ("-d", "--directory"):
            directory = a
        else:
            assert False, "unhandled option"

    # loading data
    X_train, y_train, X_test, y_test = load_data()

    # creating sqlite database
    
    log_path = os.path.join(directory, "nohup.out")
    logging.basicConfig(filename=log_path,level=logging.DEBUG)
    
    # create sqlite tables
    db_path = os.path.join(directory, "baseline.db")
    conn = persistence.create_connection(db_path)
    with conn:
        persistence.create_nnar_table(conn, "result")


    db_path = os.path.join(directory, "weighted_forward.db")
    conn = persistence.create_connection(db_path)
    with conn:
        persistence.create_nnar_table(conn, "result")


    fp_male = 0.2
    fn_male = 0.1
    fp_female = 0.1
    fn_female = 0.3

    variations = np.array([-0.1, -0.05, -0.01, 0, 0.01, 0.05, 0.1])
    fps_male = fp_male + variations
    fns_male = fn_male + variations
    fps_female = fp_female + variations
    fns_female = fn_female + variations


    T_male = np.array([[1-fp_male, fp_male],
                      [ fn_male , 1-fn_male]]).astype(np.float32)

    T_female = np.array([[1-fp_female, fp_female],
                        [ fn_female , 1-fn_female]]).astype(np.float32)
    
    y_train_polluted = pollute(X_train[:,-2:], y_train, T_male, T_female)

    error_rates = []
    for fp_male in fps_male:
        for fn_male in fns_male:
            for fp_female in fps_female:
                for fn_female in fns_female:
                    error_rates.append( (fp_male, fn_male, fp_female, fn_female) )

    epochs = 6
    n_jobs = 8

    with parallel_backend('multiprocessing'):
        for chunck in chunks(error_rates, n_jobs):    
            results = Parallel(n_jobs=n_jobs)(delayed(worker)(directory,\
                                X_train, y_train, y_train_polluted,\
                                X_test, y_test,\
                                fp_male, fn_male, fp_female, fn_female, epochs)\
                                    for fp_male, fn_male, fp_female, fn_female in chunck)
            results = None                                    

if __name__ == "__main__":
    main()

#%%
