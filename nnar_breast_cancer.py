#%% imports
from __future__ import absolute_import, division, print_function
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
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
from tensorflow.keras.backend import dot, transpose, categorical_crossentropy

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
        keras.layers.Flatten(input_shape=(33,)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.softmax)
    ])

    return model
#%% custom loss function for forward

def forward_categorical_crossentropy(T):

    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true,y_pred):
        pred = transpose(dot(transpose(T), transpose(y_pred)))
        return categorical_crossentropy(y_true, pred)
   
    # Return a function
    return loss

#%% custom loss function for backward

def backward_categorical_crossentropy(T):

    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true,y_pred):
        base_loss = categorical_crossentropy(y_true, y_pred)
        return dot(np.linalg.pinv(T), base_loss)
   
    # Return a function
    return loss

#%% pollute
def pollute(data, T):
    polluted_data = data.copy()
    for i in range(data.shape[0]):
        r = np.random.rand(1)
        # if class is 0
        if data[i,0] == 1:       
            # probability of false positive
            if r < T[0,1]:
                polluted_data[i,0] = 0
                polluted_data[i,1] = 1
        # if class is 1
        elif data[i,1] == 1:
            # probability of false negative
            if r < T[1,0]:
                polluted_data[i,1] = 0
                polluted_data[i,0] = 1
    return polluted_data

def worker(db_path,\
            X_train_white, y_train_white,\
            X_train_black, y_train_black,\
            X_test, y_test,\
            fp_white, fn_white, fp_black, fn_black, epochs):
    evaluater = Evaluater(db_path,\
                            X_train_white, y_train_white,\
                            X_train_black, y_train_black,\
                            X_test, y_test,\
                            fp_white, fn_white, fp_black, fn_black, epochs)
    del evaluater
    _ = gc.collect()

#%% evaluater

class Evaluater(object):

    def __init__(self, directory, X_train_white, y_train_white,\
                            X_train_black, y_train_black, \
                            X_test, y_test,\
                            fp_white, fn_white, fp_black, fn_black,\
                            training_epochs):
        logging.info("[%.1f,%.1f,%.1f,%.1f] I am alive! " % (fp_white, fn_white, fp_black, fn_black) )    
                             
        self.directory = directory
        self.training_epochs = training_epochs
        self.fp_white = fp_white
        self.fn_white = fn_white
        self.fp_black = fp_black
        self.fn_black = fn_black
        
        self.error_rates = [self.fp_white, self.fn_white, self.fp_black, self.fn_black]
        self.X_train_white = X_train_white
        self.X_train_black = X_train_black
        self.y_train_white = y_train_white
        self.y_train_black = y_train_black
        self.X_test = X_test
        self.y_test = y_test
        
        self.T_white = np.array([[1-fp_white, fp_white],
                        [ fn_white , 1-fn_white]]).astype(np.float32)

        self.T_black = np.array([[1-fp_black, fp_black],
                            [ fn_black , 1-fn_black]]).astype(np.float32)

        self.polluted_white_labels = pollute(self.y_train_white, self.T_white)
        self.polluted_black_labels = pollute(self.y_train_black, self.T_black)

        self.X_train = np.vstack([self.X_train_white, self.X_train_black])
        self.y_train = np.vstack([self.y_train_white, self.y_train_black])
        self.polluted_labels = np.vstack([self.polluted_white_labels, self.polluted_black_labels])
        
        male_loss = forward_categorical_crossentropy(self.T_white)
        female_loss = forward_categorical_crossentropy(self.T_black)
        
        # testing without correction
        test_loss, test_acc, test_pred = self.baseline()
        self.baseline_result = [test_loss, test_acc]
        self.baseline_pred = test_pred
        #logging.info("[%.1f,%.1f,%.1f,%.1f] Baseline: %f" % (self.fp_white, self.fn_white, self.fp_black, self.fn_black, test_acc))
        
        # polluted data with two step forward
        test_loss, test_acc, test_pred = self.two_step_evaluate(male_loss, female_loss)
        self.two_step_forward_result = [test_loss, test_acc]
        self.two_step_forward_pred = test_pred
        #logging.info("[%.1f,%.1f,%.1f,%.1f] Two step forward: %f" % (self.fp_white, self.fn_white, self.fp_black, self.fn_black, test_acc) )

        # polluted data with alternating forward
        test_loss, test_acc, test_pred = self.alternating_evaluate(male_loss, female_loss)
        self.alternating_forward_result = [test_loss, test_acc]
        self.alternating_forward_pred = test_pred
        #logging.info("[%.1f,%.1f,%.1f,%.1f] Alternating forward: %f" % (self.fp_white, self.fn_white, self.fp_black, self.fn_black, test_acc) )
        self.persist_results()
        
        logging.info("[%.1f,%.1f,%.1f,%.1f] I am done! " % (fp_white, fn_white, fp_black, fn_black) )
        del self
    
    def persist_results(self):
        db_path = os.path.join(self.directory, "baseline.db")
        conn = persistence.create_connection(db_path)
        with conn:
            persistence.persist_nnar(conn, "result", self.error_rates + self.baseline_result, self.baseline_pred)
        
        db_path = os.path.join(self.directory, "two_step_forward.db")
        conn = persistence.create_connection(db_path)
        with conn:
            persistence.persist_nnar(conn, "result", self.error_rates + self.two_step_forward_result, self.two_step_forward_pred)

        db_path = os.path.join(self.directory, "alternating_forward.db")
        conn = persistence.create_connection(db_path)
        with conn:
            persistence.persist_nnar(conn, "result", self.error_rates + self.alternating_forward_result, self.alternating_forward_pred)

    def two_step_evaluate(self, male_loss, female_loss):

        # initializing model
        model = create_model()

        # specifying optmizer, loss and metrics
        model.compile(optimizer='adam', 
                    loss=female_loss,
                    metrics=['accuracy'])

        model.fit(self.X_train_black, self.polluted_black_labels, epochs=self.training_epochs, verbose=0)


        # specifying optmizer, loss and metrics
        model.compile(optimizer='adam', 
                    loss=male_loss,
                    metrics=['accuracy'])

        model.fit(self.X_train_white, self.polluted_white_labels, epochs=self.training_epochs, verbose=0)

        # testing with non polluted data
        loss, acc = model.evaluate(self.X_test, self.y_test, verbose=0)
        pred = model.predict_classes(self.X_test)

        
        return float(loss), float(acc), pred

    def alternating_evaluate(self, male_loss, female_loss):

        # initializing model
        model = create_model()

        for _ in range(self.training_epochs): 
            # specifying optmizer, loss and metrics
            model.compile(optimizer='adam', 
                        loss=female_loss,
                        metrics=['accuracy'])

            model.fit(self.X_train_black, self.polluted_black_labels, epochs=1, verbose=0)

            # specifying optmizer, loss and metrics
            model.compile(optimizer='adam', 
                        loss=male_loss,
                        metrics=['accuracy'])

            model.fit(self.X_train_white, self.polluted_white_labels, epochs=1, verbose=0)

        loss, acc = model.evaluate(self.X_test, self.y_test, verbose=0)
        pred = model.predict_classes(self.X_test)

        # testing with non polluted data
        return float(loss), float(acc), pred

    def baseline(self):

        # initializing model
        model = create_model()

        # specifying optmizer, loss and metrics
        model.compile(optimizer='adam', 
                    loss=categorical_crossentropy,
                    metrics=['accuracy'])

        model.fit(self.X_train, self.polluted_labels, epochs=self.training_epochs, verbose=0)

        # testing with non polluted data
        loss, acc = model.evaluate(self.X_test, self.y_test, verbose=0)
        pred = model.predict_classes(self.X_test)

        
        return float(loss), float(acc), pred
      
#%%
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

#%%
def load_data():
    breast_cancer_dataset = load_breast_cancer(as_frame=True)

    X = breast_cancer_dataset.data
    y = breast_cancer_dataset.target

    #create random race
    race = pd.DataFrame(np.where(np.random.normal(0.0, 1.0, size=569)<=0,'White','Black'), columns=['race'])
    age = pd.DataFrame((np.random.randn(569) * 12 + 50).astype('int'), columns=['age'])

    X_extended = pd.concat([X,age,race], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X_extended, y, test_size=0.33, random_state=42)

    categorical_features = [31]
    numerical_features = [x for x in range(0,31)]
    ct = ColumnTransformer([
        ("categorical_onehot", OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ("numerical", MinMaxScaler(), numerical_features),
        ])

    ct.fit(X_train)

    yct = OneHotEncoder()
    yct.fit(y_train.to_numpy().reshape(-1,1))

    y_train_white = y_train[X_train.race == "White"]
    y_train_black = y_train[X_train.race == "Black"] 

    # indentifying males and females
    X_train_white = X_train[X_train.race == "White"]
    X_train_black = X_train[X_train.race == "Black"]

    X_train_white = ct.transform(X_train_white)
    X_train_black = ct.transform(X_train_black)
    X_test = ct.transform(X_test)

    y_train_white = yct.transform(y_train_white.to_numpy().reshape(-1,1)).todense()
    y_train_black = yct.transform(y_train_black.to_numpy().reshape(-1,1)).todense()
    y_test = yct.transform(y_test.to_numpy().reshape(-1,1)).todense()

    return X_train_white, y_train_white, X_train_black, y_train_black, X_test, y_test

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
            assert False, "unhaldled option"
        
    # configuring log
    

    # loading data
    X_train_white, y_train_white, X_train_black, y_train_black, X_test, y_test = load_data()

    # creating sqlite database
    
    log_path = os.path.join(directory, "nohup.out")
    logging.basicConfig(filename=log_path,level=logging.DEBUG)
    
    db_path = os.path.join(directory, "baseline.db")
    conn = persistence.create_connection(db_path)

    # create sqlite tables
    db_path = os.path.join(directory, "baseline.db")
    conn = persistence.create_connection(db_path)
    with conn:
        persistence.create_nnar_table(conn, "result")
    
    db_path = os.path.join(directory, "two_step_forward.db")
    conn = persistence.create_connection(db_path)
    with conn:
        persistence.create_nnar_table(conn, "result")
    
    db_path = os.path.join(directory, "alternating_forward.db")
    conn = persistence.create_connection(db_path)
    with conn:
        persistence.create_nnar_table(conn, "result")

    fps_white = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    fns_white = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    fps_black = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    fns_black = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    error_rates = []
    for fp_white in fps_white:
        for fn_white in fns_white:
            for fp_black in fps_black:
                for fn_black in fns_black:
                    error_rates.append( (fp_white, fn_white, fp_black, fn_black) )

    epochs = 6
    n_jobs = 8

    with parallel_backend('multiprocessing'):
        for chunck in chunks(error_rates, n_jobs):    
            results = Parallel(n_jobs=n_jobs)(delayed(worker)(directory,\
                                X_train_white, y_train_white,\
                                X_train_black, y_train_black,\
                                X_test, y_test,\
                                fp_white, fn_white, fp_black, fn_black, epochs)\
                                    for fp_white, fn_white, fp_black, fn_black in chunck)
            results = None                                    

if __name__ == "__main__":
    main()

#%%
