#%% imports
from __future__ import absolute_import, division, print_function
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from pprint import pprint
import getopt, sys
import persistence
import logging

# default gpu to 0
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

# importing tf and keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
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
        keras.layers.Flatten(input_shape=(108,)),
        keras.layers.Dense(128, activation=tf.nn.relu),
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
#%% evaluater

class Evaluater(object):

    def __init__(self, db_path, X_train_male, y_train_male,\
                            X_train_female, y_train_female, \
                            X_test, y_test,\
                            fp_male, fn_male, fp_female, fn_female,\
                            training_epochs):
        logging.info("[%.1f,%.1f,%.1f,%.1f] I am alive! " % (fp_male, fn_male, fp_female, fn_female) )    
                             
        self.db_path = db_path
        self.training_epochs = training_epochs
        self.fp_male = fp_male
        self.fn_male = fn_male
        self.fp_female = fp_female
        self.fn_female = fn_female
        
        self.error_rates = [self.fp_male, self.fn_male, self.fp_female, self.fn_female]
        self.X_train_male = X_train_male
        self.X_train_female = X_train_female
        self.y_train_male = y_train_male
        self.y_train_female = y_train_female
        self.X_test = X_test
        self.y_test = y_test
        
        self.T_male = np.array([[1-fp_male, fp_male],
                        [ fn_male , 1-fn_male]]).astype(np.float32)

        self.T_female = np.array([[1-fp_female, fp_female],
                            [ fn_female , 1-fn_female]]).astype(np.float32)

        self.polluted_male_labels = pollute(self.y_train_male, self.T_male)
        self.polluted_female_labels = pollute(self.y_train_female, self.T_female)

        self.X_train = np.vstack([self.X_train_male, self.X_train_female])
        self.y_train = np.vstack([self.y_train_male, self.y_train_female])
        self.polluted_labels = np.vstack([self.polluted_male_labels, self.polluted_female_labels])
        
        male_loss = forward_categorical_crossentropy(self.T_male)
        female_loss = forward_categorical_crossentropy(self.T_female)
        
        # testing with non polluted data
        test_loss, test_acc, test_pred = self.baseline()
        self.baseline_result = [test_loss, test_acc]
        self.baseline_pred = test_pred
        #logging.info("[%.1f,%.1f,%.1f,%.1f] Baseline: %f" % (self.fp_male, self.fn_male, self.fp_female, self.fn_female, test_acc))

        # polluted data with two step forward
        test_loss, test_acc, test_pred = self.two_step_evaluate(male_loss, female_loss)
        self.two_step_forward_result = [test_loss, test_acc]
        self.two_step_forward_pred = test_pred
        #logging.info("[%.1f,%.1f,%.1f,%.1f] Two step forward: %f" % (self.fp_male, self.fn_male, self.fp_female, self.fn_female, test_acc) )

        # polluted data with alternating forward
        test_loss, test_acc, test_pred = self.alternating_evaluate(male_loss, female_loss)
        self.alternating_forward_result = [test_loss, test_acc]
        self.alternating_forward_pred = test_pred
        #logging.info("[%.1f,%.1f,%.1f,%.1f] Alternating forward: %f" % (self.fp_male, self.fn_male, self.fp_female, self.fn_female, test_acc) )
        self.persist_results()

        logging.info("[%.1f,%.1f,%.1f,%.1f] I am done! " % (fp_male, fn_male, fp_female, fn_female) )
    
    def persist_results(self):
        conn = persistence.create_connection(self.db_path)
        
        with conn:
            persistence.persist_nnar(conn, "baseline", self.error_rates + self.baseline_result, self.baseline_pred)
            persistence.persist_nnar(conn, "two_step_forward", self.error_rates + self.two_step_forward_result, self.two_step_forward_pred)
            persistence.persist_nnar(conn, "alternating_forward", self.error_rates + self.alternating_forward_result, self.alternating_forward_pred)

    def two_step_evaluate(self, male_loss, female_loss):

        # initializing model
        model = create_model()

        # specifying optmizer, loss and metrics
        model.compile(optimizer='adam', 
                    loss=female_loss,
                    metrics=['accuracy'])

        model.fit(self.X_train_female, self.polluted_female_labels, epochs=self.training_epochs, verbose=0)


        # specifying optmizer, loss and metrics
        model.compile(optimizer='adam', 
                    loss=male_loss,
                    metrics=['accuracy'])

        model.fit(self.X_train_male, self.polluted_male_labels, epochs=self.training_epochs, verbose=0)

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

            model.fit(self.X_train_female, self.polluted_female_labels, epochs=1, verbose=0)

            # specifying optmizer, loss and metrics
            model.compile(optimizer='adam', 
                        loss=male_loss,
                        metrics=['accuracy'])

            model.fit(self.X_train_male, self.polluted_male_labels, epochs=1, verbose=0)

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

        model.fit(self.X_train, self.y_train, epochs=self.training_epochs, verbose=0)

        # testing with non polluted data
        loss, acc = model.evaluate(self.X_test, self.y_test, verbose=0)
        pred = model.predict_classes(self.X_test)

        
        return float(loss), float(acc), pred
      
#%%
def load_data():
    train = pd.read_csv("income/adult_treinamento2.csv")
    test = pd.read_csv("income/adult_teste2.csv")
    # encoding data

    categorical_features = [1,3,5,6,7,8,9,13]
    numerical_features = [0,2,4,10,11,12]
    target_class = [14]

    ct = ColumnTransformer([
        ("categorical_onehot", OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ("numerical", MinMaxScaler(), numerical_features),
        ("categorical_onehot_target", OneHotEncoder(handle_unknown='ignore'), target_class)
        ])

    ct.fit(train)
    parsed_test = ct.transform(test)

    # X_test, y_test splitting
    X_test = parsed_test[:,:-2].todense()
    y_test = parsed_test[:,-2:].todense()

    # indentifying males and females
    train_male = train[train.sex == "Male"]
    train_female = train[train.sex == "Female"]

    parsed_train_male = ct.transform(train_male)
    parsed_train_female = ct.transform(train_female)

    X_train_male = parsed_train_male[:,:-2].todense()
    y_train_male = parsed_train_male[:,-2:].todense()

    X_train_female = parsed_train_female[:,:-2].todense()
    y_train_female = parsed_train_female[:,-2:].todense()

    return X_train_male, y_train_male, X_train_female, y_train_female, X_test, y_test
#%%
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
    X_train_male, y_train_male, X_train_female, y_train_female, X_test, y_test = load_data()

    # creating sqlite database
    db_path = os.path.join(directory, "result.db")
    log_path = os.path.join(directory, "nohup.out")
    logging.basicConfig(filename=log_path,level=logging.DEBUG)
    conn = persistence.create_connection(db_path)

    # create sqlite tables
    with conn:
        persistence.create_nnar_table(conn, "baseline")
        persistence.create_nnar_table(conn, "two_step_forward")
        persistence.create_nnar_table(conn, "alternating_forward")

    fps_male = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    fns_male = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    fps_female = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    fns_female = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    error_rates = []
    for fp_male in fps_male:
        for fn_male in fns_male:
            for fp_female in fps_female:
                for fn_female in fns_female:
                    error_rates.append( (fp_male, fn_male, fp_female, fn_female) )

    epochs = 6

    with parallel_backend('loky'):
        Parallel(n_jobs=num_cpus)(delayed(Evaluater)(db_path,\
                            X_train_male, y_train_male,\
                            X_train_female, y_train_female,\
                            X_test, y_test,\
                            fp_male, fn_male, fp_female, fn_female, epochs)\
                                for fp_male, fn_male, fp_female, fn_female in error_rates)

if __name__ == "__main__":
    main()

#%%
