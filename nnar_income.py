#%% imports
from __future__ import absolute_import, division, print_function
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from pprint import pprint

# default gpu to 0
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# importing tf and keras
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras.backend import dot, transpose, categorical_crossentropy

#%% seeding random

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

#%% specifying model

def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(108,)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.softmax)
    ])

    return model


#%% pollute functions
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

#%% defining evaluate function

def evaluate(X_train, X_test, y_train, y_test, 
                model_function=create_model,
                polluted_y_data=None, traning_epochs=5,
                loss_function=categorical_crossentropy):

    # initializing model
    model = model_function()

    # specifying optmizer, loss and metrics
    model.compile(optimizer='adam', 
                loss=loss_function,
                metrics=['accuracy'])

    # disable data pollution
    if polluted_y_data is None:
        model.fit(X_train, y_train, epochs=traning_epochs)
    # disable data pollution
    else:
        model.fit(X_train, polluted_y_data, epochs=traning_epochs)
  
    # testing with non polluted data
    loss, acc = model.evaluate(X_test, y_test)
    pred = model.predict_classes(X_test)
    return loss, acc, pred

#%%
def two_step_evaluate(X_train_1st, y_train_1st,
                    X_train_2nd, y_train_2nd,
                    X_test,y_test, 
                model_function=create_model,
                traning_epochs=5,
                loss_function_1st=categorical_crossentropy,
                loss_function_2nd=categorical_crossentropy):

    # initializing model
    model = create_model()

    # specifying optmizer, loss and metrics
    model.compile(optimizer='adam', 
                loss=loss_function_1st,
                metrics=['accuracy'])

    model.fit(X_train_2nd, y_train_2nd, epochs=traning_epochs)

    # specifying optmizer, loss and metrics
    model.compile(optimizer='adam', 
                loss=loss_function_2nd,
                metrics=['accuracy'])

    model.fit(X_train_male, polluted_male_labels, epochs=traning_epochs)

    loss, acc = model.evaluate(X_test, y_test)
    pred = model.predict_classes(X_test)

    # testing with non polluted data
    return loss, acc, pred

def alternating_evaluate(X_train_1st, y_train_1st,
                    X_train_2nd, y_train_2nd,
                    X_test,y_test, 
                model_function=create_model,
                traning_epochs=5,
                loss_function_1st=categorical_crossentropy,
                loss_function_2nd=categorical_crossentropy):

    # initializing model
    model = create_model()

    for i in range(traning_epochs): 
        # specifying optmizer, loss and metrics
        model.compile(optimizer='adam', 
                    loss=loss_function_1st,
                    metrics=['accuracy'])

        model.fit(X_train_2nd, y_train_2nd, epochs=1)

        # specifying optmizer, loss and metrics
        model.compile(optimizer='adam', 
                    loss=loss_function_2nd,
                    metrics=['accuracy'])

        model.fit(X_train_male, polluted_male_labels, epochs=1)

    loss, acc = model.evaluate(X_test, y_test)
    pred = model.predict_classes(X_test)

    # testing with non polluted data
    return loss, acc, pred

#%%
def append_result(df, params, pred):
    a = np.hstack([params, pred])
    df.loc[df.shape[0] + 1 + 1] = a
#%% reading training and test data

train = pd.read_csv("income/adult_treinamento2.csv")
test = pd.read_csv("income/adult_teste2.csv")

for k in test.keys():
    categories = test[k].value_counts()
    print(categories)
    print("\n")

test.describe()

#%% encoding data

categorical_features = [1,3,5,6,7,8,9,13]
numerical_features = [0,2,4,10,11,12]
target_class = [14]

ct = ColumnTransformer([
    ("categorical_onehot", OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ("numerical", MinMaxScaler(), numerical_features),
    ("categorical_onehot_target", OneHotEncoder(handle_unknown='ignore'), target_class)
    ])

parsed_train = ct.fit_transform(train)
parsed_test = ct.transform(test)

#%% X, y splitting
X_train = parsed_train[:,:-2]
y_train = parsed_train[:,-2:].todense()

X_test = parsed_test[:,:-2]
y_test = parsed_test[:,-2:].todense()

#%% baseline
test_loss, test_acc, test_pred = evaluate(X_train, X_test, y_train, y_test,
                                polluted_y_data=None,
                                loss_function=categorical_crossentropy,
                                traning_epochs=6)
print('Baseline test accuracy:', test_acc, test_pred)

baseline_result = test_acc


#%% pandas output dataframes

cols = ['fp_male', 'fn_male', 'fp_female', 'fn_female', 'loss', 'acc']
for i in range(y_test.shape[0]):
    cols.append("pred%d" % i)
    
baseline = pd.DataFrame(columns=cols)
two_step_forward_half = pd.DataFrame(columns=cols)
two_step_forward = pd.DataFrame(columns=cols)
alternating_forward_half = pd.DataFrame(columns=cols)
alternating_forward = pd.DataFrame(columns=cols)

fps_male = np.arange(0, 0.15, 0.05)
fns_male = np.arange(0, 0.15, 0.05)
fps_female = np.arange(0, 0.15, 0.05)
fns_female = np.arange(0, 0.15, 0.05)

for fp_male in fps_male:
    for fn_male in fns_male:
        for fp_female in fps_female:
            for fn_female in fns_female:

                #%% nnar noise
                T_male = np.array([[1-fp_male, fp_male],
                                [ fn_male , 1-fn_male]]).astype(np.float32)

                T_female = np.array([[1-fp_female, fp_female],
                                    [ fn_female , 1-fn_female]]).astype(np.float32)

                forward_male_loss = forward_categorical_crossentropy(T_male)
                forward_female_loss = forward_categorical_crossentropy(T_female)

                # indentifying males and females
                train_male = train[train.sex == "Male"]
                train_female = train[train.sex == "Female"]

                parsed_train_male = ct.transform(train_male)
                parsed_train_female = ct.transform(train_female)

                X_train_male = parsed_train_male[:,:-2].todense()
                y_train_male = parsed_train_male[:,-2:].todense()

                X_train_female = parsed_train_female[:,:-2].todense()
                y_train_female = parsed_train_female[:,-2:].todense()

                polluted_male_labels = pollute(y_train_male, T_male)
                polluted_female_labels = pollute(y_train_female, T_female)

                X_train = np.vstack([X_train_male, X_train_female])
                polluted_labels = np.vstack([polluted_male_labels, polluted_female_labels])

                #%%

                # polluted data without forward
                test_loss, test_acc, test_pred = evaluate(X_train, X_test, y_train, y_test,
                                                polluted_y_data=polluted_labels,
                                                loss_function=categorical_crossentropy,
                                                traning_epochs=6)
                append_result(baseline,
                                [fp_male, fn_male, fp_female, fn_female, test_loss, test_acc], 
                                test_pred)
                pprint(baseline)
                #%%

                # polluted data with two step forward on half epochs
                test_loss, test_acc, test_pred = two_step_evaluate(X_train_female, polluted_female_labels,
                                                        X_train_male, polluted_male_labels,
                                                        X_test,y_test, 
                                                        model_function=create_model,
                                                        traning_epochs=3,
                                                        loss_function_1st=forward_female_loss,
                                                        loss_function_2nd=forward_male_loss)
                append_result(two_step_forward_half,
                                [fp_male, fn_male, fp_female, fn_female, test_loss, test_acc], 
                                test_pred)
                pprint(two_step_forward_half)
                #%%

                # polluted data with two step forward
                test_loss, test_acc, test_pred = two_step_evaluate(X_train_female, polluted_female_labels,
                                                        X_train_male, polluted_male_labels,
                                                        X_test,y_test, 
                                                        model_function=create_model,
                                                        traning_epochs=6,
                                                        loss_function_1st=forward_female_loss,
                                                        loss_function_2nd=forward_male_loss)
                append_result(two_step_forward,
                                [fp_male, fn_male, fp_female, fn_female, test_loss, test_acc], 
                                test_pred)
                pprint(two_step_forward)
                #%%

                # polluted data with alternating forward on half epochs
                test_loss, test_acc, test_pred = alternating_evaluate(X_train_female, polluted_female_labels,
                                                        X_train_male, polluted_male_labels,
                                                        X_test,y_test, 
                                                        model_function=create_model,
                                                        traning_epochs=3,
                                                        loss_function_1st=forward_female_loss,
                                                        loss_function_2nd=forward_male_loss)
                append_result(alternating_forward_half,
                                [fp_male, fn_male, fp_female, fn_female, test_loss, test_acc], 
                                test_pred)
                pprint(alternating_forward_half)

                # polluted data with alternating forward
                test_loss, test_acc, test_pred = alternating_evaluate(X_train_female, polluted_female_labels,
                                                        X_train_male, polluted_male_labels,
                                                        X_test,y_test, 
                                                        model_function=create_model,
                                                        traning_epochs=6,
                                                        loss_function_1st=forward_female_loss,
                                                        loss_function_2nd=forward_male_loss)
                append_result(alternating_forward,
                                [fp_male, fn_male, fp_female, fn_female, test_loss, test_acc], 
                                test_pred)
                pprint(alternating_forward)
