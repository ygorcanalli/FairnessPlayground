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

#%% reading training and test data

train = pd.read_csv("income/adult_treinamento2.csv")
test = pd.read_csv("income/adult_teste2.csv")

#%% encoding data

categorical_features = [1,3,5,6,7,8,9,13]
numerical_features = [0,2,4,10,11,12]
target_class = [14]

ct = ColumnTransformer([
    ("categorical_onehot", OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ("numerical", MinMaxScaler(), numerical_features),
    ("categorical_onehot_target", OneHotEncoder(handle_unknown='ignore'), target_class)
    ])

train = ct.fit_transform(train)
test = ct.transform(test)

#%% X, y splitting
X_train = train[:,:-2]
y_train = train[:,-2:]

X_test = test[:,:-2]
y_test = test[:,-2:]

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
    for i in range(data.size):
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
    return model.evaluate(X_test, y_test)

#%% baseline
test_loss, test_acc = evaluate(X_train, X_test, y_train, y_test,
                                polluted_y_data=None, loss_function=categorical_crossentropy)
print('Baseline test accuracy:', test_acc)

baseline_result = test_acc

#%% evaluating different error rates

false_positive_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
false_negative_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

without_forward_results = np.zeros(len(false_positive_rates), len(false_negative_rates))
forward_results = np.zeros(len(false_positive_rates), len(false_negative_rates))

for i, fp in enumerate(false_negative_rates):
    for j, fn in enumerate(false_negative_rates):
        print("fp_rate: ", fp, " fn_rate: ", fn)
        T = np.array([[1-fp, fp],
                      [fn  , 1-fn]]).astype(np.float32)

        polluted_labels = pollute(y_train, T)
        forward_function = forward_categorical_crossentropy(T)

        # polluted data without forward
        test_loss, test_acc = evaluate(X_train, X_test, y_train, y_test,
                                        polluted_y_data=polluted_labels,
                                        loss_function=categorical_crossentropy)
        print('Test accuracy without forward:', test_acc)
        without_forward_results[i,j] = test_acc
        # polluted data with forward
        test_loss, test_acc = evaluate(X_train, X_test, y_train, y_test,
                                        polluted_y_data=polluted_labels,
                                        loss_function=forward_function)
        print('Test accuracy with forward:', test_acc)
        forward_results[i,j] = test_acc

print('Baseline test accuracy:', baseline_result)
print("Results without forward:")
pprint(without_forward_results)
print("Results with forward:")
pprint(forward_results)