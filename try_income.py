#%% imports
from __future__ import absolute_import, division, print_function
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer

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
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(108,)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

#%% custom loss function for forward

def forward_categorical_crossentropy(T):

    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true,y_pred):
        pred = transpose(dot(transpose(T), transpose(y_pred)))
        return categorical_crossentropy(y_true, pred)
   
    # Return a function
    return loss
#%% non-sense transition matrix

T = np.array([[0.4, 0.6],[0.65, 0.35]]).astype(np.float32)


#%% specifying optmizer, loss and metrics

model.compile(optimizer='adam', 
              loss=forward_categorical_crossentropy(T),
              metrics=['accuracy'])

#%% traning
model.fit(X_train, y_train, epochs=5)
#%% testing
test_loss, test_acc = model.evaluate(X_test, y_test)

print('Test accuracy:', test_acc)
#%% making prediction
predictions = model.predict(X_test)
