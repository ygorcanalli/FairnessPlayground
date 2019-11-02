#%%imports

from nnar_income import create_model
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from tensorflow.keras.backend import categorical_crossentropy
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
from pprint import pprint

#%% load data

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

parsed_train = ct.fit_transform(train)
parsed_test = ct.transform(test)

# X_test, y_test splitting
X_test = parsed_test[:,:-2].todense()
y_test = parsed_test[:,-2:].todense()

X_train = parsed_train[:,:-2].todense()
y_train = parsed_train[:,-2:].todense()

#%% pollute
def pollute(data, sex, T_male, T_female):
    polluted_data = data.copy()
    for i in range(data.shape[0]):
        r = np.random.rand(1)
        # if class is 0
        if sex[i] == 'Male':
            if data[i,0] == 1:       
                # probability of false positive
                if r < T_male[0,1]:
                    polluted_data[i,0] = 0
                    polluted_data[i,1] = 1
            # if class is 1
            elif data[i,1] == 1:
                # probability of false negative
                if r < T_male[1,0]:
                    polluted_data[i,1] = 0
                    polluted_data[i,0] = 1
        if sex[i] == 'Female':
            if data[i,0] == 1:       
                # probability of false positive
                if r < T_female[0,1]:
                    polluted_data[i,0] = 0
                    polluted_data[i,1] = 1
            # if class is 1
            elif data[i,1] == 1:
                # probability of false negative
                if r < T_female[1,0]:
                    polluted_data[i,1] = 0
                    polluted_data[i,0] = 1
    return polluted_data

#%% polluting data 


fp_female = 0.0
fn_female = 0.0
fp_male = 0.0
fn_male = 0.0
        
T_male = np.array([[1-fp_male, fp_male],
                [ fn_male , 1-fn_male]]).astype(np.float32)

T_female = np.array([[1-fp_female, fp_female],
                    [ fn_female , 1-fn_female]]).astype(np.float32)

polluted_y_train = pollute(y_train, train.sex, T_male, T_female)

#%% traning with polluted data
training_epochs = 6
# initializing model
model = create_model()

# specifying optmizer, loss and metrics
model.compile(optimizer='adam', 
            loss=categorical_crossentropy,
            metrics=['accuracy'])

model.fit(X_train, polluted_y_train, epochs=training_epochs, verbose=0)

# testing with non polluted dataUngenered
loss, acc = model.evaluate(X_test, y_test, verbose=0)
pred = model.predict_classes(X_test)

#%% predictions by sex

y_test_male = y_test[test.sex == "Male"]
y_test_female = y_test[test.sex == "Female"]
 
pred_male = pred[test.sex == "Male"]
pred_female = pred[test.sex == "Female"]


#%% confusion matrix

confusion_ungenered = confusion_matrix( y_test[:,1],pred)/pred.shape[0]
confusion_male = confusion_matrix( y_test_male[:,1],pred_male)/pred_male.shape[0]
confusion_female = confusion_matrix( y_test_female[:,1],pred_female)/pred_female.shape[0]

print("Ungenered confusion matrix")
pprint(confusion_ungenered)
print(" Male confusion matrix")
pprint(confusion_male)
print("Male transition_matrix")
pprint(T_male)
print(" Female confusion matrix")
pprint(confusion_female)
print("Female transition matrix")
pprint(T_male)
#%%
