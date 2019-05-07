#%%
import pandas as pd

train = pd.read_csv("income/adult_treinamento2.csv")
test = pd.read_csv("income/adult_teste2.csv")

X_train = train[['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']]

y_train = train['class']

X_test = test[['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']]

y_test = test['class']

#%%
