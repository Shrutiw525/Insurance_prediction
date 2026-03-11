# 1. load training and testing data
# 2. scale(converting data to binary format) the training data
# 3. save scaled data into processed folder
from data_preprocessing import load_and_split_data
x_train,x_test,y_train,y_test=load_and_split_data()

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

import pandas as pd
pd.DataFrame(x_train_scaled).to_csv("../data/processed/x_train.csv",index=False)
pd.DataFrame(x_test_scaled).to_csv("../data/processed/x_test.csv",index=False)
pd.DataFrame(y_train).to_csv("../data/processed/y_train.csv",index=False)
pd.DataFrame(y_test).to_csv("../data/processed/y_test.csv",index=False)

import pickle
with open("../artifacts/scaler.pkl","wb") as f:
    pickle.dump(scaler,f)