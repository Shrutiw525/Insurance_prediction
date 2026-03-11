#1. load processed data from processed folder
#2. create model and train data
#3. save model in artifacts folder
import pandas as pd
x_train=pd.read_csv("../data/processed/x_train.csv")
x_test=pd.read_csv("../data/processed/x_test.csv")
y_train=pd.read_csv("../data/processed/y_train.csv")
y_test=pd.read_csv("../data/processed/y_test.csv")
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
import pickle
with open("../artifacts/model.pkl","wb") as f:
    pickle.dump(model,f)