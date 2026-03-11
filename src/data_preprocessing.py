# 1. loading original/raw data
# 2. identifying x and y(input and output feature)
# 3. split data into train and test 
import pandas as pd
def load_and_split_data():
    df=pd.read_csv("../data/raw/insurance_data.csv") 
    x=df[["Age","Annual_Income_LPA","Policy_Term_Years","Sum_Assured_Lakhs"]]
    y=df["Annual_Premium_Thousands"]
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    return x_train,x_test,y_train,y_test
 