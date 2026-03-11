# 1. load scaler.pkl and model.pkl files because we need to use them for prediction because model.pkl file contains the trained model and scaler.pkl file contains the scaler object which we used for scaling the data during training.
# 2. create a function to predict
import pickle
class Insurance_Prediction:
    def __init__(self):
        with open("D:\\TEKWORKS\\day13\\Insurance_prediction\\artifacts\\scaler.pkl","rb") as f: #it has scaled values of x_train and x_test data
            self.scaler=pickle.load(f)
        with open("D:\\TEKWORKS\\day13\\Insurance_prediction\\artifacts\\model.pkl","rb") as f: #the trained model is stored in this file
            self.model=pickle.load(f)

    def prediction(self, Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs):
        import numpy as np
        input_data=np.array([[Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs]])
        scaled_input=self.scaler.transform(input_data)
        result=self.model.predict(scaled_input)
        return result[0]
    