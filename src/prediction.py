# 1. load scaler.pkl and model.pkl files because we need to use them for prediction because model.pkl file contains the trained model and scaler.pkl file contains the scaler object which we used for scaling the data during training.
# 2. create a function to predict
import pickle
import os
class Insurance_Prediction:
    def __init__(self):
        base_path = os.path.dirname(os.path.abspath(_file_))
        artifacts_path = os.path.join(base_path, "..", "artifacts")

        scaler_file = os.path.join(artifacts_path, "scaler.pkl")
        with open(scaler_file, "rb") as f:
            self.scaler = pickle.load(f)

        model_file = os.path.join(artifacts_path, "model.pkl")
        with open(model_file, "rb") as f:
            self.model = pickle.load(f)


    def prediction(self, Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs):
        import numpy as np
        input_data=np.array([[Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs]])
        scaled_input=self.scaler.transform(input_data)
        result=self.model.predict(scaled_input)
        return result[0]
    
