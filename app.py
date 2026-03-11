import streamlit as st
from src.prediction import Insurance_Prediction
st.title("💰Insurance Prediction")
st.write("This is a simple insurance prediction app that predicts the insurance premium based on the input features.")
Age=st.number_input("Enter Age:", min_value=0, max_value=100, value=30)
Annual_Income_LPA=st.number_input("Enter Annual Income (LPA):", min_value=0, value=50)
Policy_Term_Years=st.number_input("Enter Policy Term (Years):", min_value=0, max_value=30, value=10)
Sum_Assured_Lakhs=st.number_input("Enter Sum Assured (Lakhs):", min_value=0, value=10)
if st.button("Predict"):
    model=Insurance_Prediction()
    result=model.prediction(Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs)
    st.success(f"The predicted insurance premium is: {result[0]:.2f} Thousands")

