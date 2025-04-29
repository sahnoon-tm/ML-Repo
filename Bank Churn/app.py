import streamlit as st
import pandas as pd
import pickle

# Page title
st.title("Bank Customer Churn Prediction")
st.write("Enter customer information to predict if they will leave the bank")

# Load the model

with open('Bank Churn/churn_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
        
model = model_data['model']
scaler = model_data['scaler']
feature_columns = model_data['feature_columns']


# Input fields
credit_score = st.slider("Credit Score", 300, 900, 600)
age = st.slider("Age", 18, 100, 40)
tenure = st.slider("Tenure (years)", 0, 10, 3)
balance = st.number_input("Balance", min_value=0.0, value=4678.0)
products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active = st.selectbox("Is Active Member", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)
gender = st.selectbox("Gender", ["Female", "Male"])
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])


has_card_value = 1 if has_card == "Yes" else 0
is_active_value = 1 if is_active == "Yes" else 0

# Predict button
if st.button("Predict"):
    # Create input data for prediction
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [products],
        'HasCrCard': [has_card_value],
        'IsActiveMember': [is_active_value],
        'EstimatedSalary': [estimated_salary],
        'Gender': [gender],
        'Geography': [geography]
    })
    
    # Make prediction
    input_data = pd.get_dummies(input_data, drop_first=True)
    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[feature_columns]
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    
    # Display result
    if prediction[0] == 1:
        st.error("Customer will leave the bank")
    else:
        st.success("Customer will stay in the bank")