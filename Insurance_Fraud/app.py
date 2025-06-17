import streamlit as st
import numpy as np
import pandas as pd
import pickle


with open('/Users/sahnoontm/Documents/ML/Insurance_Fraud/fraud_model_bundle.pkl', 'rb') as file:
    model_dic = pickle.load(file)

model = model_dic['model']
scaler = model_dic['scaler']
onehot = model_dic['onehot_encoder']

st.title("Insurance Fraud Prediction App")

st.header("Enter Policy and Claim Information")

input_dict = {
    'Days_Policy_Accident_num': st.selectbox("Approximately how many days after the policy started did the accident happen?", [0, 4, 12, 23, 35]),
    'Year': st.selectbox("Year", [1994, 1995, 1996]),
    'AgeOfPolicyHolder': st.selectbox("Age of Policy Holder", ['18 to 20','21 to 25','26 to 30','31 to 35','36 to 40','41 to 50','51 to 65','over 65']),
    'risk_score': st.selectbox("Risk Score", list(range(1, 15))),
    'Age': st.slider("Age", 16, 73, 30),
    'BasePolicy': st.selectbox("Base Policy", ['Collision', 'Liability', 'All Perils']),
    'AccidentArea': st.selectbox("Accident Area", ['Urban', 'Rural']),
    'NumberOfSuppliments': st.selectbox("Number of Supplements", [0, 2, 4, 6]),
    'claim_delay': st.selectbox("Claim Delay", [0, 8, 11, 12, 19, 23, 31, 35]),
    'PastNumberOfClaims_Num': st.selectbox("Past Number of Claims", [0, 1, 3, 5]),
    'AgentType': st.selectbox("Agent Type", ['Internal', 'External']),
    'Fault': st.selectbox("Fault", ['Policy Holder', 'Third Party']),
    'NumberOfCars': st.selectbox("how many car he/she has", [1, 2, 4, 6, 8]),
    'Is_New_Customer': 1 if st.selectbox("Is New Customer or has he changed previous address", ["Yes", "No"]) == "Yes" else 0,
    'Days_Policy_Claim_num': st.selectbox("Approximately how many days after the policy started was the claim filed?", [12, 23, 35]),
    'Sex': st.selectbox("Sex", ['Male', 'Female'])
}
# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# One-hot encoding using fitted encoder (avoid get_dummies)
categorical_cols = onehot.feature_names_in_.tolist()
onehot_encoded = onehot.transform(input_df[categorical_cols])
onehot_df = pd.DataFrame(onehot_encoded, columns=onehot.get_feature_names_out(categorical_cols))

# Drop the original categorical and concatenate
numerical_df = input_df.drop(columns=categorical_cols)
final_input = pd.concat([numerical_df.reset_index(drop=True), onehot_df.reset_index(drop=True)], axis=1)

# Scale the features
scaled_input = scaler.transform(final_input)

# Predict
if st.button("Predict"):
    prediction = model.predict(scaled_input)
    result = 'Fraudulent Claim Detected 🚨' if prediction[0] == 1 else 'Legitimate Claim ✅'
    st.subheader(f"Prediction Result: {result}")
