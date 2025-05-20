import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the saved model and encoders
@st.cache_resource
def load_model():
    with open('Car-price-Reggression/car_price_model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

data = load_model()
model = data['model']
one_hot_encoder = data['one_hot_encoder']
label_encoder = data['label_encoder']
one_hot_cols = data['one_hot_cols']
label_col = data['label_col']
num_cols = data['num_cols']

# Streamlit app
st.title('Car Price Prediction')

# Input form
with st.form("prediction_form"):
    st.header("Enter Car Details")
    
    # Numerical inputs
    prod_year = st.number_input('Production Year', min_value=1990, max_value=2023, value=2018)
    mileage = st.number_input('Mileage', min_value=0, value=50000)
    airbags = st.number_input('Number of Airbags', min_value=0, max_value=20, value=6)
    
    # Categorical inputs
    manufacturer = st.selectbox('Manufacturer', options=sorted(one_hot_encoder.categories_[2]))
    category = st.selectbox('Category', options=sorted(one_hot_encoder.categories_[3]))
    fuel_type = st.selectbox('Fuel Type', options=sorted(one_hot_encoder.categories_[0]))
    gear_box_type = st.selectbox('Gear Box Type', options=sorted(one_hot_encoder.categories_[1]))
    leather_interior = st.radio('Leather Interior', options=label_encoder.classes_)
    
    submitted = st.form_submit_button("Predict Price")

# When form is submitted
if submitted:
    # Create input dataframe
    input_data = pd.DataFrame({
        'Manufacturer': [manufacturer],
        'Prod. year': [prod_year],
        'Category': [category],
        'Leather interior': [leather_interior],
        'Fuel type': [fuel_type],
        'Mileage': [mileage],
        'Gear box type': [gear_box_type],
        'Airbags': [airbags]
    })
    
    # Preprocess the input
    # One-hot encode
    one_hot_encoded = one_hot_encoder.transform(input_data[one_hot_cols])
    # Label encode
    label_encoded = label_encoder.transform(input_data[label_col]).reshape(-1, 1)
    # Numerical features
    numerical_features = input_data[num_cols].values
    
    # Combine features
    X_new = np.concatenate([numerical_features, one_hot_encoded, label_encoded], axis=1)
    
    # Make prediction
    predicted_price = model.predict(X_new)[0]
    
    # Display result
    st.success(f"Predicted Car Price: ${predicted_price:,.2f}")