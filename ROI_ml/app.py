import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open("/Users/sahnoontm/Documents/ML/ROI/ROI_model.pkl", "rb") as file:
    model_dict = pickle.load(file)

encoder = model_dict['onehot']
scaler = model_dict['scaler']
model = model_dict['model']

st.title("📈 ROI Campaign Sales Predictor")

# User inputs
platform = st.selectbox("Platform", ['Instagram', 'YouTube', 'Twitter'], help="Where is the campaign run?")
influencer_category = st.selectbox("Influencer Category", ['Fashion', 'Tech', 'Fitness'],help="Type of influencer running the campaign")
campaign_type = st.selectbox("Campaign Type", ['Giveaway', 'Product Launch', 'Review'],help="Type of promotional campaign")
campaign_month = st.selectbox("Campaign Month", [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
],help="Which month is the campaign running?")
engagements = st.number_input("Number of Engagements", min_value=100,help="Total interactions like likes, comments, and shares")
estimated_reach = st.number_input("Estimated Reach", min_value=1000, help="How many unique users are expected to see the campaign")
campaign_duration_days = st.number_input("Campaign Duration (Days)", min_value=1,help="How many days will the campaign run")
engagement_rate = st.number_input("Engagement Rate (%)", min_value=0.0, max_value=100.0)
avg_daily_engagements = st.number_input("Average Daily Engagements", min_value=0,help="Average number of engagements per day")
avg_daily_sales = st.number_input("Average Daily Sales", min_value=0,help="How many products are sold per day on average")
influencer_score = st.slider("Influencer Score (0-100)", min_value=0, max_value=100,help="A score representing influencer effectiveness")
is_weekend_campaign = st.radio("Is it a Weekend Campaign?", ['Yes', 'No'])
high_intensity_campaign = st.radio("High Intensity Campaign?", ['Yes', 'No'])



if st.button("🎯 Predict Product Sales"):
    # Create input DataFrame
    input_data = pd.DataFrame({
     'platform': [platform],
    'influencer_category': [influencer_category],
    'campaign_type': [campaign_type],
    'campaign_month': [campaign_month],
    'engagements': [engagements],
    'estimated_reach': [estimated_reach],
    'campaign_duration_days': [campaign_duration_days],
    'engagement_rate': [engagement_rate],
    'avg_daily_engagements': [avg_daily_engagements],
    'avg_daily_sales': [avg_daily_sales],
    'influencer_score': [influencer_score],
    'is_weekend_campaign': [1 if is_weekend_campaign == 'Yes' else 0],
    'high_intensity_campaign': [1 if high_intensity_campaign == 'Yes' else 0]
    })

    # Split categorical and numerical
    categorical_cols = ['platform', 'influencer_category', 'campaign_type', 'campaign_month']
    numerical_cols = [col for col in input_data.columns if col not in categorical_cols]

    # Encode categorical
    X_cat = encoder.transform(input_data[categorical_cols])
    X_num = input_data[numerical_cols].values

    # Combine and scale
    X_full = np.hstack([X_cat, X_num])
    X_scaled = scaler.transform(X_full)

    # Predict
    prediction = model.predict(X_scaled)[0]
    
    st.success(f"📊 Predicted Product Sales: **{int(prediction)} units**")
