import streamlit as st
import pandas as pd
import pickle

with open('/Users/sahnoontm/Documents/ML/ROI/ROI_cluster_model.pkl', 'rb') as file:
    model = pickle.load(file)
encoder = model['onehot']
scaler = model['scaler']
kmeans = model['kmeans']
trained_columns = model['columns']

st.title("📊 Best Platform Recommendation")

st.markdown("### 📝 Enter Campaign Details")

campaign_type = st.selectbox("Campaign Type", ['Paid Partnership', 'Giveaway', 'Review', 'Brand Collaboration'])
campaign_month = st.selectbox("Campaign Month", ['January', 'February', 'March', 'April', 'May', 'June',
                                                  'July', 'August', 'September', 'October', 'November', 'December'])
campaign_duration_days = st.number_input("Campaign Duration (days)", min_value=1, max_value=100)
influencer_category = st.selectbox("Influencer Category", ['Fitness', 'Food', 'Travel', 'Fashion', 'Tech', 'Other'])
avg_daily_sales = st.number_input("Expected Avg Daily Sales", min_value=1, max_value=100000)
engagements = st.number_input("Expected Total Engagements", min_value=1, max_value=1000000)

if st.button("Predict Best Platform"):
    input_df = pd.DataFrame([{
        'campaign_type': campaign_type,
        'campaign_month': campaign_month,
        'campaign_duration_days': campaign_duration_days,
        'influencer_category': influencer_category,
        'avg_daily_sales': avg_daily_sales,
        'engagements': engagements
    }])

    cat_cols = ['campaign_type', 'campaign_month', 'influencer_category']
    num_cols = ['campaign_duration_days', 'avg_daily_sales', 'engagements']

    encoded_input = encoder.transform(input_df[cat_cols])
    encoded_df = pd.DataFrame(encoded_input,
                              columns=encoder.get_feature_names_out(cat_cols),
                              index=input_df.index)

    scaled_nums = scaler.transform(input_df[num_cols])
    scaled_df = pd.DataFrame(scaled_nums, columns=num_cols, index=input_df.index)

    final_input = pd.concat([scaled_df, encoded_df], axis=1)

    final_input = final_input.reindex(columns=trained_columns, fill_value=0)
    cluster_label = kmeans.predict(final_input)[0]


    cluster_to_platform = {
        0: "Instagram",
        1: "YouTube",
        2: "TikTok",
        3: "Facebook"
    }

    best_platform = cluster_to_platform.get(cluster_label, "Unknown")
    st.success(f"✅ Predicted Cluster: {cluster_label}")
    st.markdown(f"**💡 Recommended Platform:** {best_platform}")
