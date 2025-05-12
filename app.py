import streamlit as st
import numpy as np
import pandas as pd

import joblib  # 👈 replaced pickle with joblib

# Load saved objects
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')
model = joblib.load('kmeans_model.pkl')

# Cluster interpretation mapping
cluster_profiles = {
    0: {
        "Group": "Group 0",
        "Interpretation": "Occasional / trial users",
        "Action": "Offer welcome-back discounts or trial-to-subscription upgrades."
    },
    1: {
        "Group": "Group 1",
        "Interpretation": "Loyal, high-value, high-LTV customers",
        "Action": "Retain with loyalty rewards, exclusive offers, or upsells."
    },
    2: {
        "Group": "Group 2",
        "Interpretation": "Bulk single-time buyers",
        "Action": "Convert to subscription with discounts or bundling offers."
    },
    3: {
        "Group": "Group 3",
        "Interpretation": "Loyal but low-volume / cautious customers",
        "Action": "Encourage higher volume purchases with tiered pricing or perks."
    }
}

# Streamlit UI
st.set_page_config(page_title="Customer Cluster Predictor", layout="centered")
st.title("🧠 Customer Cluster Prediction App")
st.markdown("Fill in the customer data below:")

transaction_cost = st.number_input("Transaction Cost", min_value=200, max_value=100000, step=100)
unit_pack_size = st.selectbox("Unit Pack Size", [50, 70, 100, 120, 200, 500, 1000])
purchase_type_label = st.selectbox("Purchase Type", ["Unit", "Carton"])

input_data = pd.DataFrame({
    "Transaction Cost": [transaction_cost],
    "Unit Pack Size": [unit_pack_size],
    "Purchase Type": [purchase_type_label]
},columns=["Transaction Cost", "Unit Pack Size", "Purchase Type"])




if st.button("Predict Cluster"):
    # Transform Purchase Type using encoder
    encoded_data = encoder.transform(input_data)
    scaled_data = scaler.transform(encoded_data)

    # Predict
    cluster = model.predict(scaled_data)[0]  # get the scalar
    profile = cluster_profiles.get(cluster)

    # Display results
    st.success(f"✅ Prediction: {profile['Group']}")
    st.write(f"**Interpretation:** {profile['Interpretation']}")
    st.info(f"💡 **Recommended Business Action:** {profile['Action']}")