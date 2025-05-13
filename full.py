import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved ML objects
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')
model = joblib.load('kmeans_model.pkl')

# Cluster interpretation
cluster_profiles = {
    0: {"Group": "Group 0", "Interpretation": "Occasional / trial users", "Action": "Offer welcome-back discounts or trial-to-subscription upgrades."},
    1: {"Group": "Group 1", "Interpretation": "Loyal, high-value, high-LTV customers", "Action": "Retain with loyalty rewards, exclusive offers, or upsells."},
    2: {"Group": "Group 2", "Interpretation": "Bulk single-time buyers", "Action": "Convert to subscription with discounts or bundling offers."},
    3: {"Group": "Group 3", "Interpretation": "Loyal but low-volume / cautious customers", "Action": "Encourage higher volume purchases with tiered pricing or perks."}
}

# UI setup
st.set_page_config(page_title="📊 ML Customer Prediction Tool", layout="centered")
st.title("🔍 Customer Group Predictor from Uploaded File")

# File upload
uploaded_file = st.file_uploader("Upload Customer CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'Customer ID' not in df.columns:
        st.error("❌ 'Client ID' column not found. Please upload a valid CSV.")
    else:
        # Select a client
        selected_client = st.selectbox("Select Client ID", df["Customer ID"].unique())

        # Get and show full row
        client_row = df[df["Customer ID"] == selected_client].iloc[0]
        st.subheader("🧾 Full Customer Profile")
        st.json(client_row.to_dict())

        # Extract only required columns for prediction
        try:
           features = pd.DataFrame([{"Transaction Cost": client_row["Transaction Cost"],"Unit Pack Size": client_row["Unit Pack Size"],"Purchase Type": client_row["Purchase Type"]}])

            
           features = encoder.transform(features)

           # Now safe to scale and predict
           X_scaled = scaler.transform(features)

            # Predict
           if st.button("Predict Cluster"):
                cluster = model.predict(X_scaled)[0]
                profile = cluster_profiles.get(cluster)

                st.success(f"✅ Prediction: {profile['Group']}")
                st.write(f"**Interpretation:** {profile['Interpretation']}")
                st.info(f"💡 **Recommended Business Action:** {profile['Action']}")

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
