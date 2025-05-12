from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd

# Load model, scaler, encoder
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

# Create FastAPI app
app = FastAPI(title="Customer Clustering API", version="1.1")

# Enable CORS for frontend access (e.g., from Lovable)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cluster profiles
cluster_profiles = {
    0: {
        "group": "Group 0",
        "interpretation": "Occasional / trial users",
        "action": "Offer welcome-back discounts or trial-to-subscription upgrades."
    },
    1: {
        "group": "Group 1",
        "interpretation": "Loyal, high-value, high-LTV customers",
        "action": "Retain with loyalty rewards, exclusive offers, or upsells."
    },
    2: {
        "group": "Group 2",
        "interpretation": "Bulk single-time buyers",
        "action": "Convert to subscription with discounts or bundling offers."
    },
    3: {
        "group": "Group 3",
        "interpretation": "Loyal but low-volume / cautious customers",
        "action": "Encourage higher volume purchases with tiered pricing or perks."
    }
}

@app.get("/")
def read_root():
    return {"message": "Customer Clustering API is running 🎯"}

@app.post("/predict")
def predict_cluster(
    transaction_cost: float = Query(..., description="Transaction amount"),
    unit_pack_size: int = Query(..., description="Unit pack size"),
    purchase_type: str = Query(..., description="Purchase type (e.g., 'Unit' or 'Carton')")
):
    try:
        input_data = pd.DataFrame({"Transaction Cost": [transaction_cost],"Unit Pack Size": [unit_pack_size],"Purchase Type": [purchase_type]},columns=["Transaction Cost", "Unit Pack Size", "Purchase Type"])
        encoded_data = encoder.transform(input_data)
        scaled_data = scaler.transform(encoded_data)
        cluster = model.predict(scaled_data)[0]

        profile = cluster_profiles.get(cluster)

        return {
            "cluster": int(cluster),
            "group": profile["group"],
            "interpretation": profile["interpretation"],
            "recommended_action": profile["action"]
        }

    except Exception as e:
        return {"error": str(e)}
