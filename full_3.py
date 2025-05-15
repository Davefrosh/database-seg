import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine

# Load saved ML components
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")
model = joblib.load("kmeans_model.pkl")

# Connect to local SQLite database
engine = create_engine("sqlite:///customers.db")

# Cluster interpretation dictionary
cluster_profiles = {
    0: {
        "group_name": "Group (A) Diverse Mid-Value Shoppers",
        "interpretation": "This group represents customers with varied purchasing habits. They are generally mid-value spenders but can occasionally make larger purchases. Their preference for Noodles and Noodles is notable.",
        "reasoning_intro": "This customer likely belongs to this group because their purchasing patterns show:",
        "recommended_actions": [
            "Targeted Promotions for Noodles & Noodles: Reinforce their preference with special offers, bundles, or loyalty rewards on these specific products/categories.",
            "Encourage Higher Spends: Since they occasionally make high-value purchases, introduce premium versions or larger pack bundles of their preferred Noodles or other items in Noodles to potentially increase their average transaction value.",
            "Cross-selling Opportunities: Based on their interest in Noodles, introduce them to related or complementary products from other categories via targeted marketing."
        ]
    },
    1: {
        "group_name": "(Group B) Budget-Conscious, Small Pack Buyers (Ball Foods Focus)",
        "interpretation": "This is the largest segment, composed of budget-conscious shoppers who prefer smaller, more affordable pack sizes and are loyal to Garri and ball foods. They likely represent frequent, lower-value transactions.",
        "reasoning_intro": "This customer likely belongs to this group because their purchasing patterns show:",
        "recommended_actions": [
            "Value Propositions: Emphasize value, affordability, and smaller pack convenience in marketing communications for garri and ball foods.",
            "Loyalty Programs for Frequent Purchases: Implement loyalty programs that reward frequent, smaller purchases to retain this large customer base.",
            "Introduce Entry-Level Products: If not already available, consider introducing more entry-level or trial-size products within ball foods to cater to their preference for small packs and potentially attract new similar customers.",
            "Volume Discounts on Small Packs: Offer slight discounts for buying multiple small packs to gently encourage increased basket size without pushing them towards large, intimidating pack sizes."
        ]
    },
    2: {
        "group_name": "(Group C) High-Value, Bulk Buyers (Ball foods Focus)",
        "interpretation": "These customers are high-value shoppers who prefer to buy in bulk, focusing on garri within ball foods. They likely make less frequent but larger purchases.",
        "reasoning_intro": "This customer likely belongs to this group because their purchasing patterns show:",
        "recommended_actions": [
            "Bulk Purchase Incentives: Offer discounts, special deals, or loyalty points for bulk purchases of garri and other items in ball foods.",
            "Subscription Services: For staple items in ball foods that they buy in bulk, consider offering subscription services for regular delivery, ensuring their continued loyalty and predictable revenue.",
            "Highlight Cost Savings of Bulk: Marketing communications should emphasize the cost-effectiveness and convenience of buying larger pack sizes.",
            "Premium Offerings within ball foods: Since they are high-spenders, explore opportunities for premium or enhanced versions of garri or related items in ball foods."
        ]
    },
    3: {
        "group_name": "(Group D) High-Value, Specific Purchase Type Shoppers",
        "interpretation": "This is the smallest but highest-spending segment on average. Their defining characteristic is their exclusive use of Carton. They buy moderate pack sizes and are mainly interested in ball foods, with a primary preference for garri.",
        "reasoning_intro": "This customer likely belongs to this group because their purchasing patterns show:",
        "recommended_actions": [
            "Optimize for Carton Purchase: Ensure a seamless and rewarding experience for carton. Understand the nuances of this purchase type and cater to it specifically.",
            "Exclusive Offers for carton Users: Provide exclusive deals, early access to new products, or enhanced services for customers using carton to acknowledge their high value.",
            "Personalized Recommendations: Given their high value and specific purchase behavior, leverage data to offer highly personalized recommendations within ball foods, potentially introducing them to other products they might like (beyond garri).",
            "Premium Loyalty Tier: Consider a premium tier in your loyalty program specifically for these high-spending carton purchase users, offering unique benefits."
        ]
    }
}

# Streamlit UI
st.set_page_config(page_title="Customer Segmentation Application", layout="wide")
st.title("Customer Segmentation")

# Input fields
first_name = st.text_input("First Name")
last_name = st.text_input("Last Name")

if st.button("Predict"):
    with engine.connect() as conn:
        query = f"""
            SELECT * FROM customers
            WHERE "First Name" = '{first_name}' AND "Last Name" = '{last_name}'
        """
        df = pd.read_sql(query, conn)

    if df.empty:
        st.warning("Customer not found in database.")
    else:
        row = df.iloc[0]

        # Display customer profile
        st.subheader("📄 Customer Profile")
        keep_cols = ["Customer ID", "First Name", "Last Name", "Age", "Gender",
                     "State", "Household Size"]
        st.json({col: row[col] for col in keep_cols if col in row})

        try:
            # Prepare input for model
            X = pd.DataFrame([{
                "Transaction Cost": row["Transaction Cost"],
                "Unit Pack Size": row["Unit Pack Size"],
                "Purchase Type": row["Purchase Type"],
                "Product Name": row["Product Name"],
                "Category": row["Category"]
            }])

            X = encoder.transform(X)
            X_scaled = scaler.transform(X)
            cluster = model.predict(X_scaled)[0]
            profile = cluster_profiles[cluster]

            # Extract dynamic values
            spend = int(row["Transaction Cost"])
            pack_size = int(row["Unit Pack Size"])
            purchase_type = row["Purchase Type"]
            product_name = row["Product Name"]
            category = row["Category"]

            # Generate dynamic reasoning
            dynamic_reasoning = (
                f"This customer spends around **₦{spend:,}**, prefers **{pack_size}g** packs, "
                f"typically buys by **{purchase_type}**, and shows strong interest in **{product_name}** "
                f"within the **{category}** category."
            )

            # Output prediction and recommendations
            st.subheader(f"{profile['group_name']}")
            st.markdown(f"**Interpretation:** {profile['interpretation']}")
            st.markdown(f"> {dynamic_reasoning}")

            st.markdown("**Recommended Business Actions to Upsell/Retain:**")
            for action in profile["recommended_actions"]:
                st.info(f"💡 {action}")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
else:
    st.info("Please enter a customer's first and last name to get their segmentation profile.")
