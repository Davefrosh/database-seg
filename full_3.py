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
        "group_name": "Group (A) Noodles & Pasta Enthusiasts",
        "interpretation": "This group consists of customers who are primary consumers of Noodles and Pasta products. They mostly buy individual units and show variability in their spending and pack size choices, possibly driven by promotions or specific needs.",
        "key_characteristics_summary": "Moderate average spending with occasional high-value purchases, mixed preference for pack sizes, predominantly Purchase Type 'Unit', strong preference for 'Noodles' and 'Pasta'.",
        "recommended_actions": [
            "Noodles & Pasta Promotions: Run targeted promotions, bundle offers (e.g., Noodles + Pasta combo), or loyalty rewards specifically for Noodles and Pasta products to deepen their engagement.",
            "Introduce Variety/Premium Options: For Noodles and Pasta, introduce new flavors, healthier options, or premium versions to encourage trial and potentially increase their average transaction value.",
            "Cross-Sell Complementary Items: Suggest items that pair well with Noodles and Pasta, such as sauces, spices, or ready-to-eat side dishes.",
            "Family Pack Considerations: Test promotions on slightly larger or family-sized packs of their favorite Noodles/Pasta."
        ]
    },
    1: {
        "group_name": "Group (B) Ball Foods Regulars - Smaller Households",
        "interpretation": "These are consistent buyers of Ball Foods, primarily Garri and Poundo Yam, purchasing individual units. Their smaller household size likely influences their consumption patterns.",
        "key_characteristics_summary": "Moderate average spending, range of pack sizes (not exclusively small), exclusively Purchase Type 'Unit', strong loyalty to 'Ball Foods' (Garri, Poundo Yam), smaller households.",
        "recommended_actions": [
            "Reinforce Ball Foods Loyalty: Offer loyalty rewards or exclusive deals on Garri and Poundo Yam.",
            "Convenience Packs for Smaller Households: Highlight or introduce pack sizes suitable for 1-2 person households for their preferred Ball Foods.",
            "Promote Quality and Consistency: Emphasize the quality, taste, and consistent availability of these Ball Foods products.",
            "Trial of Related Products: Gently introduce other complementary products that might appeal to Ball Foods consumers."
        ]
    },
    2: {
        "group_name": "Group (C) Ball Foods Staples - Larger Households",
        "interpretation": "This large segment is also highly loyal to Ball Foods (Garri and Poundo Yam), purchasing individual units. The key differentiator is their much larger household size, suggesting higher overall consumption volume.",
        "key_characteristics_summary": "Moderate average spending, range of pack sizes (slightly larger on average), exclusively Purchase Type 'Unit', strong loyalty to 'Ball Foods' (Garri, Poundo Yam), larger households.",
        "recommended_actions": [
            "Family-Size Value Offers: Target this group with promotions on larger unit packs or multi-unit bundles of Garri and Poundo Yam.",
            "Subscription for Staples: Explore subscription models for regular delivery of their preferred Ball Foods.",
            "Highlight Bulk Unit Savings: Encourage purchase of multiple units by highlighting savings.",
            "Engage with Family-Oriented Content: Marketing communications could resonate with family values and meal planning for larger groups."
        ]
    },
    3: {
        "group_name": "Group (D) High-Value Carton Buyers",
        "interpretation": "This is the highest-spending segment, defined by their preference for buying in bulk via Cartons. They primarily stock up on Ball Foods (especially Garri) but also include Noodles in their bulk purchases.",
        "key_characteristics_summary": "Highest average spending, exclusively Purchase Type 'Carton', primarily 'Ball Foods' (Garri) and also 'Noodles' in bulk, average household size.",
        "recommended_actions": [
            "Carton-Specific Loyalty Program: Implement a loyalty program that specifically rewards carton purchases.",
            "Exclusive Carton Deals & Early Access: Provide this segment with exclusive discounts on cartons or early access to new products in carton format.",
            "Expand Carton Offerings: Expand the range of products available in carton format, especially for popular Garri varieties and Noodles.",
            "Understand Motivation for Carton Purchase: Conduct surveys or gather feedback to understand reasons for carton purchases (personal consumption, group buying, resale)."
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
                "Category": row["Category"],
                "Household Size": row["Household Size"],
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
