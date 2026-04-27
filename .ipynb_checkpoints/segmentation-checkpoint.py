import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# App title
st.title("Customer Segmentation App")
st.write("Enter customer details to predict the segment.")

# User inputs
age = st.number_input("Age", min_value=18, max_value=100, value=35)
income = st.number_input("Income", min_value=0, max_value=200000, value=50000)
total_spending = st.number_input("Total Spending (sum of purchases)", min_value=0, max_value=5000, value=1000)
num_web_purchases = st.number_input("Number of Web Purchases", min_value=0, max_value=100, value=10)
num_store_purchases = st.number_input("Number of Store Purchases", min_value=0, max_value=100, value=10)
num_web_visits = st.number_input("Number of Web Visits per Month", min_value=0, max_value=50, value=3)
recency = st.number_input("Recency (days since last purchases)", min_value=0, max_value=365, value=30)

# Create input dataframe
input_data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "Total_Spending": [total_spending],
    "NumWebPurchases": [num_web_purchases],
    "NumStorePurchases": [num_store_purchases],
    "NumWebVisitsMonth": [num_web_visits],
    "Recency": [recency]
})

# Show input data
st.subheader("Customer Data Preview")
st.dataframe(input_data)

# Scale input
input_scaled = scaler.transform(input_data)

# Cluster labels (IMPORTANT ADDITION)
labels = {
    0: "Low Value Customer",
    1: "High Value Customer",
    2: "Frequent Buyer",
    3: "At Risk Customer"
}

# Prediction
if st.button("Predict Segment"):
    cluster = kmeans.predict(input_scaled)[0]

    st.success(f"Predicted Segment: {labels.get(cluster, 'Unknown')}")

    # Extra insights (BONUS)
    st.subheader("Insights & Suggestions")

    if cluster == 1:
        st.write("💰 High Value Customer → Offer premium services & loyalty rewards")
    elif cluster == 2:
        st.write("🛒 Frequent Buyer → Recommend products & cross-sell")
    elif cluster == 3:
        st.write("⚠️ At Risk Customer → Offer discounts or re-engagement campaigns")
    else:
        st.write("📉 Low Value Customer → Improve engagement with offers")