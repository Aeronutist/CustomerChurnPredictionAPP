#app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load model, scaler, and feature columns
model = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

st.title("📞 Customer Churn Prediction App")
st.markdown("Enter customer information below to predict if they are likely to churn.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Has Partner", ["Yes", "No"])
Dependents = st.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (in months)", 0, 72, 1)
PhoneService = st.selectbox("Has Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, step=1.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, step=1.0)

# Mapping binary features
binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
data = {
    'gender': binary_map[gender],
    'SeniorCitizen': SeniorCitizen,
    'Partner': binary_map[Partner],
    'Dependents': binary_map[Dependents],
    'tenure': tenure,
    'PhoneService': binary_map[PhoneService],
    'PaperlessBilling': binary_map[PaperlessBilling],
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges
}

# One-hot encoded categorical features
multi_values = {
    'MultipleLines_No phone service': 1 if MultipleLines == 'No phone service' else 0,
    'MultipleLines_Yes': 1 if MultipleLines == 'Yes' else 0,
    'InternetService_Fiber optic': 1 if InternetService == 'Fiber optic' else 0,
    'InternetService_No': 1 if InternetService == 'No' else 0,
    'OnlineSecurity_Yes': 1 if OnlineSecurity == 'Yes' else 0,
    'OnlineSecurity_No internet service': 1 if OnlineSecurity == 'No internet service' else 0,
    'OnlineBackup_Yes': 1 if OnlineBackup == 'Yes' else 0,
    'OnlineBackup_No internet service': 1 if OnlineBackup == 'No internet service' else 0,
    'DeviceProtection_Yes': 1 if DeviceProtection == 'Yes' else 0,
    'DeviceProtection_No internet service': 1 if DeviceProtection == 'No internet service' else 0,
    'TechSupport_Yes': 1 if TechSupport == 'Yes' else 0,
    'TechSupport_No internet service': 1 if TechSupport == 'No internet service' else 0,
    'StreamingTV_Yes': 1 if StreamingTV == 'Yes' else 0,
    'StreamingTV_No internet service': 1 if StreamingTV == 'No internet service' else 0,
    'StreamingMovies_Yes': 1 if StreamingMovies == 'Yes' else 0,
    'StreamingMovies_No internet service': 1 if StreamingMovies == 'No internet service' else 0,
    'Contract_One year': 1 if Contract == 'One year' else 0,
    'Contract_Two year': 1 if Contract == 'Two year' else 0,
    'PaymentMethod_Electronic check': 1 if PaymentMethod == 'Electronic check' else 0,
    'PaymentMethod_Mailed check': 1 if PaymentMethod == 'Mailed check' else 0,
    'PaymentMethod_Credit card (automatic)': 1 if PaymentMethod == 'Credit card (automatic)' else 0,
    'PaymentMethod_Bank transfer (automatic)': 1 if PaymentMethod == 'Bank transfer (automatic)' else 0
}

# Combine and align input
data.update(multi_values)
final_input = pd.DataFrame([data])

# Add missing columns
for col in feature_columns:
    if col not in final_input.columns:
        final_input[col] = 0

# Reorder to match training
final_input = final_input[feature_columns]

# Scale and predict
scaled_input = scaler.transform(final_input)

if st.button("Predict Churn"):
    pred = model.predict(scaled_input)
    prob = model.predict_proba(scaled_input)[0][1]
    if pred[0] == 1:
        st.error(f"⚠️ Customer likely to churn. Probability: {prob:.2f}")
    else:
        st.success(f"✅ Customer likely to stay. Probability: {1 - prob:.2f}")

