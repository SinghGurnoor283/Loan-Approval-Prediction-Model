import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page title
st.title("🏦 Loan Approval Prediction App")

# Sidebar input
st.sidebar.header("📋 Applicant Information")

# User inputs
no_of_dependents = st.sidebar.selectbox("Number of Dependents", [0, 1, 2, 3])
education = st.sidebar.radio("Education", ["Not Graduate", "Graduate"])
self_employed = st.sidebar.radio("Self Employed", ["No", "Yes"])
income_annum = st.sidebar.number_input("Annual Income (₹)", value=1500000)
loan_amount = st.sidebar.number_input("Loan Amount Requested (₹)", value=200000)
loan_term_years = st.sidebar.slider("Loan Term (Years)", 1, 30, value=5)
cibil_score = st.sidebar.slider("CIBIL Score", 300, 900, value=750)
residential_assets = st.sidebar.number_input("Residential Assets (₹)", value=500000)
commercial_assets = st.sidebar.number_input("Commercial Assets (₹)", value=300000)
luxury_assets = st.sidebar.number_input("Luxury Assets (₹)", value=200000)
bank_assets = st.sidebar.number_input("Bank Assets (₹)", value=500000)

# Derived features
total_assets = residential_assets + commercial_assets + luxury_assets + bank_assets
income_loan_ratio = income_annum / (loan_amount + 1)
loan_term_months = loan_term_years * 12

# Encode categorical
education_encoded = 1 if education == "Graduate" else 0
self_employed_encoded = 1 if self_employed == "Yes" else 0

# Prepare input DataFrame
input_data = pd.DataFrame([{
    'no_of_dependents': no_of_dependents,
    'education': education_encoded,
    'self_employed': self_employed_encoded,
    'income_annum': income_annum,
    'loan_amount': loan_amount,
    'loan_term': loan_term_months,
    'cibil_score': cibil_score,
    'residential_assets_value': residential_assets,
    'commercial_assets_value': commercial_assets,
    'luxury_assets_value': luxury_assets,
    'bank_asset_value': bank_assets,
    'total_assets': total_assets,
    'income_loan_ratio': income_loan_ratio,
    'loan_term_years': loan_term_years
}])

input_data.columns = input_data.columns.str.strip()

# Scale numerical columns
numerical_cols = [
    'income_annum', 'loan_amount', 'cibil_score',
    'residential_assets_value', 'commercial_assets_value',
    'luxury_assets_value', 'bank_asset_value',
    'total_assets', 'income_loan_ratio', 'loan_term_years'
]
input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

# Predict
if st.button("🔮 Predict Loan Status"):
    prediction = model.predict(input_data)[0]
    result = "✅ Approved" if prediction == 0 else "❌ Rejected"
    st.subheader(f"🎯 Prediction Result: {result}")

    # Show model accuracy here (you can update this value manually if retrained)
    st.markdown("**🔍 Model Accuracy:** 99.8% (Random Forest Classifier)")

# Divider
st.markdown("---")

# Optional feature importance section
with st.expander("📊 Show Feature Importance"):
    importances = model.feature_importances_
    features = model.feature_names_in_
    feat_series = pd.Series(importances, index=features).sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    feat_series.plot(kind='barh', ax=ax, color='skyblue')
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("💡 Tip: This prediction is based on statistical patterns learned from the training dataset. Always consult a financial expert for real-world decisions.")
