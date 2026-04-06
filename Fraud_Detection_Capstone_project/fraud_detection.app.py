
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

@st.cache_resource
def load_saved_model():
    model_tuple = joblib.load("Save_model.plk")
    model = model_tuple[0]
    return model

model = load_saved_model()

st.set_page_config(page_title="Fraud Detection App", page_icon="💳", layout="centered")

st.title("💳 Fraud Detection System")
st.write("Enter the transaction details below and the model will predict whether it is **Fraud (1)** or **Not Fraud (0)**.")
st.markdown("---")

st.subheader("🔍 Enter Transaction Details")

type_options = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
type_choice = st.selectbox("Transaction Type", type_options)

step = st.number_input("Step (Time Frame)", min_value=0, format="%d")
amount = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")

oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, format="%.2f")
newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0, format="%.2f")

oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0, format="%.2f")
newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0, format="%.2f")

origDiff = oldbalanceOrg - newbalanceOrig
destDiff = oldbalanceDest - newbalanceDest

st.write(f"🔸 **origDiff** = {origDiff}")
st.write(f"🔸 **destDiff** = {destDiff}")

def build_input():
    le = LabelEncoder()
    le.fit(sorted(type_options))
    encoded_type = le.transform([type_choice])[0]

    row = pd.DataFrame([{
        "step": step,
        "type": encoded_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "origDiff": origDiff,
        "destDiff": destDiff
    }])
    return row

if st.button("🔮 Predict Fraud"):
    input_df = build_input()
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("### 📌 Prediction Result")

    if prediction == 1:
        st.error(f"🚨 Fraud Detected!\nProbability: **{probability*100:.2f}%**")
    else:
        st.success(f"✅ Not Fraud\nProbability: **{probability*100:.2f}%**")

st.markdown("---")
st.caption("Developed by Ashish • Powered by ML & Streamlit")
