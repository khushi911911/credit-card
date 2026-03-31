# ================================
# Import Libraries
# ================================
import streamlit as st
import pandas as pd
import joblib

# ================================
# Load Model & Files
# ================================
model = joblib.load("best_fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# ================================
# Page Config
# ================================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Credit Card Fraud Detection System")
st.write(
    "Use the form below to enter transaction details, then click Predict to see whether the transaction is likely fraudulent."
)

with st.sidebar:
    st.header("About")
    st.write(
        "This app loads a pre-trained fraud detection model and scales input values before predicting."
    )
    st.write("- Model type: scikit-learn classifier")
    st.write("- Output: fraud probability and label")
    st.markdown("---")
    st.header("Instructions")
    st.write(
        "Enter the transaction feature values in the form and click the Predict button."
    )

st.subheader("Transaction Details")

input_data = {}

with st.form(key="transaction_form"):
    cols = st.columns(2)
    for idx, feature in enumerate(features):
        col = cols[idx % 2]
        input_data[feature] = col.number_input(
            label=feature,
            value=0.0,
            format="%.4f",
        )

    submit_button = st.form_submit_button("🔍 Predict Transaction")

input_df = pd.DataFrame([input_data])

if submit_button:
    try:
        input_df = input_df[features]
        input_scaled = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=features)

        prediction = model.predict(input_scaled_df)[0]
        prob = model.predict_proba(input_scaled_df)[0][1]

        st.markdown("---")
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("🚨 Fraudulent Transaction Detected!")
        else:
            st.success("✅ Legitimate Transaction")

        st.metric("Fraud Probability", f"{prob:.2%}")
        st.write("### Input Summary")
        st.dataframe(input_df.T.rename(columns={0: 'Value'}), use_container_width=True)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
