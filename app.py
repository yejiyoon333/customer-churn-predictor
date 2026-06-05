import json

import pandas as pd
import joblib
import streamlit as st


st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)


@st.cache_resource
def load_model():
    return joblib.load("model/churn_model.pkl")


@st.cache_data
def load_metrics():
    with open("model/metrics.json", "r") as file:
        return json.load(file)


model = load_model()
metrics = load_metrics()

st.title("Customer Churn Predictor")

st.write("Enter customer information to predict churn risk.")

with st.expander("Model Evaluation Metrics"):
    st.write(f"Accuracy: {metrics['accuracy']:.4f}")
    st.write(f"Precision: {metrics['precision']:.4f}")
    st.write(f"Recall: {metrics['recall']:.4f}")
    st.write(f"F1: {metrics['f1']:.4f}")
    st.write(f"ROC-AUC: {metrics['roc_auc']:.4f}")

st.subheader("Customer Information")

gender = st.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure", min_value=0, max_value=100, value=12)

phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])

internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

input_data = pd.DataFrame([
    {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }
])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    churn_probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Prediction: Churn")
    else:
        st.success("Prediction: No Churn")

    st.write(f"Churn Probability: {churn_probability:.2%}")

st.caption("Built with Python, scikit-learn, and Streamlit.")
