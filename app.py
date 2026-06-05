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

def clean_feature_name(feature_name):
    return (
        feature_name
        .replace("num__", "")
        .replace("cat__", "")
        .replace("_", " = ")
    )

def get_feature_impact(model, input_data):
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    transformed_input = preprocessor.transform(input_data)

    if hasattr(transformed_input, "toarray"):
        transformed_input = transformed_input.toarray()

    feature_names = preprocessor.get_feature_names_out()

    if hasattr(classifier, "coef_"):
        importance_values = classifier.coef_[0]
        impacts = transformed_input[0] * importance_values
    elif hasattr(classifier, "feature_importances_"):
        impacts = classifier.feature_importances_
    else:
        raise ValueError("This classifier does not support explainability.")

    impact_df = pd.DataFrame(
        {
            "feature": feature_names,
            "impact": impacts
        }
    )

    impact_df["abs_impact"] = impact_df["impact"].abs()
    impact_df["feature"] = impact_df["feature"].apply(clean_feature_name)

    return impact_df.sort_values("abs_impact", ascending=False).head(10)


model = load_model()
metrics = load_metrics()

st.title("Customer Churn Predictor")

st.write("Enter customer information to predict churn risk.")

with st.expander("Model Evaluation Metrics"):
    st.write(f"Best Model: {metrics['best_model']}")

    metrics_df = pd.DataFrame(metrics["models"]).T
    metrics_df = metrics_df[["accuracy", "precision", "recall", "f1", "roc_auc"]]

    st.dataframe(metrics_df, use_container_width=True)

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
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
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

    st.subheader("Top Factors Affecting This Prediction")

    impact_df = get_feature_impact(model, input_data)

    st.dataframe(
        impact_df[["feature", "impact"]],
        use_container_width=True
    )

    st.write("The table shows the most important features used by the selected model.")

st.caption("Built with Python, scikit-learn, and Streamlit.")
