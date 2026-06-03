import streamlit as st

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)

st.title("Customer Churn Predictor")

st.write("This app predicts whether a customer may churn.")

st.subheader("Customer Information")

tenure = st.number_input("Tenure", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer",
        "Credit card"
    ]
)

if st.button("Predict"):
    st.info("Model prediction will be added later.")

st.divider()

st.caption("Built with Python, scikit-learn, SHAP, and Streamlit.")
