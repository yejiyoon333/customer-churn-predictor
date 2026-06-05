# Customer Churn Predictor

A machine learning web app that predicts customer churn using tabular customer data.

## Tech Stack

- Python
- pandas
- scikit-learn
- Streamlit
- joblib

## Project Structure

```text
customer-churn-predictor/
  app.py
  train_model.py
  requirements.txt
  README.md
  data/
    Telco-Customer-Churn.csv
  model/
    churn_model.pkl
    metrics.json


Dataset

This project uses the Telco Customer Churn dataset.

Download the dataset:

mkdir -p data

curl -L "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" -o data/Telco-Customer-Churn.csv


Setup

Create a virtual environment:

python3.11 -m venv venv
source venv/bin/activate
Install dependencies:

pip install -r requirements.txt


Train Model
python train_model.py

This creates:

model/churn_model.pkl
model/metrics.json
Run App
python -m streamlit run app.py
Features
Load real customer churn data
Train a baseline churn classifier
Evaluate model performance
Predict churn probability
Display top factors affecting each prediction
Model Output

The app shows:

Prediction: Churn or No Churn
Churn probability
Model evaluation metrics
Top feature impacts
