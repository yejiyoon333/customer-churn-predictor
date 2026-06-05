import json

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }


def main():
    data_path = "data/Telco-Customer-Churn.csv"

    df = pd.read_csv(data_path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    classifiers = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=42
        )
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    all_metrics = {}
    trained_models = {}

    for model_name, classifier in classifiers.items():
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier)
            ]
        )

        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        all_metrics[model_name] = metrics
        trained_models[model_name] = model

        print(f"\nModel: {model_name}")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

    best_model_name = max(
        all_metrics,
        key=lambda name: all_metrics[name]["roc_auc"]
    )

    best_model = trained_models[best_model_name]

    output = {
        "best_model": best_model_name,
        "models": all_metrics
    }

    joblib.dump(best_model, "model/churn_model.pkl")

    with open("model/metrics.json", "w") as file:
        json.dump(output, file, indent=4)

    print(f"\nBest model: {best_model_name}")
    print("Best model saved to model/churn_model.pkl")
    print("Metrics saved to model/metrics.json")


if __name__ == "__main__":
    main()
