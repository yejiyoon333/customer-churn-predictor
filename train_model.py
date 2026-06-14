import json
import os

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

    categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object", "string"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
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

    model_configs = {
        "logistic_regression": {
            "classifier": LogisticRegression(random_state=42, max_iter=5000),
            "params": {
                "classifier__C": [0.01, 0.1, 1, 10],
                "classifier__solver": ["lbfgs"]
            }
        },
        "random_forest": {
            "classifier": RandomForestClassifier(random_state=42, class_weight="balanced"),
            "params": {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [None, 5, 10, 20],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4]
            }
        },
        "gradient_boosting": {
            "classifier": GradientBoostingClassifier(random_state=42),
            "params": {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__learning_rate": [0.01, 0.05, 0.1],
                "classifier__max_depth": [2, 3, 4],
                "classifier__min_samples_split": [2, 5, 10]
            }
        }
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    all_metrics = {}
    trained_models = {}

    for model_name, config in model_configs.items():
        print(f"\nTraining model: {model_name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", config["classifier"])
            ]
        )

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=config["params"],
            n_iter=10,
            scoring="roc_auc",
            cv=cv,
            random_state=42,
            n_jobs=-1
        )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        test_metrics = evaluate_model(best_model, X_test, y_test)

        metrics = {
            "cv_roc_auc": search.best_score_,
            "best_params": search.best_params_,
            **test_metrics
        }

        all_metrics[model_name] = metrics
        trained_models[model_name] = best_model

        print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
        print(f"Best params: {search.best_params_}")
        print("Test results:")
        print(f"accuracy: {metrics['accuracy']:.4f}")
        print(f"precision: {metrics['precision']:.4f}")
        print(f"recall: {metrics['recall']:.4f}")
        print(f"f1: {metrics['f1']:.4f}")
        print(f"roc_auc: {metrics['roc_auc']:.4f}")

    best_model_name = max(
        all_metrics,
        key=lambda name: all_metrics[name]["cv_roc_auc"]
    )

    best_model = trained_models[best_model_name]

    output = {
        "best_model": best_model_name,
        "selection_metric": "cv_roc_auc",
        "models": all_metrics
    }

    os.makedirs("model", exist_ok=True)

    joblib.dump(best_model, "model/churn_model.pkl")

    with open("model/metrics.json", "w") as file:
        json.dump(output, file, indent=4)

    print(f"\nBest model: {best_model_name}")
    print("Selection metric: cv_roc_auc")
    print("Best model saved to model/churn_model.pkl")
    print("Metrics saved to model/metrics.json")


if __name__ == "__main__":
    main()
