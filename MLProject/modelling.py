import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import os
import shutil
import json
import platform
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, log_loss, precision_score, 
    recall_score, roc_auc_score, ConfusionMatrixDisplay, 
    RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.utils import estimator_html_repr

def train():
    # 1. Setup Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=453)
    parser.add_argument("--max_depth", type=int, default=40)
    parser.add_argument("--min_samples_split", type=int, default=20)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    args = parser.parse_args()

    # 2. Inisialisasi Tracking
    repo_owner = "mrestuilahi1405"
    repo_name = "Eksperimen_SML_Muhammad-Restu-Ilahi"
    
    if os.getenv("GITHUB_ACTIONS") == "true":
        mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
    else:
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

    # 3. Load Data
    data_path = "creditrisk_preprocessing.csv" 
    if not os.path.exists(data_path):
        print(f"Error: File {data_path} tidak ditemukan!")
        return

    df = pd.read_csv(data_path)
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. MLflow Logging & Registration
    with mlflow.start_run():

        model = RandomForestClassifier(
            n_estimators=args.n_estimators, 
            max_depth=args.max_depth, 
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            random_state=42
        )

        model.fit(X_train, y_train)

        tags = {
                    "estimator_name": model.__class__.__name__,
                    "estimator_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
                    "python_version": platform.python_version(),
                    "sklearn_version": sklearn.__version__,
                    "mlflow_version": mlflow.__version__,
                    "os": platform.system()
                }
        mlflow.set_tags(tags)

        params = model.get_params()
        mlflow.log_params(params)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        metrics = {
            "training_accuracy_score": accuracy_score(y_test, y_pred),
            "training_f1_score": f1_score(y_test, y_pred),
            "training_log_loss": log_loss(y_test, y_prob),
            "training_precision_score": precision_score(y_test, y_pred),
            "training_recall_score": recall_score(y_test, y_pred),
            "training_roc_auc": roc_auc_score(y_test, y_prob[:, 1]),
            "training_score": model.score(X_test, y_test)
        }
        
        mlflow.log_metrics(metrics)

        with open("estimator.html", "w", encoding="utf-8") as f:
            f.write(estimator_html_repr(model))
        mlflow.log_artifact("estimator.html")

        plt.figure(figsize=(8, 6))
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Greens', ax=plt.gca())
        plt.title('Training Confusion Matrix')
        plt.savefig("training_confusion_matrix.png")
        mlflow.log_artifact("training_confusion_matrix.png")
        plt.close()

        plt.figure(figsize=(8, 6))
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=plt.gca())
        plt.title('Training ROC Curve')
        plt.savefig("training_roc_curve.png")
        mlflow.log_artifact("training_roc_curve.png")
        plt.close()

        plt.figure(figsize=(8, 6))
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=plt.gca())
        plt.title('Training Precision Recall Curve')
        plt.savefig("training_precision_recall_curve.png")
        mlflow.log_artifact("training_precision_recall_curve.png")
        plt.close()

        with open("metric_info.json", "w") as f:
            json.dump(metrics, f, indent=4)
        mlflow.log_artifact("metric_info.json")

        input_example = X_test.iloc[[0]]
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="model_credit_risk",
            registered_model_name="model_credit_risk",
            input_example=input_example
        )

        base_artifacts_path = "../artifacts"
        model_save_path = os.path.join(base_artifacts_path, "model_credit_risk")
        dataset_artifacts_path = os.path.join(base_artifacts_path, "dataset")

        if os.path.exists(base_artifacts_path):
            shutil.rmtree(base_artifacts_path)

        os.makedirs(dataset_artifacts_path, exist_ok=True)

        mlflow.sklearn.save_model(
            sk_model=model,
            path=model_save_path
        )
        
        shutil.copy(data_path, os.path.join(dataset_artifacts_path, data_path))

        mlflow.log_artifact(data_path, "dataset")

        print(f"✅ Struktur Artefak Berhasil Dibuat!")
        print(f"📍 Model: {model_save_path}")
        print(f"📍 Data : {dataset_artifacts_path}")
        print(f"✅ Training & Registration Selesai! F1: {f1:.4f}, Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train()