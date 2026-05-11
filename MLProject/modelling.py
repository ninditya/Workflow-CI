"""
MLProject entry point — trains earthquake significance model.
Stores artifacts to DagsHub (configured via MLFLOW_TRACKING_URI env var).
"""
import argparse
import json
import os
import tempfile
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)


def _plot_confusion_matrix(cm: np.ndarray, path: str) -> None:
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Not Significant", "Significant"],
        yticklabels=["Not Significant", "Significant"],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()


def main(n_estimators: int, max_depth, min_samples_split: int) -> None:
    data_path = os.path.join(
        os.path.dirname(__file__),
        "earthquake_preprocessing",
        "earthquake_preprocessing.csv",
    )
    df = pd.read_csv(data_path)
    X = df.drop("significant", axis=1)
    y = df["significant"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    from contextlib import nullcontext
    if os.environ.get("MLFLOW_RUN_ID"):
        ctx = nullcontext()
    else:
        mlflow.set_experiment("Earthquake-CI")
        ctx = mlflow.start_run()
    with ctx:
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("random_state", 42)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        acc     = accuracy_score(y_test, y_pred)
        prec    = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec     = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1      = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba[:, 1])

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="earthquake-prediction-model",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cm = confusion_matrix(y_test, y_pred)
            cm_path = os.path.join(tmpdir, "training_confusion_matrix.png")
            _plot_confusion_matrix(cm, cm_path)
            mlflow.log_artifact(cm_path)

        print(f"accuracy={acc:.4f}  roc_auc={roc_auc:.4f}  f1={f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--min_samples_split", type=int, default=2)
    args = parser.parse_args()

    max_depth = None if args.max_depth == 0 else args.max_depth
    main(args.n_estimators, max_depth, args.min_samples_split)
