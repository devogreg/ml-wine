from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EXPERIMENT_NAME = "wine-baseline"
REGISTERED_MODEL_NAME = "wineclf_rf"


def setup_mlflow() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path).dropna()

    df["target"] = (df["quality"] >= 7).astype(int)

    drop_cols = ["quality", "target"]
    if "Id" in df.columns:
        drop_cols.append("Id")

    X = df.drop(columns=drop_cols)
    y = df["target"]
    return X, y


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )


def eval_and_log(
    pipe: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    title_prefix: str,
) -> dict[str, float]:
    y_pred = pipe.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    for name, value in metrics.items():
        mlflow.log_metric(name, float(value))

    disp = ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test)
    plt.title(f"Confusion Matrix - {title_prefix}")
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "confusion_matrix.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(str(out_path), artifact_path="plots")

    return metrics


def run_logistic_regression(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    C: float,
    max_iter: int,
) -> None:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=max_iter)),
        ]
    )

    with mlflow.start_run(run_name="logreg_baseline"):
        mlflow.set_tag("model_family", "linear")
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("scaler", "StandardScaler")

        pipe.fit(X_train, y_train)
        metrics = eval_and_log(pipe, X_test, y_test, "LogisticRegression")

        mlflow.sklearn.log_model(pipe, artifact_path="model")

        print("[LogReg] metrics:", metrics)


def run_random_forest(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    n_estimators: int,
    max_depth: Optional[int],
    class_weight: Optional[str],
    random_state: int = 42,
) -> None:
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=random_state,
    )
    pipe = Pipeline([("clf", rf)])

    with mlflow.start_run(run_name="random_forest_baseline") as run:
        mlflow.set_tag("model_family", "tree_ensemble")
        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth if max_depth is not None else "None")
        mlflow.log_param("class_weight", class_weight if class_weight else "None")
        mlflow.log_param("random_state", random_state)

        pipe.fit(X_train, y_train)
        metrics = eval_and_log(pipe, X_test, y_test, "RandomForest")

        mlflow.sklearn.log_model(pipe, artifact_path="model")
        print("[RandomForest] metrics:", metrics)

        model_uri = f"runs:/{run.info.run_id}/model"
        registered_model = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
        mlflow.set_tag("registered_model_name", REGISTERED_MODEL_NAME)
        print(
            f"[RandomForest] registered in Model Registry as "
            f"{REGISTERED_MODEL_NAME} (version {registered_model.version})"
        )

        clf: RandomForestClassifier = pipe.named_steps["clf"]
        importances = clf.feature_importances_
        feature_names = (
            X_train.columns
            if hasattr(X_train, "columns")
            else [f"f{i}" for i in range(len(importances))]
        )
        fi = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)

            csv_path = tmp_dir / "feature_importances.csv"
            fi.to_csv(csv_path, index=False)
            mlflow.log_artifact(str(csv_path), artifact_path="feature_importance")

            topn = min(15, len(fi))
            plt.figure(figsize=(8, 5))
            plt.barh(
                fi["feature"].head(topn)[::-1],
                fi["importance"].head(topn)[::-1],
            )
            plt.title("RandomForest Feature Importances (Top 15)")
            plt.tight_layout()
            png_path = tmp_dir / "feature_importances.png"
            plt.savefig(png_path, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(str(png_path), artifact_path="feature_importance")


def train_eval(
    data_path: str,
    model_choice: Literal["logreg", "rf", "both"],
    C: float,
    max_iter: int,
    n_estimators: int,
    max_depth: Optional[int],
    class_weight: Optional[str],
) -> None:
    """Ezt hívja a CLI, az Airflow DAG és a Docker is."""
    setup_mlflow()
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    if model_choice in ("logreg", "both"):
        run_logistic_regression(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            C=C,
            max_iter=max_iter,
        )

    if model_choice in ("rf", "both"):
        run_random_forest(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/raw/WineQT.csv",
        help="Path to WineQT CSV (konténerben: /app/data/raw/WineQT.csv)",
    )
    parser.add_argument(
        "--model",
        choices=["logreg", "rf", "both"],
        default="both",
        help="Melyik modellt tanítsuk.",
    )

    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=1000)

    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument(
        "--max_depth",
        type=int,
        default=None,
        help="Max tree depth. None = nincs limit.",
    )
    parser.add_argument(
        "--class_weight",
        type=str,
        default=None,
        help="pl. 'balanced' vagy hagyd üresen",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_eval(
        data_path=args.data,
        model_choice=args.model,
        C=args.C,
        max_iter=args.max_iter,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        class_weight=args.class_weight,
    )
