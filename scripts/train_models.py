"""Train core nanoparticle antibacterial activity models.

The script supports three small, reproducible baseline workflows:
- MIC classification (`MIC_class`)
- MIC regression (`MIC (µg/mL)`)
- MBC regression (`MBC (µg/mL)`)

Outputs are written to `outputs/` by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from paths import MBC_DATA, MIC_DATA, MIC_WITH_CLASS_DATA, OUTPUT_DIR

ID_COLUMNS = {"No.", "Ref", "Material 1", "Formula"}
TARGET_COLUMNS = {"MIC (µg/mL)", "MBC (µg/mL)", "MIC_class"}

TASKS = {
    "mic_class": {
        "data": MIC_WITH_CLASS_DATA,
        "target": "MIC_class",
        "kind": "classification",
    },
    "mic_regression": {
        "data": MIC_DATA,
        "target": "MIC (µg/mL)",
        "kind": "regression",
    },
    "mbc_regression": {
        "data": MBC_DATA,
        "target": "MBC (µg/mL)",
        "kind": "regression",
    },
}


def load_dataset(path: Path, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load a CSV and split it into features and target."""
    df = pd.read_csv(path, encoding="latin1")
    df.columns = df.columns.str.strip()

    if target not in df.columns:
        raise ValueError(f"Target column {target!r} was not found in {path}.")

    df = df.dropna(subset=[target]).copy()
    drop_columns = [c for c in df.columns if c in ID_COLUMNS or c in TARGET_COLUMNS]
    X = df.drop(columns=drop_columns, errors="ignore")
    y = df[target]

    for column in X.select_dtypes(include=["object", "string"]).columns:
        X[column] = X[column].astype(str).str.strip()

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing for numeric and categorical features."""
    categorical_columns = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_columns = [c for c in X.columns if c not in categorical_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def train_classification(X: pd.DataFrame, y: pd.Series, seed: int) -> tuple[Pipeline, dict[str, Any]]:
    """Train and evaluate the MIC class model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=seed,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics: dict[str, Any] = {
        "accuracy": accuracy_score(y_test, predictions),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
    }
    return model, metrics


def train_regression(X: pd.DataFrame, y: pd.Series, seed: int) -> tuple[Pipeline, dict[str, float]]:
    """Train and evaluate a regression model on the original target scale."""
    y_numeric = pd.to_numeric(y, errors="coerce")
    valid_rows = y_numeric.notna()
    X = X.loc[valid_rows].copy()
    y_log = np.log1p(y_numeric.loc[valid_rows])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=seed
    )
    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    actual = np.expm1(y_test)
    predicted = np.expm1(model.predict(X_test))
    metrics = {
        "r2": float(r2_score(actual, predicted)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }
    return model, metrics


def run_task(task_name: str, output_dir: Path, seed: int) -> dict[str, Any]:
    """Run one configured task and persist its model and metrics."""
    config = TASKS[task_name]
    X, y = load_dataset(config["data"], config["target"])

    if config["kind"] == "classification":
        model, metrics = train_classification(X, y, seed)
    else:
        model, metrics = train_regression(X, y, seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{task_name}_model.joblib"
    metrics_path = output_dir / f"{task_name}_metrics.json"
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {
        "task": task_name,
        "rows": len(y),
        "features": X.shape[1],
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=["all", *TASKS.keys()],
        default="all",
        help="Model workflow to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for trained models and metrics.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_names = TASKS.keys() if args.task == "all" else [args.task]
    results = [run_task(task, args.output_dir, args.seed) for task in task_names]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
