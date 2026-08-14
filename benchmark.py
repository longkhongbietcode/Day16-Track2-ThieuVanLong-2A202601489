#!/usr/bin/env python3
"""LightGBM benchmark for the Kaggle Credit Card Fraud dataset."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and benchmark LightGBM for fraud detection."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("creditcard.csv"),
        help="Dataset path (default: creditcard.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_result.json"),
        help="Result path (default: benchmark_result.json)",
    )
    return parser.parse_args()


def measure_prediction_times(
    model: LGBMClassifier, rows: pd.DataFrame, repeats: int
) -> list[float]:
    """Warm up the model and return prediction durations in seconds."""
    model.predict_proba(rows)
    durations: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict_proba(rows)
        durations.append(time.perf_counter() - start)
    return durations


def main() -> None:
    args = parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.data.resolve()}")

    load_start = time.perf_counter()
    data = pd.read_csv(args.data)
    load_seconds = time.perf_counter() - load_start

    if "Class" not in data.columns:
        raise ValueError("Dataset must contain the target column 'Class'.")

    x = data.drop(columns=["Class"])
    y = data["Class"].astype(np.int8)

    # Reserve 20% as an untouched test set. Take validation data only from
    # the remaining training portion so early stopping does not leak test data.
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    model = LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=100,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    training_start = time.perf_counter()
    model.fit(
        x_train,
        y_train,
        eval_X=x_valid,
        eval_y=y_valid,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50, first_metric_only=True, verbose=False
            )
        ],
    )
    training_seconds = time.perf_counter() - training_start

    fraud_probability = model.predict_proba(x_test)[:, 1]
    prediction = (fraud_probability >= 0.5).astype(np.int8)

    metrics = {
        "auc_roc": float(roc_auc_score(y_test, fraud_probability)),
        "accuracy": float(accuracy_score(y_test, prediction)),
        "f1_score": float(f1_score(y_test, prediction, zero_division=0)),
        "precision": float(precision_score(y_test, prediction, zero_division=0)),
        "recall": float(recall_score(y_test, prediction, zero_division=0)),
    }

    one_row = x_test.iloc[[0]]
    batch_size = min(1000, len(x_test))
    thousand_rows = x_test.iloc[:batch_size]

    latency_runs = measure_prediction_times(model, one_row, repeats=100)
    throughput_runs = measure_prediction_times(model, thousand_rows, repeats=10)
    latency_seconds = statistics.median(latency_runs)
    batch_seconds = statistics.median(throughput_runs)
    throughput_rows_per_second = batch_size / batch_seconds

    best_iteration = int(model.best_iteration_ or model.n_estimators_)
    result = {
        "dataset": {
            "path": str(args.data.resolve()),
            "total_rows": int(len(data)),
            "feature_count": int(x.shape[1]),
            "fraud_rows": int(y.sum()),
            "train_rows": int(len(x_train)),
            "validation_rows": int(len(x_valid)),
            "test_rows": int(len(x_test)),
        },
        "timing": {
            "data_load_seconds": load_seconds,
            "training_seconds": training_seconds,
        },
        "model": {
            "best_iteration": best_iteration,
            "decision_threshold": 0.5,
            "num_leaves": 15,
            "min_child_samples": 100,
            "reg_lambda": 1.0,
        },
        "metrics": metrics,
        "inference": {
            "latency_1_row_ms": latency_seconds * 1000,
            "latency_measurement": "median of 100 runs after warm-up",
            "batch_size": batch_size,
            "batch_1000_seconds": batch_seconds,
            "throughput_rows_per_second": throughput_rows_per_second,
            "throughput_measurement": "median of 10 runs after warm-up",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "lightgbm": lgb.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "random_state": RANDOM_STATE,
    }

    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nLIGHTGBM FRAUD-DETECTION BENCHMARK")
    print("=" * 62)
    print(f"{'Metric':<43}Result")
    print("-" * 62)
    print(f"{'Data load time':<43}{load_seconds:.6f} s")
    print(f"{'Training time':<43}{training_seconds:.6f} s")
    print(f"{'Best iteration':<43}{best_iteration}")
    print(f"{'AUC-ROC':<43}{metrics['auc_roc']:.6f}")
    print(f"{'Accuracy':<43}{metrics['accuracy']:.6f}")
    print(f"{'F1-Score':<43}{metrics['f1_score']:.6f}")
    print(f"{'Precision':<43}{metrics['precision']:.6f}")
    print(f"{'Recall':<43}{metrics['recall']:.6f}")
    print(f"{'Inference latency (1 row, median)':<43}{latency_seconds * 1000:.6f} ms")
    print(f"{'Inference time (1000 rows, median)':<43}{batch_seconds:.6f} s")
    print(f"{'Inference throughput (1000 rows)':<43}{throughput_rows_per_second:.2f} rows/s")
    print("=" * 62)
    print(f"Results written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
