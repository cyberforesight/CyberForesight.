"""
CyberForeSight AI — Baseline Model Pipeline (Phase 2)
Project: SIH26153 · AI-based Network Attack Forecasting from Network Traffic Data (NTRO)

Module: src/baseline.py
Description:
    Implements Phase 2 (Steps 19-21) of PHASES.md and ADR-002:
      1. Loads the normalized time-window feature dataset (data/windows/window_features_normalized.csv).
      2. Performs a strict temporal/chronological train/test split (no shuffling).
      3. Trains a LogisticRegression classifier with `is_attack_window` as the binary target.
      4. Evaluates the classifier on the held-out test set: Precision, Recall, F1-score,
         False Positive Rate (FPR), AUC-ROC, and Confusion Matrix.
      5. Prints metrics to the terminal and serializes the model to `models/baseline_lr.joblib`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("baseline")

# Default file paths
DEFAULT_DATA_PATH = Path("data/windows/window_features_normalized.csv")
DEFAULT_MODELS_DIR = Path("models")
DEFAULT_MODEL_PATH = DEFAULT_MODELS_DIR / "baseline_lr.joblib"


def load_window_dataset(
    data_path: Union[str, Path] = DEFAULT_DATA_PATH
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Loads the normalized time-window dataset and extracts feature column names.

    Args:
        data_path: Path to the normalized CSV file.

    Returns:
        Tuple of (DataFrame, list of feature column names).
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature dataset not found at: {path}")

    logger.info(f"Loading normalized window dataset from: {path}")
    df = pd.read_csv(path)

    # Validate target column exists
    if "is_attack_window" not in df.columns:
        raise ValueError("Target column 'is_attack_window' missing from dataset.")

    # Sort chronologically if window_timestamp is available
    if "window_timestamp" in df.columns:
        df["window_timestamp"] = pd.to_datetime(df["window_timestamp"], errors="coerce")
        df = df.sort_values("window_timestamp").reset_index(drop=True)

    non_feature_cols = {"window_timestamp", "is_attack_window"}
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    logger.info(
        f"Loaded dataset: {df.shape[0]:,} windows x {df.shape[1]} columns ({len(feature_cols)} features)."
    )
    return df, feature_cols


def temporal_train_test_split(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "is_attack_window",
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Splits the dataset chronologically into train and test sets without shuffling,
    respecting temporal order per ARCHITECTURE.md and ADR-002.

    Args:
        df: Input DataFrame sorted chronologically.
        feature_cols: List of numerical feature columns.
        target_col: Target column name.
        test_size: Proportion of dataset to include in test split (default: 0.2).

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, timestamps_train, timestamps_test).
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

    n_samples = len(df)
    split_idx = int(n_samples * (1.0 - test_size))

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].astype(int).copy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].astype(int).copy()

    ts_train = train_df["window_timestamp"] if "window_timestamp" in train_df else pd.Series(dtype=object)
    ts_test = test_df["window_timestamp"] if "window_timestamp" in test_df else pd.Series(dtype=object)

    logger.info(
        f"Temporal split ({1.0 - test_size:.0%}/{test_size:.0%}): "
        f"Train={len(X_train):,} samples (Attacks={y_train.sum():,}), "
        f"Test={len(X_test):,} samples (Attacks={y_test.sum():,})."
    )

    return X_train, X_test, y_train, y_test, ts_train, ts_test


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_iter: int = 1000,
    random_state: int = 42,
    solver: str = "lbfgs",
    class_weight: Optional[str] = None,
) -> LogisticRegression:
    """
    Trains a Logistic Regression baseline classifier.

    Args:
        X_train: Feature matrix for training.
        y_train: Target labels for training.
        max_iter: Maximum solver iterations.
        random_state: Random seed for reproducibility.
        solver: Optimization algorithm.
        class_weight: Optional class weighting (e.g. 'balanced').

    Returns:
        Fitted LogisticRegression model.
    """
    logger.info(
        f"Training LogisticRegression baseline (max_iter={max_iter}, class_weight={class_weight})..."
    )
    model = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        solver=solver,
        class_weight=class_weight,
    )
    model.fit(X_train, y_train)
    logger.info("Baseline LogisticRegression model training complete.")
    return model


def evaluate_baseline_model(
    model: LogisticRegression,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Evaluates the model on the test set and calculates key metrics:
    Precision, Recall, F1-score, False Positive Rate (FPR), Accuracy,
    AUC-ROC, and Confusion Matrix.

    Args:
        model: Trained classifier.
        X_test: Test features.
        y_test: Ground truth labels.

    Returns:
        Dictionary of evaluation metrics.
    """
    logger.info("Evaluating baseline model on held-out test set...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Confusion matrix breakdown
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # False Positive Rate: FP / (FP + TN)
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_test, y_pred))
    auc_roc = float(roc_auc_score(y_test, y_prob)) if y_prob is not None else 0.0

    metrics: Dict[str, Any] = {
        "test_samples": len(y_test),
        "attack_windows": int(y_test.sum()),
        "benign_windows": int(len(y_test) - y_test.sum()),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate": fpr,
        "accuracy": accuracy,
        "auc_roc": auc_roc,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }

    return metrics


def save_baseline_model(
    model: LogisticRegression,
    output_path: Union[str, Path] = DEFAULT_MODEL_PATH,
) -> None:
    """
    Serializes and saves the trained model to disk using joblib.

    Args:
        model: Trained model instance.
        output_path: Destination filepath.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Trained baseline model successfully saved to: {path}")


def print_evaluation_report(metrics: Dict[str, Any], model_path: Path) -> None:
    """Prints a formatted evaluation report to the terminal."""
    cm = metrics["confusion_matrix"]
    print(f"\n{'=' * 65}")
    print("PHASE 2 -- BASELINE LOGISTIC REGRESSION EVALUATION REPORT")
    print(f"{'=' * 65}")
    print(f"Test Set Windows:       {metrics['test_samples']:,}")
    print(f"  - Benign Windows:     {metrics['benign_windows']:,}")
    print(f"  - Attack Windows:     {metrics['attack_windows']:,}")
    print("-" * 65)
    print(f"Precision:              {metrics['precision']:.4f} ({metrics['precision'] * 100:.2f}%)")
    print(f"Recall:                 {metrics['recall']:.4f} ({metrics['recall'] * 100:.2f}%)")
    print(f"F1-Score:               {metrics['f1_score']:.4f} ({metrics['f1_score'] * 100:.2f}%)")
    print(f"False Positive Rate:    {metrics['false_positive_rate']:.4f} ({metrics['false_positive_rate'] * 100:.2f}%)")
    print(f"AUC-ROC:                {metrics['auc_roc']:.4f}")
    print(f"Accuracy:               {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print("-" * 65)
    print("Confusion Matrix:")
    print(f"  True Negatives (TN):   {cm['true_negatives']:>5}")
    print(f"  False Positives (FP):  {cm['false_positives']:>5}")
    print(f"  False Negatives (FN):  {cm['false_negatives']:>5}")
    print(f"  True Positives (TP):   {cm['true_positives']:>5}")
    print("-" * 65)
    print(f"Model Artifact:         {model_path}")
    print(f"{'=' * 65}\n")


def run_baseline_pipeline(
    data_path: Union[str, Path] = DEFAULT_DATA_PATH,
    model_output_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    test_size: float = 0.2,
    max_iter: int = 1000,
    class_weight: Optional[str] = None,
) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """
    Executes the complete baseline model pipeline.

    Args:
        data_path: Path to normalized CSV.
        model_output_path: Path to save trained model.
        test_size: Fraction of chronological data reserved for testing.
        max_iter: Max iterations for solver.
        class_weight: Optional class weight strategy ('balanced' or None).

    Returns:
        Tuple of (trained model, evaluation metrics dictionary).
    """
    df, feature_cols = load_window_dataset(data_path)
    X_train, X_test, y_train, y_test, _, _ = temporal_train_test_split(
        df=df, feature_cols=feature_cols, test_size=test_size
    )

    model = train_baseline_model(
        X_train=X_train,
        y_train=y_train,
        max_iter=max_iter,
        class_weight=class_weight,
    )

    metrics = evaluate_baseline_model(model=model, X_test=X_test, y_test=y_test)
    save_baseline_model(model=model, output_path=model_output_path)
    print_evaluation_report(metrics=metrics, model_path=Path(model_output_path))

    return model, metrics


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CyberForeSight AI — Baseline Logistic Regression Pipeline (Phase 2)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to normalized time-window CSV dataset (default: data/windows/window_features_normalized.csv)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Destination path for trained model artifact (default: models/baseline_lr.joblib)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of chronological dataset for test evaluation (default: 0.2)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum solver iterations for Logistic Regression (default: 1000)",
    )
    parser.add_argument(
        "--class-weight",
        type=str,
        default=None,
        choices=["balanced", "none"],
        help="Class weighting strategy ('balanced' or None)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI execution entrypoint."""
    args = parse_args()
    class_weight = None if args.class_weight == "none" else args.class_weight

    run_baseline_pipeline(
        data_path=args.data_path,
        model_output_path=args.model_path,
        test_size=args.test_size,
        max_iter=args.max_iter,
        class_weight=class_weight,
    )


if __name__ == "__main__":
    main()
