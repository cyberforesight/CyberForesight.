"""
CyberForeSight AI — Explainability Layer (Phase 4)
Project: SIH26153 · AI-based Network Attack Forecasting from Network Traffic Data (NTRO)

Module: src/explain.py
Description:
    Implements Phase 4 (Steps 28-29) of PHASES.md:
      1. SHAP DeepExplainer on the PyTorch LSTM World Model (using a baseline sample tensor).
      2. SHAP LinearExplainer on the Logistic Regression baseline model.
      3. Extracts per-window and global feature attributions.
      4. Identifies top contributing features that drive attack risk forecasts.
      5. Serializes SHAP explanation metrics to `models/shap_summary.json`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import shap
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lstm_model import LSTMWorldModel, create_sequences, load_window_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("explain")

DEFAULT_DATA_PATH = Path("data/windows/window_features_normalized.csv")
DEFAULT_LSTM_PATH = Path("models/lstm_world_model.pth")
DEFAULT_BASELINE_PATH = Path("models/baseline_lr.joblib")
DEFAULT_OUTPUT_JSON = Path("models/shap_summary.json")


# ---------------------------------------------------------------------------
# 1. SHAP DeepExplainer for PyTorch LSTM World Model
# ---------------------------------------------------------------------------


def explain_lstm_deep(
    model: LSTMWorldModel,
    background_tensor: torch.Tensor,
    sample_tensor: torch.Tensor,
    feature_names: List[str],
    check_additivity: bool = False,
) -> Dict[str, Any]:
    """
    Computes feature attributions for the LSTM World Model using shap.DeepExplainer.

    Args:
        model: Trained LSTMWorldModel in evaluation mode.
        background_tensor: Baseline sample tensor (e.g. 50 sequences from training set).
        sample_tensor: Target sequences to explain (e.g. test set attack windows).
        feature_names: List of 36 feature column names.
        check_additivity: Additivity verification flag (False for recurrent approximations).

    Returns:
        Dictionary containing global feature importances, sample attributions, and top drivers.
    """
    model.eval()
    device = next(model.parameters()).device
    bg = background_tensor.to(device)
    samples = sample_tensor.to(device)

    logger.info(
        f"Initializing shap.DeepExplainer with {len(bg)} background sequences..."
    )
    explainer = shap.DeepExplainer(model, bg)

    logger.info(f"Computing SHAP values for {len(samples)} evaluation sequences...")
    raw_shap_values = explainer.shap_values(samples, check_additivity=check_additivity)

    # raw_shap_values shape: (N_samples, seq_len, num_features, 1) or (N_samples, seq_len, num_features)
    sv_arr = np.array(raw_shap_values)
    if sv_arr.ndim == 4 and sv_arr.shape[-1] == 1:
        sv_arr = sv_arr.squeeze(-1)

    # Average attributions across the temporal sequence (axis 1: seq_len)
    # Result: (N_samples, num_features)
    sample_attributions = np.mean(sv_arr, axis=1)

    # Mean absolute SHAP value per feature across evaluated samples
    global_importance = np.mean(np.abs(sample_attributions), axis=0)

    # Rank features by global attribution magnitude
    ranked_indices = np.argsort(global_importance)[::-1]
    feature_rankings = [
        {
            "rank": int(rank + 1),
            "feature": feature_names[idx],
            "mean_abs_shap": float(global_importance[idx]),
            "mean_attribution": float(np.mean(sample_attributions[:, idx])),
        }
        for rank, idx in enumerate(ranked_indices)
    ]

    return {
        "model_type": "LSTMWorldModel",
        "num_background_samples": len(bg),
        "num_evaluated_samples": len(samples),
        "feature_rankings": feature_rankings,
        "raw_attributions": sample_attributions.tolist(),
    }


# ---------------------------------------------------------------------------
# 2. SHAP LinearExplainer for Baseline Logistic Regression
# ---------------------------------------------------------------------------


def explain_baseline_linear(
    model_path: Union[str, Path] = DEFAULT_BASELINE_PATH,
    data_path: Union[str, Path] = DEFAULT_DATA_PATH,
    num_background: int = 50,
    num_samples: int = 50,
) -> Dict[str, Any]:
    """
    Computes feature attributions for the Logistic Regression baseline model
    using shap.LinearExplainer (PHASES.md Step 28).

    Args:
        model_path: Path to baseline_lr.joblib.
        data_path: Path to normalized features CSV.
        num_background: Number of background references.
        num_samples: Number of test samples to explain.

    Returns:
        Dictionary containing linear model feature importances.
    """
    m_path = Path(model_path)
    if not m_path.exists():
        logger.warning(f"Baseline model not found at {m_path}; skipping LinearExplainer.")
        return {}

    lr_model = joblib.load(m_path)
    df, feature_names = load_window_data(data_path)

    X = df[feature_names]
    bg_df = X.iloc[:num_background]
    test_df = X.iloc[-num_samples:]

    logger.info(f"Computing SHAP values for baseline model using LinearExplainer...")
    explainer = shap.LinearExplainer(lr_model, bg_df)
    shap_vals = explainer.shap_values(test_df)

    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    ranked_indices = np.argsort(mean_abs)[::-1]

    rankings = [
        {
            "rank": int(rank + 1),
            "feature": feature_names[idx],
            "mean_abs_shap": float(mean_abs[idx]),
            "coefficient": float(lr_model.coef_[0][idx]),
        }
        for rank, idx in enumerate(ranked_indices)
    ]

    return {
        "model_type": "LogisticRegressionBaseline",
        "feature_rankings": rankings,
    }


# ---------------------------------------------------------------------------
# 3. Individual Window Explanation Helper
# ---------------------------------------------------------------------------


def explain_single_sequence(
    model: LSTMWorldModel,
    sequence: Union[np.ndarray, torch.Tensor],
    background_tensor: torch.Tensor,
    feature_names: List[str],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Explains a single network state sequence, returning the top-K features
    most responsible for the attack probability forecast.

    Args:
        model: Trained LSTMWorldModel.
        sequence: Sequence tensor of shape (seq_len, num_features).
        background_tensor: Background sample tensor.
        feature_names: Feature names.
        top_k: Number of top features to return.

    Returns:
        List of dictionaries with feature name, attribution, and risk direction.
    """
    model.eval()
    if isinstance(sequence, np.ndarray):
        seq_tensor = torch.tensor(sequence, dtype=torch.float32)
    else:
        seq_tensor = sequence.clone().detach().to(dtype=torch.float32)

    if seq_tensor.dim() == 2:
        seq_tensor = seq_tensor.unsqueeze(0)

    explainer = shap.DeepExplainer(model, background_tensor)
    raw_sv = explainer.shap_values(seq_tensor, check_additivity=False)
    sv = np.array(raw_sv).squeeze()
    # Average across sequence steps -> (num_features,)
    attributions = np.mean(sv, axis=0) if sv.ndim == 2 else sv

    ranked_indices = np.argsort(np.abs(attributions))[::-1][:top_k]
    top_features = []
    for idx in ranked_indices:
        val = float(attributions[idx])
        top_features.append(
            {
                "feature": feature_names[idx],
                "shap_attribution": val,
                "impact": "Increases Attack Risk" if val > 0 else "Decreases Attack Risk",
            }
        )
    return top_features


# ---------------------------------------------------------------------------
# 4. Pipeline Execution & Serialization
# ---------------------------------------------------------------------------


def run_explainability_pipeline(
    data_path: Union[str, Path] = DEFAULT_DATA_PATH,
    lstm_path: Union[str, Path] = DEFAULT_LSTM_PATH,
    baseline_path: Union[str, Path] = DEFAULT_BASELINE_PATH,
    output_json: Union[str, Path] = DEFAULT_OUTPUT_JSON,
    num_bg: int = 50,
    num_test: int = 25,
) -> Dict[str, Any]:
    """
    Executes the full Phase 4 Explainability pipeline.
    """
    df, feature_names = load_window_data(data_path)
    X_seq, _, _ = create_sequences(df, feature_names, seq_len=5)

    # Initialize LSTM model
    model = LSTMWorldModel(
        input_size=len(feature_names),
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
    )
    m_path = Path(lstm_path)
    if not m_path.exists():
        raise FileNotFoundError(f"LSTM model weights not found at: {m_path}")

    model.load_state_dict(torch.load(m_path, map_location="cpu"))
    model.eval()

    # Create background baseline and test sample tensors
    bg_tensor = torch.tensor(X_seq[:num_bg], dtype=torch.float32)
    test_tensor = torch.tensor(X_seq[-num_test:], dtype=torch.float32)

    # 1. Run LSTM DeepExplainer
    lstm_explanations = explain_lstm_deep(
        model=model,
        background_tensor=bg_tensor,
        sample_tensor=test_tensor,
        feature_names=feature_names,
    )

    # 2. Run Baseline LinearExplainer
    baseline_explanations = explain_baseline_linear(
        model_path=baseline_path,
        data_path=data_path,
        num_background=num_bg,
        num_samples=num_test,
    )

    results = {
        "phase": "Phase 4 Explainability (SHAP)",
        "lstm_deep_explainer": lstm_explanations,
        "baseline_linear_explainer": baseline_explanations,
    }

    # Save to disk
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"SHAP explanation results successfully saved to: {out_path}")

    # Print summary report
    print(f"\n{'=' * 65}")
    print("PHASE 4 -- SHAP EXPLAINABILITY SUMMARY REPORT")
    print(f"{'=' * 65}")
    print(f"Evaluated Model:       LSTMWorldModel (2-Layer Stacked)")
    print(f"Background Baseline:   {num_bg} sequences")
    print(f"Explained Windows:     {num_test} sequences")
    print("-" * 65)
    print("Top 10 Influential Features (SHAP DeepExplainer):")
    for r in lstm_explanations["feature_rankings"][:10]:
        print(f"  {r['rank']:2d}. {r['feature']:<30}: {r['mean_abs_shap']:.6f}")
    print("-" * 65)
    print(f"Artifact Saved:        {out_path}")
    print(f"{'=' * 65}\n")

    return results


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(
        description="CyberForeSight AI — Phase 4 SHAP Explainability Layer"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to normalized time-window features CSV",
    )
    parser.add_argument(
        "--lstm-path",
        type=str,
        default=str(DEFAULT_LSTM_PATH),
        help="Path to trained LSTM weights (.pth)",
    )
    parser.add_argument(
        "--baseline-path",
        type=str,
        default=str(DEFAULT_BASELINE_PATH),
        help="Path to trained baseline model (.joblib)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(DEFAULT_OUTPUT_JSON),
        help="Output path for SHAP summary JSON",
    )
    parser.add_argument(
        "--num-bg",
        type=int,
        default=50,
        help="Number of background samples for DeepExplainer (default: 50)",
    )
    parser.add_argument(
        "--num-test",
        type=int,
        default=25,
        help="Number of test sequences to explain (default: 25)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI execution entrypoint."""
    args = parse_args()
    run_explainability_pipeline(
        data_path=args.data_path,
        lstm_path=args.lstm_path,
        baseline_path=args.baseline_path,
        output_json=args.output_json,
        num_bg=args.num_bg,
        num_test=args.num_test,
    )


if __name__ == "__main__":
    main()
