"""
CyberForeSight AI — LSTM World Model Pipeline (Phase 3)
Project: SIH26153 · AI-based Network Attack Forecasting from Network Traffic Data (NTRO)

Module: src/lstm_model.py
Description:
    Implements Phase 3 (Steps 22-27) of PHASES.md and ADR-001:
      1. Loads normalized time-window features (data/windows/window_features_normalized.csv).
      2. Constructs rolling sequences of length L=5 to forecast next-window attack state (S_t+1).
      3. Implements PyTorch `LSTMWorldModel` (LSTM layer -> Linear layer -> Sigmoid output).
      4. Performs a chronological train/test split, trains for 20 epochs with BCELoss, and
         evaluates Precision, Recall, F1, False Positive Rate (FPR), and AUC-ROC.
      5. Provides a K-step rollout forecasting function.
      6. Serializes model weights to `models/lstm_world_model.pth`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("lstm_model")

# File paths
DEFAULT_DATA_PATH = Path("data/windows/window_features_normalized.csv")
DEFAULT_MODELS_DIR = Path("models")
DEFAULT_MODEL_PATH = DEFAULT_MODELS_DIR / "lstm_world_model.pth"
DEFAULT_METADATA_PATH = DEFAULT_MODELS_DIR / "lstm_metadata.json"


# ---------------------------------------------------------------------------
# 1. Dataset Loading & Sequence Generation
# ---------------------------------------------------------------------------


def load_window_data(
    data_path: Union[str, Path] = DEFAULT_DATA_PATH
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Loads normalized window feature matrix from CSV, ensuring chronological sort.

    Args:
        data_path: Path to window_features_normalized.csv.

    Returns:
        Tuple of (DataFrame, list of feature names).
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Normalized window dataset not found at: {path}")

    df = pd.read_csv(path)
    if "is_attack_window" not in df.columns:
        raise ValueError("Target column 'is_attack_window' missing from dataset.")

    if "window_timestamp" in df.columns:
        df["window_timestamp"] = pd.to_datetime(df["window_timestamp"], errors="coerce")
        df = df.sort_values("window_timestamp").reset_index(drop=True)

    non_feature_cols = {"window_timestamp", "is_attack_window"}
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    logger.info(
        f"Loaded window dataset: {len(df):,} windows, {len(feature_cols)} features."
    )
    return df, feature_cols


def create_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int = 5,
    target_col: str = "is_attack_window",
) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
    """
    Constructs rolling temporal sequences of length L (last L windows)
    to forecast the target state of the next window (S_{t+1}).

    Args:
        df: DataFrame sorted chronologically.
        feature_cols: List of state vector feature columns.
        seq_len: Number of previous time windows (L=5).
        target_col: Target column name.

    Returns:
        Tuple of:
          - X_seq: Array of shape (N - seq_len, seq_len, num_features)
          - y_seq: Array of shape (N - seq_len,) with target labels at t+1
          - target_timestamps: Timestamps corresponding to forecasted windows
    """
    if len(df) <= seq_len:
        raise ValueError(
            f"Dataset length ({len(df)}) must exceed sequence length ({seq_len})."
        )

    features_matrix = df[feature_cols].values.astype(np.float32)
    targets_vector = df[target_col].values.astype(np.float32)
    timestamps = (
        df["window_timestamp"].tolist()
        if "window_timestamp" in df.columns
        else [None] * len(df)
    )

    X_seq: List[np.ndarray] = []
    y_seq: List[float] = []
    target_ts: List[Any] = []

    for i in range(len(df) - seq_len):
        X_seq.append(features_matrix[i : i + seq_len])
        y_seq.append(targets_vector[i + seq_len])
        target_ts.append(timestamps[i + seq_len])

    X_arr = np.array(X_seq, dtype=np.float32)
    y_arr = np.array(y_seq, dtype=np.float32)

    logger.info(
        f"Constructed {len(X_arr):,} temporal sequences (L={seq_len}, feature_dim={len(feature_cols)})."
    )
    return X_arr, y_arr, target_ts


def temporal_sequence_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Splits sequential dataset chronologically into train and test sets without shuffling,
    respecting temporal order per ARCHITECTURE.md and ADR-001.

    Args:
        X: Sequence tensor array (N, L, D).
        y: Target label array (N,).
        test_size: Fraction of chronological sequences reserved for testing.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test) as PyTorch Tensors.
    """
    n_samples = len(X)
    split_idx = int(n_samples * (1.0 - test_size))

    X_train = torch.tensor(X[:split_idx], dtype=torch.float32)
    y_train = torch.tensor(y[:split_idx], dtype=torch.float32)

    X_test = torch.tensor(X[split_idx:], dtype=torch.float32)
    y_test = torch.tensor(y[split_idx:], dtype=torch.float32)

    logger.info(
        f"Temporal sequence split ({1.0 - test_size:.0%}/{test_size:.0%}): "
        f"Train={len(X_train):,} sequences (Attacks={int(y_train.sum()):,}), "
        f"Test={len(X_test):,} sequences (Attacks={int(y_test.sum()):,})."
    )
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# 2. PyTorch LSTMWorldModel Definition
# ---------------------------------------------------------------------------


class LSTMWorldModel(nn.Module):
    """
    Temporal World Model for network attack forecasting.
    Composed of a 2-layer stacked LSTM -> Dropout (0.2) -> Linear layer -> Sigmoid output.
    """

    def __init__(
        self,
        input_size: int = 36,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """
        Initializes the optimized 2-layer LSTM World Model.

        Args:
            input_size: Number of state vector features per time step (D=36).
            hidden_size: Number of recurrent hidden units (H=64).
            num_layers: Number of stacked LSTM layers (default: 2).
            dropout: Dropout probability (default: 0.2).
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def init_hidden(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Explicitly initializes hidden state (h0) and cell state (c0) to zeros.

        Args:
            batch_size: Number of sequences in batch.
            device: Execution device.

        Returns:
            Tuple of (h0, c0) tensors.
        """
        if device is None:
            device = next(self.parameters()).device
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h0, c0)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the stacked LSTM world model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size).
            hidden: Optional initial (h0, c0) state tuple.

        Returns:
            Attack probability tensor of shape (batch_size,).
        """
        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device)

        out, _ = self.lstm(x, hidden)
        # Take hidden state representation from the final sequence step
        last_step_hidden = out[:, -1, :]
        last_step_dropped = self.dropout(last_step_hidden)
        logits = self.linear(last_step_dropped)
        probabilities = self.sigmoid(logits)
        # Returns 2D tensor (batch_size, 1) for shap.DeepExplainer and downstream compatibility
        return probabilities


# ---------------------------------------------------------------------------
# 3. Model Training & Evaluation
# ---------------------------------------------------------------------------


def train_lstm_model(
    model: LSTMWorldModel,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.003,
    device: Optional[torch.device] = None,
) -> List[float]:
    """
    Trains the LSTM World Model using Binary Cross Entropy Loss (BCELoss).

    Args:
        model: LSTMWorldModel instance.
        X_train: Training sequence features tensor.
        y_train: Training target labels tensor.
        epochs: Number of training epochs (default: 20 per PHASES.md).
        batch_size: Mini-batch size.
        learning_rate: Adam optimizer learning rate.
        device: Torch execution device (CPU/CUDA).

    Returns:
        List of average training loss values per epoch.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    epoch_losses: List[float] = []
    logger.info(
        f"Starting training on {len(X_train):,} sequences for {epochs} epochs (device={device})..."
    )

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_x).squeeze(-1)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(batch_y)

        epoch_loss = running_loss / len(X_train)
        epoch_losses.append(epoch_loss)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            logger.info(f"Epoch [{epoch:2d}/{epochs:2d}] - Loss: {epoch_loss:.4f}")

    return epoch_losses


def evaluate_lstm_model(
    model: LSTMWorldModel,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    threshold: float = 0.5,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Evaluates the trained LSTM model on the held-out test set.
    Calculates Precision, Recall, F1, FPR, AUC-ROC, and Confusion Matrix.

    Args:
        model: Trained model.
        X_test: Test sequences tensor.
        y_test: Test target ground truth tensor.
        threshold: Decision threshold for classification (default: 0.5).
        device: Execution device.

    Returns:
        Dictionary of evaluation metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    with torch.no_grad():
        test_inputs = X_test.to(device)
        probs_tensor = model(test_inputs).squeeze(-1)
        y_probs = probs_tensor.cpu().numpy()

    y_true = y_test.numpy().astype(int)
    y_pred = (y_probs >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # False Positive Rate = FP / (FP + TN)
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))
    auc_roc = float(roc_auc_score(y_true, y_probs)) if len(np.unique(y_true)) > 1 else 0.0

    metrics: Dict[str, Any] = {
        "test_samples": len(y_test),
        "attack_windows": int(y_true.sum()),
        "benign_windows": int(len(y_true) - y_true.sum()),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate": fpr,
        "accuracy": accuracy,
        "auc_roc": auc_roc,
        "threshold": threshold,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }

    return metrics


# ---------------------------------------------------------------------------
# 4. K-Step Rollout Forecasting (PHASES.md Steps 26 & 27)
# ---------------------------------------------------------------------------


def predict_k_step_rollout(
    model: LSTMWorldModel,
    initial_sequence: Union[np.ndarray, torch.Tensor],
    k_steps: int = 5,
    device: Optional[torch.device] = None,
    fixed_features: Optional[Dict[int, float]] = None,
) -> List[float]:
    """
    Feeds the initial historical sequence and predicts forward K windows,
    outputting an attack probability at each step.

    Design & Scoping Note (PHASES.md Step 27):
        Rollout uses a simplified state approximation (autoregressive temporal
        projection), not full 36-dimensional next-state vector regression.
        This provides lightweight forward risk forecasting suitable for the MVP.

    Args:
        model: Trained LSTMWorldModel.
        initial_sequence: Historical sequence tensor of shape (seq_len, input_size)
                          or (1, seq_len, input_size).
        k_steps: Number of forward windows to project (default: 5).
        device: Torch execution device.
        fixed_features: Optional dict mapping feature_index -> fixed_value.
                        These features are held constant at the specified value
                        throughout the rollout (e.g., for intervention simulation).

    Returns:
        List of forecasted attack probabilities for steps t+1, t+2, ..., t+K.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    if isinstance(initial_sequence, np.ndarray):
        seq_tensor = torch.tensor(initial_sequence, dtype=torch.float32)
    else:
        seq_tensor = initial_sequence.clone().detach().to(dtype=torch.float32)

    if seq_tensor.dim() == 2:
        seq_tensor = seq_tensor.unsqueeze(0)  # (1, seq_len, input_size)

    current_seq = seq_tensor.to(device)
    forecasted_probs: List[float] = []

    with torch.no_grad():
        for step in range(k_steps):
            prob = model(current_seq).item()
            forecasted_probs.append(prob)

            # Roll sequence window forward: drop oldest, synthesize approximate next step
            # Preserve any fixed features (interventions) at their specified values
            last_observed_state = current_seq[:, -1:, :].clone()
            
            if fixed_features:
                # Apply fixed feature values to the last observed state before synthesis
                for feat_idx, fixed_val in fixed_features.items():
                    if 0 <= feat_idx < last_observed_state.size(2):
                        last_observed_state[0, 0, feat_idx] = fixed_val
            
            # Approximates next state features modulated by forecasted attack probability
            # But keep fixed features at their intervention values
            pseudo_next_state = last_observed_state * (1.0 + (prob - 0.5) * 0.1)
            
            if fixed_features:
                # Re-apply fixed features after scaling
                for feat_idx, fixed_val in fixed_features.items():
                    if 0 <= feat_idx < pseudo_next_state.size(2):
                        pseudo_next_state[0, 0, feat_idx] = fixed_val
            
            current_seq = torch.cat([current_seq[:, 1:, :], pseudo_next_state], dim=1)

    return forecasted_probs


# ---------------------------------------------------------------------------
# 5. Serialization & Reporting
# ---------------------------------------------------------------------------


def save_model_artifacts(
    model: LSTMWorldModel,
    metrics: Dict[str, Any],
    feature_names: List[str],
    seq_len: int,
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    metadata_path: Union[str, Path] = DEFAULT_METADATA_PATH,
) -> None:
    """
    Saves model state dictionary to .pth and architecture metadata to .json.
    """
    m_path = Path(model_path)
    m_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), m_path)
    logger.info(f"Model weights successfully saved to: {m_path}")

    meta_path = Path(metadata_path)
    metadata = {
        "model_architecture": "LSTMWorldModel",
        "input_size": model.input_size,
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "dropout": model.dropout_p,
        "sequence_length": seq_len,
        "feature_names": feature_names,
        "test_metrics": metrics,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Model metadata successfully saved to: {meta_path}")


def print_evaluation_report(
    metrics: Dict[str, Any],
    model_path: Path,
    epochs: int,
) -> None:
    """Prints a formatted evaluation report comparing against Phase 2 baseline."""
    cm = metrics["confusion_matrix"]
    print(f"\n{'=' * 65}")
    print("PHASE 3 -- LSTM WORLD MODEL EVALUATION REPORT")
    print(f"{'=' * 65}")
    print(f"Training Epochs:        {epochs}")
    print(f"Test Set Sequences:     {metrics['test_samples']:,}")
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


def run_lstm_pipeline(
    data_path: Union[str, Path] = DEFAULT_DATA_PATH,
    model_output_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    metadata_output_path: Union[str, Path] = DEFAULT_METADATA_PATH,
    seq_len: int = 5,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.003,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[LSTMWorldModel, Dict[str, Any]]:
    """
    Executes the end-to-end Phase 3 LSTM training and evaluation pipeline.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    df, feature_cols = load_window_data(data_path)
    X_seq, y_seq, _ = create_sequences(
        df=df, feature_cols=feature_cols, seq_len=seq_len
    )

    X_train, y_train, X_test, y_test = temporal_sequence_split(
        X=X_seq, y=y_seq, test_size=test_size
    )

    model = LSTMWorldModel(
        input_size=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )

    train_lstm_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    metrics = evaluate_lstm_model(model=model, X_test=X_test, y_test=y_test)
    save_model_artifacts(
        model=model,
        metrics=metrics,
        feature_names=feature_cols,
        seq_len=seq_len,
        model_path=model_output_path,
        metadata_path=metadata_output_path,
    )

    print_evaluation_report(
        metrics=metrics, model_path=Path(model_output_path), epochs=epochs
    )

    # Demonstrate 5-step forward rollout on the last test sequence
    sample_seq = X_test[-1]
    rollout_probs = predict_k_step_rollout(model, sample_seq, k_steps=5)
    logger.info(
        f"K-step rollout forecast from last sequence (t+1..t+5): "
        f"{[round(p, 4) for p in rollout_probs]}"
    )

    return model, metrics


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(
        description="CyberForeSight AI — Phase 3 LSTM World Model Pipeline"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to normalized time-window features CSV (default: data/windows/window_features_normalized.csv)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Destination path for PyTorch weights (default: models/lstm_world_model.pth)",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=str(DEFAULT_METADATA_PATH),
        help="Destination path for model metadata JSON (default: models/lstm_metadata.json)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=5,
        help="Sequence length L (default: 5)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.003,
        help="Learning rate (default: 0.003)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="LSTM hidden state dimension (default: 64)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of stacked LSTM layers (default: 2)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability (default: 0.2)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Chronological test set proportion (default: 0.2)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI execution entrypoint."""
    args = parse_args()
    run_lstm_pipeline(
        data_path=args.data_path,
        model_output_path=args.model_path,
        metadata_output_path=args.metadata_path,
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()
