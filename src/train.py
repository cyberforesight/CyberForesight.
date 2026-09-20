"""
CyberForeSight AI — V2 Training Pipeline
Phase 2: End-to-end training orchestration with:
- 36 canonical features
- Partition-specific sequence construction
- Train-only scaler
- Strict chronological splits
- Early stopping on supplemental validation
- Frozen threshold (0.5)
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

from src.constants_v2 import (
    CANONICAL_36_FEATURES,
    EVAL_THRESHOLD,
    LSTM_V2_CONFIG,
    MONITORING_VAL_DATE,
    RAW_DATA_DIR,
    SEQ_LEN,
    TEST_DATES,
    TRAIN_DATES,
    TRAIN_CONFIG,
    V2_BASELINE_PATH,
    V2_LSTM_PATH,
    V2_METADATA_PATH,
    V2_SCALER_PATH,
    V2_SHAP_PATH,
    SUPPLEMENTAL_VAL_FRACTION,
    WINDOW_SIZE_SECONDS,
)
from src.features import scale_feature_matrix
from src.lstm_model import LSTMWorldModel, LSTMWorldModel
from src.scalers_v2 import fit_scaler_v2, load_scaler_v2, transform_with_scaler_v2
from src.sequences_v2 import (
    create_sequences_for_partition,
    prepare_all_partition_sequences,
)

logger = logging.getLogger("train_v2")

def train_lstm_v2(
    train_X: np.ndarray,
    train_y: np.ndarray,
    supp_val_X: np.ndarray,
    supp_val_y: np.ndarray,
    config: Dict,
    device: Optional[torch.device] = None,
) -> Tuple[LSTMWorldModel, List[float], List[float]]:
    """
    Trains V2 LSTM with early stopping on supplemental validation.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = LSTMWorldModel(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )
    model.to(device)
    
    pos_weight = config.get("pos_weight")
    if pos_weight is not None:
        pos_weight_tensor = torch.tensor(pos_weight, device=device)
        # Use BCEWithLogitsLoss for numerical stability; extract logits from model
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        use_logits = True
    else:
        criterion = nn.BCELoss()
        use_logits = False
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    train_dataset = TensorDataset(
        torch.tensor(train_X, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False)
    
    val_dataset = TensorDataset(
        torch.tensor(supp_val_X, dtype=torch.float32),
        torch.tensor(supp_val_y, dtype=torch.float32),
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    
    train_losses: List[float] = []
    val_losses: List[float] = []
    
    patience_counter = 0
    patience = config.get("early_stopping_patience", 5)
    min_delta = config.get("early_stopping_min_delta", 1e-4)
    best_val_loss = float("inf")
    best_model_state = None
    
    logger.info(f"Starting V2 training on {len(train_X):,} sequences, "
                f"{len(supp_val_X):,} validation sequences, {config['epochs']} max epochs...")
    
    # For BCEWithLogitsLoss, we need logits (pre-sigmoid). 
    # Access linear layer directly: model.linear(model.dropout(model.lstm(x)[0][:, -1, :]))
    def get_logits(x):
        hidden = model.init_hidden(x.size(0), x.device)
        out, _ = model.lstm(x, hidden)
        last_step_hidden = out[:, -1, :]
        last_step_dropped = model.dropout(last_step_hidden)
        return model.linear(last_step_dropped).squeeze(-1)
    
    for epoch in range(1, config["epochs"] + 1):
        # Training
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            if use_logits:
                predictions = get_logits(batch_x)
            else:
                predictions = model(batch_x).squeeze(-1)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * len(batch_y)
        
        epoch_train_loss = running_loss / len(train_X)
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                if use_logits:
                    predictions = get_logits(batch_x)
                else:
                    predictions = model(batch_x).squeeze(-1)
                loss = criterion(predictions, batch_y)
                running_val_loss += loss.item() * len(batch_y)
        
        epoch_val_loss = running_val_loss / len(supp_val_X)
        val_losses.append(epoch_val_loss)
        
        if epoch % 5 == 0 or epoch == 1 or epoch == config["epochs"]:
            logger.info(f"Epoch [{epoch:2d}/{config['epochs']:2d}] - "
                       f"Train Loss: {epoch_train_loss:.4f} - Val Loss: {epoch_val_loss:.4f}")
        
        # Early stopping
        if epoch_val_loss < best_val_loss - min_delta:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch} (val loss not improved for {patience} epochs)")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break
    
    return model, train_losses, val_losses


def find_optimal_threshold(
    model: LSTMWorldModel,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: Optional[torch.device] = None,
) -> float:
    """
    Finds the optimal decision threshold by maximizing F1 score on validation data.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        probs = model(val_tensor).squeeze(-1).cpu().numpy()
    
    y_true = y_val.astype(int)
    
    # Sweep thresholds from 0.01 to 0.99
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1 = 0.0
    best_threshold = 0.5
    
    for thresh in thresholds:
        y_pred = (probs >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    logger.info(f"Optimal threshold (F1-max on validation): {best_threshold:.4f} (F1={best_f1:.4f})")
    return float(best_threshold)


def find_temperature_scaling(
    model: LSTMWorldModel,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: Optional[torch.device] = None,
) -> float:
    """
    Finds the optimal temperature scaling parameter T on validation data.
    Temperature scaling calibrates probabilities: p_calibrated = sigmoid(logits / T).
    Optimizes T to minimize Negative Log Likelihood (NLL) on validation set.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()
    
    # Get logits (pre-sigmoid) from model
    def get_logits(x):
        hidden = model.init_hidden(x.size(0), x.device)
        out, _ = model.lstm(x, hidden)
        last_step_hidden = out[:, -1, :]
        last_step_dropped = model.dropout(last_step_hidden)
        return model.linear(last_step_dropped).squeeze(-1)
    
    with torch.no_grad():
        val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_true = torch.tensor(y_val, dtype=torch.float32).to(device)
        logits = get_logits(val_tensor)
    
    # Optimize temperature T using gradient descent on NLL
    logits_cpu = logits.cpu()
    y_true_cpu = y_true.cpu()
    
    # Initialize temperature
    T = torch.tensor(1.0, requires_grad=True)
    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=50)
    
    def closure():
        optimizer.zero_grad()
        # Calibrated probabilities: sigmoid(logits / T)
        probs_cal = torch.sigmoid(logits_cpu / T.clamp(min=0.01))
        # NLL loss
        nll = -(y_true_cpu * torch.log(probs_cal + 1e-8) + (1 - y_true_cpu) * torch.log(1 - probs_cal + 1e-8)).mean()
        nll.backward()
        return nll
    
    optimizer.step(closure)
    
    optimal_T = float(T.clamp(min=0.01).item())
    logger.info(f"Optimal temperature scaling T: {optimal_T:.4f}")
    return optimal_T


def apply_temperature_scaling(
    probs: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """
    Applies temperature scaling to probabilities.
    For probabilities from sigmoid: logits = log(p / (1-p))
    Then calibrated = sigmoid(logits / T)
    """
    # Convert probs to logits
    eps = 1e-8
    probs_clipped = np.clip(probs, eps, 1 - eps)
    logits = np.log(probs_clipped / (1 - probs_clipped))
    # Apply temperature
    logits_cal = logits / temperature
    # Back to probabilities
    probs_cal = 1 / (1 + np.exp(-logits_cal))
    return probs_cal


def evaluate_lstm_v2(
    model: LSTMWorldModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = EVAL_THRESHOLD,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Evaluates V2 LSTM on test set with all required metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        probs = model(test_tensor).squeeze(-1).cpu().numpy()
    
    y_true = y_test.astype(int)
    y_pred = (probs >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))
    auc_roc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 0.0
    
    metrics = {
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
    
    return metrics, probs


def compute_early_warning_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    target_timestamps: List,
    threshold: float = EVAL_THRESHOLD,
) -> Dict:
    """
    Computes early-warning evaluation metrics with corrected timing.
    
    Each entry in y_true/probs corresponds to target window t+1.
    The target timestamp is the timestamp of window t+1.
    
    Attack onset = first attack target window t+1 in a contiguous episode
    Warning = first probability >= threshold at or before onset
    Lead time = (onset_timestamp - warning_timestamp) in minutes
    """
    y_true = np.array(y_true)
    probs = np.array(probs)
    
    # Find contiguous attack episodes in target windows
    attack_indices = np.where(y_true == 1)[0]
    
    # Group into contiguous episodes
    episodes = []
    if len(attack_indices) > 0:
        current_start = attack_indices[0]
        current_end = attack_indices[0]
        for idx in attack_indices[1:]:
            if idx == current_end + 1:
                current_end = idx
            else:
                episodes.append((current_start, current_end))
                current_start = idx
                current_end = idx
        episodes.append((current_start, current_end))
    
    lead_times = []
    early_warning_count = 0
    missed_attacks = 0
    
    for onset_idx, _ in episodes:
        onset_ts = target_timestamps[onset_idx]
        
        # Scan backwards from onset to find first warning at or before onset
        warning_idx = None
        for i in range(onset_idx, -1, -1):
            if probs[i] >= threshold:
                if y_true[i] == 0:  # This is a false warning (predicted attack in benign window)
                    warning_idx = i
                    break
                elif y_true[i] == 1 and i < onset_idx:
                    # This is a true attack window before onset (continuous attack)
                    warning_idx = i
                    break
        
        if warning_idx is not None:
            warning_ts = target_timestamps[warning_idx]
            lead_time_seconds = (onset_ts - warning_ts).total_seconds()
            lead_time_minutes = lead_time_seconds / 60.0
            lead_times.append(lead_time_minutes)
            if lead_time_minutes > 0:
                early_warning_count += 1
        else:
            # No warning before onset = missed attack
            missed_attacks += 1
    
    # False warnings: predicted attack in benign windows
    benign_indices = np.where(y_true == 0)[0]
    false_warnings = 0
    for idx in benign_indices:
        if probs[idx] >= threshold:
            # Check if this is within 5 windows before an attack (not a false warning)
            is_before_attack = False
            for onset_idx, _ in episodes:
                if onset_idx - idx <= 5 and onset_idx > idx:
                    is_before_attack = True
                    break
            if not is_before_attack:
                false_warnings += 1
    
    positive_lead_times = [lt for lt in lead_times if lt > 0]
    
    return {
        "early_warning_count": early_warning_count,
        "total_attacks": len(episodes),
        "early_warning_rate": early_warning_count / len(episodes) if len(episodes) > 0 else 0.0,
        "missed_attack_rate": missed_attacks / len(episodes) if len(episodes) > 0 else 0.0,
        "mean_lead_time_minutes": float(np.mean(positive_lead_times)) if positive_lead_times else 0.0,
        "median_lead_time_minutes": float(np.median(positive_lead_times)) if positive_lead_times else 0.0,
        "false_warnings": false_warnings,
        "false_warning_rate": false_warnings / len(benign_indices) if len(benign_indices) > 0 else 0.0,
    }


def train_baseline_v2(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
) -> Tuple[object, Dict]:
    """
    Trains V2 Logistic Regression baseline on the same scaled features.
    Uses flattened single-window representation (not sequential).
    """
    from sklearn.linear_model import LogisticRegression
    
    # Flatten sequences: LSTM uses (5, 36) but LR uses (36,) - single window
    # For baseline, use the last window of each sequence (t) to predict t+1
    train_X_flat = train_X[:, -1, :]  # Last window of each sequence
    test_X_flat = test_X[:, -1, :]
    
    model = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
    model.fit(train_X_flat, train_y)
    
    y_pred = model.predict(test_X_flat)
    y_prob = model.predict_proba(test_X_flat)[:, 1] if hasattr(model, "predict_proba") else None
    
    cm = confusion_matrix(test_y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    precision = float(precision_score(test_y, y_pred, zero_division=0))
    recall = float(recall_score(test_y, y_pred, zero_division=0))
    f1 = float(f1_score(test_y, y_pred, zero_division=0))
    accuracy = float(accuracy_score(test_y, y_pred))
    auc_roc = float(roc_auc_score(test_y, y_prob)) if y_prob is not None and len(np.unique(test_y)) > 1 else 0.0
    
    metrics = {
        "test_samples": len(test_y),
        "attack_windows": int(test_y.sum()),
        "benign_windows": int(len(test_y) - test_y.sum()),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate": fpr,
        "accuracy": accuracy,
        "auc_roc": auc_roc,
        "threshold": 0.5,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }
    
    return model, metrics


def save_v2_artifacts(
    lstm_model: LSTMWorldModel,
    lr_model,
    scaler: StandardScaler,
    lstm_metrics: Dict,
    lr_metrics: Dict,
    train_losses: List[float],
    val_losses: List[float],
    train_config: Dict,
    model_config: Dict,
    optimal_threshold: float = EVAL_THRESHOLD,
    optimal_temperature: float = 1.0,
    benign_means: Optional[Dict[str, float]] = None,
) -> None:
    """
    Saves ALL V2 artifacts to versioned paths.
    Does NOT touch V1 artifacts.
    """
    # Save LSTM weights
    V2_LSTM_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(lstm_model.state_dict(), V2_LSTM_PATH)
    logger.info(f"V2 LSTM weights saved to: {V2_LSTM_PATH}")
    
    # Save scaler
    import joblib
    joblib.dump(scaler, V2_SCALER_PATH)
    logger.info(f"V2 scaler saved to: {V2_SCALER_PATH}")
    
    # Save LR baseline
    joblib.dump(lr_model, V2_BASELINE_PATH)
    logger.info(f"V2 baseline saved to: {V2_BASELINE_PATH}")
    
    # Save metadata
    metadata = {
        "model_architecture": "LSTMWorldModel",
        "version": "v2",
        "input_size": model_config["input_size"],
        "hidden_size": model_config["hidden_size"],
        "num_layers": model_config["num_layers"],
        "dropout": model_config["dropout"],
        "sequence_length": model_config["sequence_length"],
        "feature_names": CANONICAL_36_FEATURES,
        "train_dates": TRAIN_DATES,
        "supplemental_validation_fraction": SUPPLEMENTAL_VAL_FRACTION,
        "monitoring_val_date": MONITORING_VAL_DATE,
        "test_dates": TEST_DATES,
        "scaler_path": str(V2_SCALER_PATH),
        "loss_function": train_config["loss"],
        "optimizer": train_config["optimizer"],
        "learning_rate": train_config["learning_rate"],
        "batch_size": train_config["batch_size"],
        "max_epochs": train_config["epochs"],
        "actual_epochs_trained": len(train_losses),
        "early_stopping_patience": train_config.get("early_stopping_patience"),
        "class_weight": train_config.get("class_weight"),
        "pos_weight": train_config.get("pos_weight"),
        "threshold": optimal_threshold,
        "temperature": optimal_temperature,
        "seed": train_config["seed"],
        "train_losses": train_losses,
        "val_losses": val_losses,
        "lstm_test_metrics": lstm_metrics,
        "baseline_test_metrics": lr_metrics,
    }
    
    # Add benign baselines for intervention simulation
    if benign_means is not None:
        metadata["benign_feature_means"] = benign_means
    
    V2_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(V2_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"V2 metadata saved to: {V2_METADATA_PATH}")
    
    # Save SHAP placeholder (to be populated by explainability phase)
    shap_data = {
        "version": "v2",
        "status": "not_computed",
        "note": "SHAP will be computed after V2 model training completes"
    }
    V2_SHAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(V2_SHAP_PATH, "w") as f:
        json.dump(shap_data, f, indent=2)
    logger.info(f"V2 SHAP placeholder saved to: {V2_SHAP_PATH}")


def save_v2_shap(
    lstm_model: LSTMWorldModel,
    lr_model,
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    feature_names: List[str],
    shap_path: Union[str, Path] = V2_SHAP_PATH,
    num_bg: int = 50,
    num_test: int = 25,
) -> Dict:
    """
    Generates V2 SHAP explanations using the trained V2 LSTM model.
    Uses explain_lstm_deep from src.explain.py.
    """
    from src.explain import explain_lstm_deep
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model.to(device)
    lstm_model.eval()
    
    # Background: train sequences
    bg_size = min(num_bg, len(train_X))
    bg_tensor = torch.tensor(train_X[:bg_size], dtype=torch.float32).to(device)
    
    # Samples: test attack sequences (where y == 1)
    test_attack_idx = np.where(test_y == 1)[0] if len(test_y) > 0 else np.array([])
    if len(test_attack_idx) > 0:
        sample_indices = test_attack_idx[:min(num_test, len(test_attack_idx))]
    else:
        # Fallback: use any available test samples
        sample_indices = np.arange(min(num_test, len(test_X)))
    
    if len(sample_indices) == 0:
        logger.warning("No test samples available for SHAP; saving status only")
        shap_data = {
            "version": "v2",
            "status": "no_samples",
            "lstm_deep_explainer": {},
            "baseline_linear_explainer": {},
        }
        with open(shap_path, "w") as f:
            json.dump(shap_data, f, indent=2)
        return shap_data
    
    sample_tensor = torch.tensor(test_X[sample_indices], dtype=torch.float32).to(device)
    
    try:
        lstm_shap = explain_lstm_deep(
            model=lstm_model,
            background_tensor=bg_tensor,
            sample_tensor=sample_tensor,
            feature_names=feature_names,
            check_additivity=False,
        )
    except Exception as e:
        logger.error(f"SHAP DeepExplainer failed: {e}")
        lstm_shap = {"error": str(e), "feature_rankings": []}
    
    shap_data = {
        "version": "v2",
        "status": "computed" if "feature_rankings" in lstm_shap and lstm_shap.get("feature_rankings") else "partial",
        "lstm_deep_explainer": lstm_shap,
        "baseline_linear_explainer": {},
    }
    
    with open(shap_path, "w") as f:
        json.dump(shap_data, f, indent=2, default=str)
    logger.info(f"V2 SHAP saved to: {shap_path}")
    
    return shap_data


def run_v2_training_pipeline(
    raw_dir: Path = RAW_DATA_DIR,
    nrows_per_file: Optional[int] = None,
) -> Dict:
    """
    Full V2 training pipeline orchestration.
    
    1. Prepare partition-specific sequences (train, supp_val, monitoring_val, test)
    2. Fit scaler on train windows only
    3. Transform all partitions
    4. Create sequences per partition
    5. Train LSTM with early stopping
    6. Train LR baseline
    7. Evaluate on test
    8. Save V2 artifacts
    """
    print("\n" + "=" * 80)
    print("CYBERFORESIGHT AI — V2 TRAINING PIPELINE")
    print("=" * 80)
    
    # Step 1: Prepare raw windows per partition
    print("\n[1/8] Preparing partition windows...")
    partitions = prepare_all_partition_sequences(
        raw_dir=raw_dir,
        train_dates=TRAIN_DATES,
        supp_val_fraction=SUPPLEMENTAL_VAL_FRACTION,
        monitoring_val_dates=MONITORING_VAL_DATE,
        test_dates=TEST_DATES,
        nrows_per_file=nrows_per_file,


    )
    
    # Log partition stats
    for name, data in partitions.items():
        print(f"  {name}: {len(data['windows']):,} windows, "
              f"{int(data['windows']['is_attack_window'].sum())} attacks")
    
    # Step 2: Fit scaler on TRAIN windows only
    print("\n[2/8] Fitting V2 scaler on TRAIN partition only...")
    scaler = fit_scaler_v2(partitions["train"]["windows"], V2_SCALER_PATH)
    print(f"  Scaler fitted: {scaler.n_features_in_} features")
    
    # Step 3: Transform all partitions
    print("\n[3/8] Transforming all partitions with V2 scaler...")
    for name in ["train", "supp_val", "monitoring_val", "test"]:
        original_cols = partitions[name]["windows"].columns
        partitions[name]["windows"] = transform_with_scaler_v2(
            partitions[name]["windows"], scaler
        )
    
    # Compute benign feature baselines for intervention simulation (AFTER scaling!)
    print("\n[3b/8] Computing benign feature baselines for intervention simulation...")
    train_windows = partitions["train"]["windows"]
    benign_mask = train_windows["is_attack_window"] == 0
    feature_cols = [c for c in train_windows.columns if c not in ["window_id", "window_start", "window_end", "is_attack_window", "src_ip_mode", "dst_ip_mode", "label_mode"]]
    benign_means = train_windows.loc[benign_mask, feature_cols].mean().to_dict()
    print(f"  Computed benign baselines for {len(benign_means)} features (scaled space)")
    
    # Step 4: Create sequences per partition
    print("\n[4/8] Creating partition-specific sequences...")
    for name in ["train", "supp_val", "monitoring_val", "test"]:
        X, y, ts = create_sequences_for_partition(
            partitions[name]["windows"], CANONICAL_36_FEATURES
        )
        partitions[name]["X"] = X
        partitions[name]["y"] = y
        partitions[name]["timestamps"] = ts
        print(f"  {name}: X.shape={X.shape}, y.shape={y.shape}, "
              f"attacks={int(y.sum())}")
    
    # Step 5: Train LSTM with early stopping
    print("\n[5/8] Training LSTM V2...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    merged_config = {**LSTM_V2_CONFIG, **TRAIN_CONFIG}
    lstm_model, train_losses, val_losses = train_lstm_v2(
        partitions["train"]["X"], partitions["train"]["y"],
        partitions["supp_val"]["X"], partitions["supp_val"]["y"],
        merged_config,
        device=device,
    )
    print(f"  Trained {len(train_losses)} epochs, final val_loss={val_losses[-1]:.4f}")
    
    # Step 5b: Find optimal threshold on supplemental validation set
    print("\n[5b/8] Finding optimal threshold on supplemental validation...")
    optimal_threshold = find_optimal_threshold(
        lstm_model,
        partitions["supp_val"]["X"],
        partitions["supp_val"]["y"],
        device=device,
    )
    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    
    # Step 5c: Temperature scaling for probability calibration
    print("\n[5c/8] Finding optimal temperature scaling on supplemental validation...")
    optimal_temperature = find_temperature_scaling(
        lstm_model,
        partitions["supp_val"]["X"],
        partitions["supp_val"]["y"],
        device=device,
    )
    print(f"  Optimal temperature: {optimal_temperature:.4f}")
    
    # Step 6: Train LR baseline
    print("\n[6/8] Training V2 Logistic Regression baseline...")
    lr_model, lr_train_metrics = train_baseline_v2(
        partitions["train"]["X"], partitions["train"]["y"],
        partitions["supp_val"]["X"], partitions["supp_val"]["y"],
    )
    print(f"  LR val F1: {lr_train_metrics['f1_score']:.4f}")
    
    # Step 7: Evaluate on test using optimal threshold
    print("\n[7/8] Evaluating on Feb 28 test set...")
    lstm_metrics, lstm_probs = evaluate_lstm_v2(
        lstm_model,
        partitions["test"]["X"],
        partitions["test"]["y"],
        threshold=optimal_threshold,
        device=device,
    )
    # Apply temperature scaling to test probabilities
    lstm_probs_cal = apply_temperature_scaling(lstm_probs, optimal_temperature)
    # Re-evaluate with calibrated probabilities
    lstm_metrics_cal, _ = evaluate_lstm_v2(
        lstm_model,
        partitions["test"]["X"],
        partitions["test"]["y"],
        threshold=optimal_threshold,
        device=device,
    )
    # Override probs with calibrated ones for metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix
    y_true = partitions["test"]["y"].astype(int)
    y_pred_cal = (lstm_probs_cal >= optimal_threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred_cal, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    lstm_metrics_cal = {
        "test_samples": len(y_true),
        "attack_windows": int(y_true.sum()),
        "benign_windows": int(len(y_true) - y_true.sum()),
        "precision": float(precision_score(y_true, y_pred_cal, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_cal, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred_cal, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred_cal)),
        "auc_roc": float(roc_auc_score(y_true, lstm_probs_cal)) if len(np.unique(y_true)) > 1 else 0.0,
        "threshold": optimal_threshold,
        "temperature": optimal_temperature,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }
    print(f"  LSTM Test (calibrated): Attack windows={lstm_metrics_cal['attack_windows']}, "
          f"Benign={lstm_metrics_cal['benign_windows']}, "
          f"Recall={lstm_metrics_cal['recall']:.4f}, F1={lstm_metrics_cal['f1_score']:.4f}")
    
    # Also compute uncalibrated for comparison
    print(f"  LSTM Test (raw):       Recall={lstm_metrics['recall']:.4f}, F1={lstm_metrics['f1_score']:.4f}")
    
    lr_test_metrics = {}
    # Evaluate LR on test (train fresh on train_X, evaluate on test_X)
    lr_model_test, lr_test_metrics = train_baseline_v2(
        partitions["train"]["X"], partitions["train"]["y"],
        partitions["test"]["X"], partitions["test"]["y"],
    )
    print(f"  LR Test: Recall={lr_test_metrics['recall']:.4f}, F1={lr_test_metrics['f1_score']:.4f}")
    
    # Early warning metrics using calibrated probabilities
    ew_metrics = compute_early_warning_metrics(
        partitions["test"]["y"],
        lstm_probs_cal,
        partitions["test"]["timestamps"],
        threshold=optimal_threshold,
    )
    print(f"  Early Warning: rate={ew_metrics['early_warning_rate']:.4f}, "
          f"mean_lead={ew_metrics['mean_lead_time_minutes']:.1f}min, "
          f"missed={ew_metrics['missed_attack_rate']:.4f}")
    
    # Step 8: Save V2 artifacts
    print("\n[8/8] Saving V2 artifacts...")
    save_v2_artifacts(
        lstm_model=lstm_model,
        lr_model=lr_model_test,
        scaler=scaler,
        lstm_metrics=lstm_metrics_cal,
        lr_metrics=lr_test_metrics,
        train_losses=train_losses,
        val_losses=val_losses,
        train_config=TRAIN_CONFIG,
        model_config=LSTM_V2_CONFIG,
        optimal_threshold=optimal_threshold,
        optimal_temperature=optimal_temperature,
        benign_means=benign_means,
    )
    
    # Compute SHAP explanations
    print("\n[8b/8] Computing V2 SHAP explanations...")
    try:
        shap_data = save_v2_shap(
            lstm_model=lstm_model,
            lr_model=lr_model_test,
            train_X=partitions["train"]["X"],
            train_y=partitions["train"]["y"],
            test_X=partitions["test"]["X"],
            test_y=partitions["test"]["y"],
            feature_names=CANONICAL_36_FEATURES,
        )
        print(f"  SHAP status: {shap_data.get('status', 'unknown')}")
    except Exception as e:
        print(f"  SHAP generation failed: {e}")
        print(f"  V2 LSTM and scaler artifacts preserved.")
    
    # Final report
    print("\n" + "=" * 80)
    print("V2 TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best epoch: {len(train_losses)} total epochs trained")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final val loss: {val_losses[-1]:.4f}")
    print(f"\nLSTM Test Metrics:")
    print(f"  Precision: {lstm_metrics['precision']:.4f}")
    print(f"  Recall: {lstm_metrics['recall']:.4f}")
    print(f"  F1: {lstm_metrics['f1_score']:.4f}")
    print(f"  AUC-ROC: {lstm_metrics['auc_roc']:.4f}")
    print(f"  Accuracy: {lstm_metrics['accuracy']:.4f}")
    print(f"\nLR Baseline Test Metrics:")
    print(f"  Precision: {lr_test_metrics['precision']:.4f}")
    print(f"  Recall: {lr_test_metrics['recall']:.4f}")
    print(f"  F1: {lr_test_metrics['f1_score']:.4f}")
    print(f"  AUC-ROC: {lr_test_metrics['auc_roc']:.4f}")
    print("=" * 80)
    
    return {
        "partitions": {k: {"X_shape": v["X"].shape, "y_sum": int(v["y"].sum())} 
                       for k, v in partitions.items()},
        "lstm_metrics": lstm_metrics,
        "lr_metrics": lr_test_metrics,
        "early_warning": ew_metrics,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "shap_status": shap_data.get("status", "error") if 'shap_data' in dir() else "not_computed",
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_v2_training_pipeline()