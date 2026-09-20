"""
CyberForeSight AI — V2 Constants
Phase 2: Canonical feature definitions and configuration for V2 training pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

# ──────────────────────────────────────────────────────────────────────────────
# V2 Artifact Paths (versioned to preserve V1 artifacts)
# ──────────────────────────────────────────────────────────────────────────────
V2_MODELS_DIR = Path("models")
V2_LSTM_PATH = V2_MODELS_DIR / "lstm_v2.pth"
V2_SCALER_PATH = V2_MODELS_DIR / "scaler_v2.joblib"
V2_METADATA_PATH = V2_MODELS_DIR / "lstm_metadata_v2.json"
V2_BASELINE_PATH = V2_MODELS_DIR / "baseline_lr_v2.joblib"
V2_SHAP_PATH = V2_MODELS_DIR / "shap_summary_v2.json"
DEFAULT_SCALER_PATH = V2_MODELS_DIR / "scaler.joblib"
RAW_DATA_DIR = V2_MODELS_DIR / ".." / "data" / "raw"

# ──────────────────────────────────────────────────────────────────────────────
# Canonical 36 Features (exact order matching V1 scaler and normalized CSV)
# ──────────────────────────────────────────────────────────────────────────────
CANONICAL_36_FEATURES: List[str] = [
    "flow_duration_mean",
    "flow_duration_max",
    "tot_fwd_pkts_sum",
    "tot_fwd_pkts_mean",
    "tot_bwd_pkts_sum",
    "tot_bwd_pkts_mean",
    "totlen_fwd_pkts_sum",
    "totlen_fwd_pkts_mean",
    "totlen_bwd_pkts_sum",
    "totlen_bwd_pkts_mean",
    "fwd_pkt_len_mean_mean",
    "bwd_pkt_len_mean_mean",
    "flow_byts_s_mean",
    "flow_byts_s_max",
    "flow_pkts_s_mean",
    "flow_pkts_s_max",
    "flow_iat_mean_mean",
    "flow_iat_max_max",
    "syn_flag_cnt_sum",
    "rst_flag_cnt_sum",
    "psh_flag_cnt_sum",
    "ack_flag_cnt_sum",
    "urg_flag_cnt_sum",
    "dst_port_nunique",
    "protocol_nunique",
    "flow_count",
    "fwd_pkt_len_max_max",
    "bwd_pkt_len_max_max",
    "pkt_len_max_max",
    "pkt_len_mean_mean",
    "pkt_len_min_min",
    "fin_flag_cnt_sum",
    "init_fwd_win_byts_mean",
    "init_bwd_win_byts_mean",
    "active_mean_mean",
    "idle_mean_mean",
]

# Verify feature count
assert len(CANONICAL_36_FEATURES) == 36, f"Expected 36 features, got {len(CANONICAL_36_FEATURES)}"

# ──────────────────────────────────────────────────────────────────────────────
# Dataset Split Configuration
# ──────────────────────────────────────────────────────────────────────────────
TRAIN_DATES = ["2018-02-14", "2018-02-15"]
MONITORING_VAL_DATE = ["2018-02-22"]  # Cross-day distribution-shift monitor only
TEST_DATES = ["2018-02-28"]            # Strictly held-out final test

# Supplemental validation: last 15% of training windows chronologically
SUPPLEMENTAL_VAL_FRACTION = 0.15

# ──────────────────────────────────────────────────────────────────────────────
# Sequence Configuration
# ──────────────────────────────────────────────────────────────────────────────
SEQ_LEN = 5  # [t-4, t-3, t-2, t-1, t] → predict t+1

# ──────────────────────────────────────────────────────────────────────────────
# LSTM V2 Architecture
# ──────────────────────────────────────────────────────────────────────────────
LSTM_V2_CONFIG = {
    "input_size": 36,
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "sequence_length": SEQ_LEN,
}

# ──────────────────────────────────────────────────────────────────────────────
# Training Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
TRAIN_CONFIG = {
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 0.003,
    "loss": "BCEWithLogitsLoss",
    "optimizer": "Adam",
    "early_stopping_patience": 5,
    "early_stopping_min_delta": 1e-4,
    "class_weight": "balanced",
    "pos_weight": 19.0,
    "seed": 42,
}

# ──────────────────────────────────────────────────────────────────────────────
# Evaluation Threshold (frozen)
# ──────────────────────────────────────────────────────────────────────────────
EVAL_THRESHOLD = 0.5

# ──────────────────────────────────────────────────────────────────────────────
# Window Configuration
# ──────────────────────────────────────────────────────────────────────────────
WINDOW_SIZE_SECONDS = 60

# ──────────────────────────────────────────────────────────────────────────────
# Raw Data Files
# ──────────────────────────────────────────────────────────────────────────────
RAW_DATA_FILES = {
    "2018-02-14": "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "2018-02-15": "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "2018-02-22": "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
    "2018-02-28": "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
}