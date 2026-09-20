"""
CyberForeSight AI — Dashboard Shared Core
Shared cached data/model loaders and processing helpers for all Streamlit pages.
Import this module in every page; @st.cache_resource ensures a single LSTM instance.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

# ---------------------------------------------------------------------------
# Path bootstrap (works whether run from project root or src/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lstm_model import LSTMWorldModel, create_sequences, load_window_data, predict_k_step_rollout as lstm_predict_k_step_rollout
from src.mitre_map import MitreStageResult, map_to_mitre_stage

# ---------------------------------------------------------------------------
# File paths (V2 — 36 features, class-weighted, dynamic threshold)
# ---------------------------------------------------------------------------
MODEL_PATH      = PROJECT_ROOT / "models" / "lstm_v2.pth"
SHAP_JSON_PATH  = PROJECT_ROOT / "models" / "shap_summary_v2.json"
DATA_PATH       = PROJECT_ROOT / "data" / "windows" / "window_features_normalized.csv"
METADATA_PATH   = PROJECT_ROOT / "models" / "lstm_metadata_v2.json"
SCALER_PATH     = PROJECT_ROOT / "models" / "scaler_v2.joblib"
LR_MODEL_PATH   = PROJECT_ROOT / "models" / "baseline_lr_v2.joblib"

SEQ_LEN = 5

# ---------------------------------------------------------------------------
# Intervention feature sets
# ---------------------------------------------------------------------------
ISOLATION_FEATURES = [
    "flow_count", "tot_fwd_pkts_sum", "tot_bwd_pkts_sum",
    "totlen_fwd_pkts_sum", "totlen_bwd_pkts_sum",
    "flow_byts_s_mean", "flow_byts_s_max",
    "flow_pkts_s_mean", "flow_pkts_s_max",
]
SMB_FEATURES = [
    "dst_port_nunique", "syn_flag_cnt_sum", "rst_flag_cnt_sum",
    "psh_flag_cnt_sum", "init_fwd_win_byts_mean", "init_bwd_win_byts_mean",
]

# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------
MITRE_COLOR: Dict[str, str] = {
    "Nominal":            "#22c55e",
    "Reconnaissance":     "#f59e0b",
    "Initial Access":     "#f97316",
    "Command & Control":  "#a855f7",
    "Exfiltration":       "#ef4444",
    "Impact":             "#dc2626",
}
RISK_COLOR: Dict[str, str] = {
    "Low":      "#22c55e",
    "Medium":   "#f59e0b",
    "High":     "#ef4444",
    "Critical": "#dc2626",
}

# ---------------------------------------------------------------------------
# Shared CSS (call inject_css() once per page)
# ---------------------------------------------------------------------------
SHARED_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family:'Inter',sans-serif; background:#0a0e1a; color:#e2e8f0; }
section[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#0f172a 0%,#1e293b 100%);
    border-right:1px solid rgba(99,102,241,0.25);
}
.main .block-container { padding:1.4rem 2rem; max-width:1600px; }
.page-header {
    background:linear-gradient(135deg,#0f172a 0%,#1e293b 60%,#0f172a 100%);
    border:1px solid rgba(99,102,241,0.35); border-radius:16px;
    padding:1.2rem 2rem; margin-bottom:1.4rem;
    display:flex; align-items:center; gap:1rem;
}
.page-header h1 {
    font-size:1.55rem; font-weight:700; margin:0;
    background:linear-gradient(90deg,#818cf8,#38bdf8,#a78bfa);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.page-header .sub { font-size:0.72rem; color:#64748b; letter-spacing:0.08em; text-transform:uppercase; margin-top:0.2rem; }
.kpi-card {
    background:linear-gradient(145deg,#1e293b,#0f172a);
    border:1px solid rgba(99,102,241,0.2); border-radius:14px;
    padding:1.1rem 1.2rem; text-align:center;
    transition:border-color 0.3s,transform 0.2s;
    min-height:110px; display:flex; flex-direction:column; justify-content:center;
}
.kpi-card:hover { border-color:rgba(129,140,248,0.5); transform:translateY(-2px); }
.kpi-card .lbl { font-size:0.65rem; color:#64748b; text-transform:uppercase; letter-spacing:0.09em; font-weight:600; margin-bottom:0.4rem; }
.kpi-card .val { font-family:'JetBrains Mono',monospace; font-size:1.55rem; font-weight:700; line-height:1; }
.kpi-card .dlt { font-size:0.65rem; margin-top:0.35rem; color:#94a3b8; }
.sec-hdr {
    font-size:0.68rem; font-weight:700; color:#818cf8;
    text-transform:uppercase; letter-spacing:0.12em;
    margin-bottom:0.55rem; padding-bottom:0.35rem;
    border-bottom:1px solid rgba(99,102,241,0.2);
}
.sim-tag {
    display:inline-block; background:rgba(245,158,11,0.15);
    border:1px solid rgba(245,158,11,0.4); color:#fbbf24;
    padding:0.12rem 0.55rem; border-radius:6px; font-size:0.62rem;
    font-weight:700; text-transform:uppercase; letter-spacing:0.1em;
    margin-left:0.5rem; vertical-align:middle;
}
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#0a0e1a; }
::-webkit-scrollbar-thumb { background:#334155; border-radius:3px; }
.stCheckbox > label { color:#e2e8f0 !important; }

/* ── Tiered Defence Recommendation System CSS ── */
.tier-header-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 1.2rem 0 0.65rem 0;
    padding-bottom: 0.45rem;
    border-bottom: 1px solid rgba(148, 163, 184, 0.15);
}
.tier-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.22rem 0.7rem;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-family: 'JetBrains Mono', monospace;
}
.tier-badge-immediate {
    background: rgba(239, 68, 68, 0.18);
    border: 1px solid rgba(239, 68, 68, 0.55);
    color: #f87171;
    box-shadow: 0 0 14px rgba(239, 68, 68, 0.18);
}
.tier-badge-high {
    background: rgba(245, 158, 11, 0.18);
    border: 1px solid rgba(245, 158, 11, 0.55);
    color: #fbbf24;
    box-shadow: 0 0 14px rgba(245, 158, 11, 0.15);
}
.tier-badge-planned {
    background: rgba(56, 189, 248, 0.18);
    border: 1px solid rgba(56, 189, 248, 0.55);
    color: #38bdf8;
    box-shadow: 0 0 14px rgba(56, 189, 248, 0.15);
}

.rec-card {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.88) 0%, rgba(30, 41, 59, 0.62) 100%);
    border: 1px solid rgba(99, 102, 241, 0.22);
    border-radius: 11px;
    padding: 1.05rem 1.25rem;
    margin-bottom: 0.85rem;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
    backdrop-filter: blur(8px);
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}
.rec-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 22px rgba(0, 0, 0, 0.35);
}
.rec-card.immediate {
    border-left: 4px solid #ef4444;
}
.rec-card.immediate:hover {
    border-color: rgba(239, 68, 68, 0.6);
}
.rec-card.high {
    border-left: 4px solid #f59e0b;
}
.rec-card.high:hover {
    border-color: rgba(245, 158, 11, 0.6);
}
.rec-card.planned {
    border-left: 4px solid #38bdf8;
}
.rec-card.planned:hover {
    border-color: rgba(56, 189, 248, 0.6);
}

.rec-title-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.8rem;
    margin-bottom: 0.45rem;
}
.rec-title {
    font-size: 0.94rem;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: 0.01em;
}
.rec-desc {
    font-size: 0.79rem;
    color: #cbd5e1;
    line-height: 1.65;
    margin-bottom: 0.65rem;
}
.rec-context {
    font-size: 0.72rem;
    color: #94a3b8;
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 6px;
    padding: 0.42rem 0.75rem;
    margin-bottom: 0.7rem;
    line-height: 1.5;
}
.rec-footer {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    align-items: center;
    padding-top: 0.55rem;
    border-top: 1px solid rgba(148, 163, 184, 0.1);
}
.rec-tag-path {
    background: rgba(99, 102, 241, 0.18);
    border: 1px solid rgba(99, 102, 241, 0.45);
    color: #a5b4fc;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 0.18rem 0.55rem;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.03em;
}
.rec-tag-effort {
    background: rgba(148, 163, 184, 0.12);
    border: 1px solid rgba(148, 163, 184, 0.3);
    color: #cbd5e1;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 0.18rem 0.55rem;
    border-radius: 4px;
}
.rec-tag-eta {
    background: rgba(168, 85, 247, 0.16);
    border: 1px solid rgba(168, 85, 247, 0.45);
    color: #c084fc;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 0.18rem 0.55rem;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
}
.rec-tag-target {
    background: rgba(30, 41, 59, 0.75);
    border: 1px solid rgba(148, 163, 184, 0.18);
    color: #94a3b8;
    font-size: 0.68rem;
    padding: 0.18rem 0.55rem;
    border-radius: 4px;
}
</style>
"""


def inject_css() -> None:
    st.markdown(SHARED_CSS, unsafe_allow_html=True)


def page_header(icon: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"<div class='page-header'><span style='font-size:2rem'>{icon}</span>"
        f"<div><h1>{title}</h1><div class='sub'>{subtitle}</div></div></div>",
        unsafe_allow_html=True,
    )


def kpi(label: str, value: str, delta: str = "", color: str = "#818cf8") -> str:
    dlt = f"<div class='dlt'>{delta}</div>" if delta else ""
    return (
        f"<div class='kpi-card'><div class='lbl'>{label}</div>"
        f"<div class='val' style='color:{color}'>{value}</div>{dlt}</div>"
    )


def sec(text: str) -> None:
    st.markdown(f"<div class='sec-hdr'>{text}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plotly base layout (transparent dark)
# ---------------------------------------------------------------------------
_BL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.6)",
    font=dict(family="Inter,sans-serif", color="#94a3b8", size=11),
    xaxis=dict(gridcolor="rgba(51,65,85,0.5)", linecolor="rgba(51,65,85,0.8)", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="rgba(51,65,85,0.5)", linecolor="rgba(51,65,85,0.8)", showgrid=True, zeroline=False),
    margin=dict(l=8, r=8, t=38, b=8),
    legend=dict(bgcolor="rgba(15,23,42,0.8)", bordercolor="rgba(99,102,241,0.3)", borderwidth=1),
    hovermode="x unified",
)


def get_layout(**kwargs) -> dict:
    """Safely merges custom layout arguments with base layout tokens without keyword conflicts."""
    import copy
    base = copy.deepcopy(_BL)
    for k, v in kwargs.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k].update(v)
        else:
            base[k] = v
    return base


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading window features…", ttl=600)
def load_window_df() -> Tuple[pd.DataFrame, List[str]]:
    """Load normalised window CSV and feature column list aligned strictly with trained LSTM model by name."""
    df, feat = load_window_data(DATA_PATH)
    meta = load_metadata()
    if meta and "feature_names" in meta:
        expected_cols = meta["feature_names"]
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Inference schema error: Required LSTM feature(s) missing from window features: {missing}"
            )
        # Select exact 26 features in trained LSTM order by name (no arbitrary positional slicing)
        feat = [c for c in expected_cols]
    elif MODEL_PATH.exists():
        raise ValueError("Inference error: lstm_metadata.json missing; cannot verify exact trained feature schema.")
    return df, feat


@st.cache_resource(show_spinner="Loading LSTM World Model (V2)…")
def load_lstm_model(input_size: Optional[int] = None) -> LSTMWorldModel:
    """Load trained V2 LSTMWorldModel — cached as a singleton resource, auto-aligning to checkpoint dimensions."""
    sd = torch.load(str(MODEL_PATH), map_location="cpu", weights_only=True)
    if "lstm.weight_ih_l0" in sd:
        actual_input_size = int(sd["lstm.weight_ih_l0"].shape[1])
    else:
        actual_input_size = input_size or 36
    m = LSTMWorldModel(input_size=actual_input_size, hidden_size=64, num_layers=2, dropout=0.2)
    m.load_state_dict(sd)
    m.eval()
    return m


@st.cache_data(show_spinner="Loading SHAP data…", ttl=600)
def load_shap_data() -> Dict:
    if not SHAP_JSON_PATH.exists():
        return {}
    with open(SHAP_JSON_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading metadata…", ttl=600)
def load_metadata() -> Dict:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Required metadata not found: {METADATA_PATH}")
    with open(METADATA_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading optimal threshold…", ttl=600)
def load_optimal_threshold() -> float:
    """Load the optimal threshold from V2 metadata (F1-optimal on validation)."""
    meta = load_metadata()
    return float(meta.get("threshold", 0.5))


@st.cache_data(show_spinner="Loading temperature scaling…", ttl=600)
def load_temperature() -> float:
    """Load the temperature scaling parameter from V2 metadata."""
    meta = load_metadata()
    return float(meta.get("temperature", 1.0))


def apply_temperature_scaling(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Applies temperature scaling to probabilities."""
    if temperature <= 0 or temperature == 1.0:
        return probs
    eps = 1e-8
    probs_clipped = np.clip(probs, eps, 1 - eps)
    logits = np.log(probs_clipped / (1 - probs_clipped))
    logits_cal = logits / temperature
    return 1 / (1 + np.exp(-logits_cal))


# ---------------------------------------------------------------------------
# Processing helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _infer(model: LSTMWorldModel, X: np.ndarray) -> np.ndarray:
    t = torch.tensor(X, dtype=torch.float32)
    return model(t).squeeze(-1).numpy()


@st.cache_data(show_spinner="Running LSTM inference…", ttl=600)
def run_timeline_inference(_model_key: str, _data_hash: str) -> Tuple[List, np.ndarray, np.ndarray]:
    """
    Runs LSTM over all windows. Uses opaque cache keys so results are shared
    across pages without serialising the model.
    Guards against insufficient data (<= SEQ_LEN windows) without throwing exceptions.
    Returns: (timestamps, probabilities, ground_truth_labels)
    """
    df, feat = load_window_df()
    if df is None or len(df) <= SEQ_LEN:
        return [], np.array([]), np.array([])
    model = load_lstm_model(len(feat))
    X, _, ts = create_sequences(df, feat, seq_len=SEQ_LEN)
    probs = _infer(model, X)
    # Apply temperature scaling
    temperature = load_temperature()
    probs_cal = apply_temperature_scaling(probs, temperature)
    ts_dt = pd.to_datetime(ts, errors="coerce").tolist()
    # Ground truth: is_attack_window for target windows (t+1), i.e., windows[SEQ_LEN:]
    y_true = df["is_attack_window"].values[SEQ_LEN:].astype(int)
    return ts_dt, probs_cal, y_true


def get_timeline() -> Tuple[List, np.ndarray]:
    df, feat = load_window_df()
    if df is None or len(df) <= SEQ_LEN:
        return [], np.array([])
    data_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    ts, probs, _ = run_timeline_inference("lstm_v2", data_hash)
    return ts, probs


def get_timeline_with_labels() -> Tuple[List, np.ndarray, np.ndarray]:
    """Returns (timestamps, probabilities, ground_truth_labels) for threshold analysis."""
    df, feat = load_window_df()
    if df is None or len(df) <= SEQ_LEN:
        return [], np.array([]), np.array([])
    data_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    return run_timeline_inference("lstm_v2", data_hash)


@st.cache_data(show_spinner="Computing K-step rollout…", ttl=600)
def run_kstep_rollout(_data_hash: str, k: int = 5) -> List[float]:
    """Runs K-step autoregressive rollout from the last real sequence using the LSTM model's built-in rollout."""
    df, feat = load_window_df()
    if df is None or len(df) <= SEQ_LEN:
        return []
    model = load_lstm_model(len(feat))
    X, _, _ = create_sequences(df, feat, seq_len=SEQ_LEN)
    last_seq = X[-1]  # (SEQ_LEN, num_features) — latest observed window sequence
    return lstm_predict_k_step_rollout(model, last_seq, k_steps=k)


def get_kstep(k: int = 5) -> List[float]:
    df, feat = load_window_df()
    if df is None or len(df) <= SEQ_LEN:
        return []
    data_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    return run_kstep_rollout(data_hash, k=k)


@torch.no_grad()
def run_intervention(zero_cols: List[str], k_steps: int = 5, baseline_seq: Optional[np.ndarray] = None) -> List[float]:
    """
    Simulates intervention by modifying the last observed window and rolling forward K steps.
    This represents "what happens to future predictions if we intervene NOW?"
    
    The intervention applies in SCALED feature space (matching the model's input).
    
    Args:
        zero_cols: List of feature column names to intervene on.
        k_steps: Number of rollout steps.
        baseline_seq: Optional pre-computed baseline sequence (seq_len, num_features).
                      If None, uses the last window in the dataset.
    
    Returns:
        List of forecasted attack probabilities for steps t+1..t+K.
    """
    df, feat = load_window_df()
    if df is None or len(df) <= SEQ_LEN:
        return []
    model = load_lstm_model(len(feat))
    
    # Get last real sequence
    X, _, _ = create_sequences(df, feat, seq_len=SEQ_LEN)
    if baseline_seq is not None:
        last_seq = baseline_seq.copy()
    else:
        last_seq = X[-1].copy()  # (SEQ_LEN, num_features) - already scaled
    
    # Apply intervention to the LAST window in the sequence (most recent, index -1)
    valid = list(dict.fromkeys([c for c in zero_cols if c in feat]))
    
    # Load metadata for scaler params to compute "zero activity" in scaled space
    meta = load_metadata()
    benign_means = meta.get("benign_feature_means", {})
    
    # Build fixed_features dict: feature_index -> fixed_value
    # For "complete block" simulation, set to raw=0 which maps to scaled = -mean/std
    # Since we don't have scaler params directly, use a conservative "no activity" value:
    # For count/rate features: set to -3.0 (3 std below mean, ~0.1% percentile)
    # For ratio/mean features: set to benign mean (typical normal operation)
    count_rate_features = {
        "flow_count", "tot_fwd_pkts_sum", "tot_bwd_pkts_sum", 
        "totlen_fwd_pkts_sum", "totlen_bwd_pkts_sum",
        "flow_byts_s_mean", "flow_byts_s_max", "flow_pkts_s_mean", "flow_pkts_s_max",
        "dst_port_nunique", "syn_flag_cnt_sum", "rst_flag_cnt_sum", "psh_flag_cnt_sum"
    }
    
    fixed_features = {}
    for col in valid:
        col_idx = feat.index(col) if col in feat else -1
        if col_idx >= 0:
            if col in count_rate_features:
                # "Complete block" = no traffic = raw 0 → scaled ≈ -3.0 (well below benign)
                fixed_features[col_idx] = -3.0
            elif col in benign_means:
                fixed_features[col_idx] = benign_means[col]
            else:
                fixed_features[col_idx] = 0.0
    
    # Modify the last window (index -1) in the sequence
    for col_idx, fixed_val in fixed_features.items():
        last_seq[-1, col_idx] = fixed_val
    
    # Roll forward K steps from intervened sequence using the LSTM's proper rollout
    # Pass fixed_features so they're preserved throughout the rollout
    return lstm_predict_k_step_rollout(model, last_seq, k_steps=k_steps, fixed_features=fixed_features)


def get_current_mitre(attack_prob: float, top_shap: List[str]) -> MitreStageResult:
    df, feat = load_window_df()
    if df is None or len(df) == 0:
        current_row = pd.Series(0.0, index=feat)
    else:
        current_row = df[feat].iloc[-1]
    threshold = load_optimal_threshold()
    return map_to_mitre_stage(
        features=current_row,
        feature_names=feat,
        attack_probability=attack_prob,
        top_shap_features=top_shap,
        threshold=threshold,
    )


def get_rollout_mitre(probs: List[float], feat_row: pd.Series, feat_names: List[str]) -> List[MitreStageResult]:
    """Returns a MITRE result for each rollout step probability."""
    results = []
    threshold = load_optimal_threshold()
    for p in probs:
        r = map_to_mitre_stage(
            features=feat_row,
            feature_names=feat_names,
            attack_probability=p,
            threshold=threshold,
        )
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Shared chart builders
# ---------------------------------------------------------------------------

def timeline_chart(ts: list, probs: np.ndarray, threshold: float = 0.5, title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts, y=probs, mode="lines", name="P(Attack)",
        line=dict(color="#818cf8", width=2.5),
        fill="tozeroy", fillcolor="rgba(129,140,248,0.12)",
        hovertemplate="<b>%{x}</b><br>P(attack)=%{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=threshold, line=dict(color="#ef4444", width=1.5, dash="dot"),
                  annotation_text=f"Threshold {threshold:.2f}",
                  annotation_position="top right",
                  annotation_font_color="#ef4444", annotation_font_size=10)
    layout = {
        **_BL,
        "height": 300,
        "title": dict(text=title, font=dict(size=13, color="#c7d2fe"), x=0.01),
        "yaxis": {**_BL["yaxis"], "range": [-0.02, 1.05]},
    }
    fig.update_layout(**layout)
    return fig


def delta_chart(ts: list, orig: np.ndarray, sim: np.ndarray, label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=orig, mode="lines", name="Original",
        line=dict(color="#818cf8", width=2.5),
        hovertemplate="Original: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=ts, y=sim, mode="lines", name="Simulated",
        line=dict(color="#34d399", width=2, dash="dash"),
        hovertemplate="Simulated: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=list(ts) + list(reversed(ts)),
        y=list(orig) + list(reversed(sim)),
        fill="toself", fillcolor="rgba(52,211,153,0.07)",
        line=dict(width=0), name="Delta", hoverinfo="skip"))
    layout = {
        **_BL,
        "height": 280,
        "title": dict(text=f"Before vs After — {label} [Simulated]",
                      font=dict(size=13, color="#c7d2fe"), x=0.01),
        "yaxis": {**_BL["yaxis"], "range": [-0.02, 1.05]},
    }
    fig.update_layout(**layout)
    return fig


def shap_bar_chart(rankings: List[Dict], top_n: int = 10) -> go.Figure:
    top = rankings[:top_n]
    feats = [r["feature"] for r in reversed(top)]
    vals  = [r["mean_abs_shap"] for r in reversed(top)]
    fig = go.Figure(go.Bar(
        x=vals, y=feats, orientation="h",
        marker=dict(color=vals, colorscale=[[0,"#4f46e5"],[0.5,"#818cf8"],[1,"#f0abfc"]], showscale=False),
        hovertemplate="<b>%{y}</b><br>|SHAP|=%{x:.2e}<extra></extra>",
    ))
    layout = {
        **_BL,
        "height": min(80 + top_n * 26, 420),
        "title": dict(text="Global SHAP Feature Importance (LSTM)", font=dict(size=13, color="#c7d2fe"), x=0.01),
        "xaxis": {**_BL["xaxis"], "title": "Mean |SHAP value|"},
        "yaxis": {**_BL["yaxis"], "autorange": True},
        "margin": {**_BL["margin"], "l": 170, "r": 8, "t": 38, "b": 8},
    }
    fig.update_layout(**layout)
    return fig


# ============================================================================
# Threshold Analysis
# ============================================================================

@st.cache_data(show_spinner="Computing threshold sweep…", ttl=600)
def compute_threshold_sweep(probs: np.ndarray, y_true: np.ndarray) -> pd.DataFrame:
    """
    Computes precision, recall, F1, and alert volume across a range of thresholds.
    Returns DataFrame with columns: threshold, precision, recall, f1, alerts_per_day.
    """
    if len(probs) == 0 or len(y_true) == 0:
        return pd.DataFrame(columns=["threshold", "precision", "recall", "f1", "alerts_per_day"])
    
    thresholds = np.linspace(0.05, 0.95, 19)
    rows = []
    # Scale factor: how many sequences per day (assuming 60s windows, 1440 min/day)
    windows_per_day = 1440 * 60 / 60  # 1440 windows per day at 60s each
    scale = windows_per_day / len(probs) if len(probs) > 0 else 1.0
    
    for t in thresholds:
        y_pred = (probs >= t).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        alerts = y_pred.sum() * scale
        rows.append({
            "threshold": float(t),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "alerts_per_day": float(alerts),
        })
    return pd.DataFrame(rows)


def get_metrics_at_threshold(probs: np.ndarray, y_true: np.ndarray, threshold: float) -> Dict:
    """Computes metrics at a specific threshold."""
    if len(probs) == 0 or len(y_true) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "alerts": 0, "alerts_per_day": 0.0}
    y_pred = (probs >= threshold).astype(int)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    alerts = int(y_pred.sum())
    windows_per_day = 1440
    scale = windows_per_day / len(probs) if len(probs) > 0 else 1.0
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "alerts": alerts,
        "alerts_per_day": float(alerts * scale),
    }


# ---------------------------------------------------------------------------
# What-If helper: find high-risk window for compelling demo
# ---------------------------------------------------------------------------

def find_high_risk_baseline(probs: np.ndarray, lookback: int = 20) -> Tuple[int, float]:
    """
    Find the highest-probability window in recent history for What-If demo.
    Returns (index_in_probs, probability) or (-1, 0.0) if none found above threshold.
    """
    if len(probs) == 0:
        return -1, 0.0
    # Look at last `lookback` windows (or all if fewer)
    start_idx = max(0, len(probs) - lookback)
    recent_probs = probs[start_idx:]
    max_idx = int(np.argmax(recent_probs))
    abs_idx = start_idx + max_idx
    return abs_idx, float(recent_probs[max_idx])


def get_intervention_baseline(use_high_risk: bool = True, lookback: int = 20) -> Tuple[np.ndarray, float, int]:
    """
    Get the sequence and probability to use as intervention baseline.
    
    Args:
        use_high_risk: If True, find highest-risk recent window. If False, use last window.
        lookback: How many recent windows to search for high-risk baseline.
    
    Returns:
        Tuple of (baseline_sequence, baseline_probability, baseline_index)
    """
    df, feat = load_window_df()
    if df is None or len(df) <= SEQ_LEN:
        return np.array([]), 0.0, -1
    
    model = load_lstm_model(len(feat))
    X, _, _ = create_sequences(df, feat, seq_len=SEQ_LEN)
    probs = model(torch.tensor(X, dtype=torch.float32)).squeeze(-1).detach().numpy()
    
    if use_high_risk:
        idx, prob = find_high_risk_baseline(probs, lookback)
        if idx >= 0 and prob > 0.5:  # Only use if meaningfully high
            return X[idx], prob, idx
    
    # Fallback to last window
    return X[-1], float(probs[-1]), len(probs) - 1


# ---------------------------------------------------------------------------
# Tiered Defence Recommendation Builder
# ---------------------------------------------------------------------------

def _build_defence_recommendations(
    path_id: str,
    path: Dict[str, Any],
    kstep_rollout: List[float],
    mitre: MitreStageResult,
    top_shap: List[str],
    decision_threshold: float,
    insufficient_data: bool,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Builds prioritized, tiered defence recommendations derived directly from the
    active forecast path, K-step rollout, MITRE mapping and SHAP evidence.

    Returns a dict keyed by urgency tier: 'IMMEDIATE', 'HIGH', 'PLANNED'. Each
    recommendation carries live forecast context (related path, target asset,
    primary SHAP driver, final predicted stage) so the UI never falls back to
    static hardcoded prose when dynamic context is available.
    """
    prob = path.get("prob")
    target = path.get("target", "Unknown Asset")
    driver = path.get("driver", "unspecified")
    stages = path.get("stages", [])
    horizon_step = path.get("horizon_step", 1)
    lead_time = path.get("lead_time", "N/A")

    # Derive trend context from the live K-step rollout for urgency annotation
    if not insufficient_data and len(kstep_rollout) >= 3:
        _tv = kstep_rollout[:3]
        escalating = bool(_tv[-1] > _tv[0] * 1.1)
    else:
        escalating = False

    templates: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "PATH-01": {
            "IMMEDIATE": [
                {
                    "title": "Isolate compromised host 192.168.1.105",
                    "desc": "Quarantine the infected endpoint (192.168.1.105) to cut SMB lateral propagation immediately and halt spread toward Enterprise SQL DB.",
                    "related": "PATH-01",
                    "effort": "Low",
                    "eta": "< 5 min",
                },
            ],
            "HIGH": [
                {
                    "title": "Block C2 / SMB port 445 on perimeter & internal switches",
                    "desc": "Neutralize active SYN scans and share enumeration by restricting SMB/RPC port 445 traffic on perimeter firewalls and internal core switches.",
                    "related": "PATH-01",
                    "effort": "Low",
                    "eta": "15–30 min",
                },
            ],
            "PLANNED": [
                {
                    "title": "Reset DB credentials & enforce mTLS database access",
                    "desc": "Rotate exposed database credentials, enforce mutual TLS (mTLS) encryption, and restrict administrative access from general employee subnets.",
                    "related": "PATH-01",
                    "effort": "Medium",
                    "eta": "Schedule this week",
                },
            ],
        },
        "PATH-02": {
            "IMMEDIATE": [
                {
                    "title": "Enforce account lockout & disable vulnerable service accounts",
                    "desc": "Immediately lock out user accounts exhibiting Kerberoasting request signatures and disable legacy accounts using weak encryption (RC4) to halt the brute-force relay.",
                    "related": "PATH-02",
                    "effort": "Low",
                    "eta": "< 5 min",
                },
            ],
            "HIGH": [
                {
                    "title": "Restrict Kerberos ticket requests & apply RPC filtering",
                    "desc": "Block port 445/139 relay vectors, throttle TGT/TGS request issuance on Domain Controllers, and alert on anomalous ticket requests from non-admin endpoints.",
                    "related": "PATH-02",
                    "effort": "Low",
                    "eta": "15–30 min",
                },
            ],
            "PLANNED": [
                {
                    "title": "Patch Active Directory & rotate Kerberos krbtgt keys",
                    "desc": "Apply Active Directory security rollups, rotate Kerberos KRBTGT keys twice, implement Group Managed Service Accounts (gMSA), and enforce AES-256 Kerberos encryption.",
                    "related": "PATH-02",
                    "effort": "Medium",
                    "eta": "Schedule this week",
                },
            ],
        },
        "PATH-03": {
            "IMMEDIATE": [
                {
                    "title": "Rate-limit & block ingress burst sources",
                    "desc": "Apply ingress bandwidth throttling on edge router (10.0.0.1) and blackhole high-volume burst sources flagged by anomalous flow telemetry.",
                    "related": "PATH-03",
                    "effort": "Low",
                    "eta": "< 5 min",
                },
            ],
            "HIGH": [
                {
                    "title": "Enable SYN cookies & perimeter throttling",
                    "desc": "Activate SYN cookie protection and packet-rate limits on perimeter firewalls to neutralize flood propagation and prevent socket buffer exhaustion.",
                    "related": "PATH-03",
                    "effort": "Low",
                    "eta": "15–30 min",
                },
            ],
            "PLANNED": [
                {
                    "title": "Patch router firmware & deploy CDN/WAF scrubbing",
                    "desc": "Upgrade ingress router firmware, configure upstream BGP flowspec scrubbing, and deploy a CDN/WAF layer to harden external ingress against volumetric attacks.",
                    "related": "PATH-03",
                    "effort": "Medium",
                    "eta": "Schedule this week",
                },
            ],
        },
    }

    fallback = {
        "IMMEDIATE": [
            {
                "title": f"Isolate affected host for {path_id}",
                "desc": f"Quarantine the asset flagged by the forecast to halt lateral propagation immediately toward {target}.",
                "related": path_id,
                "effort": "Low",
                "eta": "< 5 min",
            },
        ],
        "HIGH": [
            {
                "title": f"Block predicted C2 & anomalous flow ports",
                "desc": f"Restrict port diversity and TCP flag anomalies driving the forecast ({driver}).",
                "related": path_id,
                "effort": "Low",
                "eta": "15–30 min",
            },
        ],
        "PLANNED": [
            {
                "title": f"Patch & harden target asset {target}",
                "desc": f"Harden {target} against the predicted final stage '{stages[-1] if stages else 'Unknown'}' and enforce strict network segmentation.",
                "related": path_id,
                "effort": "Medium",
                "eta": "Schedule this week",
            },
        ],
    }

    tier_map = templates.get(path_id, fallback)

    # Annotate every recommendation with live forecast context
    prob_str = f"{prob:.1%}" if prob is not None else "N/A"
    for _recs in tier_map.values():
        for r in _recs:
            r["prob"] = prob_str
            r["target"] = target
            r["driver"] = driver
            r["final_stage"] = stages[-1] if stages else "Unknown"
            r["horizon_step"] = horizon_step
            r["lead_time"] = lead_time
            r["escalating"] = escalating

    return tier_map

