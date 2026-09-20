"""
CyberForeSight AI — Feature & Window Engine Workspace
Modules 2 (Feature Extraction), 3 (Time-Window Engine), 4 (State Vectors)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src._dashboard_core import (
    inject_css, kpi, page_header, sec, _BL, get_layout,
    load_window_df, load_metadata,
)

st.set_page_config(page_title="CyberForeSight AI · Data & Features", page_icon="🎛️", layout="wide")
inject_css()
page_header("🎛️", "Feature Extraction & Window Engine", "Modules 2 · 3 · 4 — Real CIC-IDS2018 Data")

# ─── Load data ────────────────────────────────────────────────────────────────
try:
    df, feat = load_window_df()
except FileNotFoundError:
    st.error("Window features not found. Run `python src/features.py` first.")
    st.stop()

# ─── Feature registry classification ────────────────────────────────────────
FLOW_LEVEL = [f for f in feat if any(k in f for k in
    ["flow_duration","tot_fwd","tot_bwd","totlen_fwd","totlen_bwd","flow_byts","flow_pkts","flow_iat","flow_count"])]
PKT_DERIVED = [f for f in feat if any(k in f for k in
    ["pkt_len","fwd_pkt_len","bwd_pkt_len","pkt_len_max","pkt_len_mean","pkt_len_min"])]
TCP_FLAGS   = [f for f in feat if "flag" in f or "win" in f]
TIMING      = [f for f in feat if any(k in f for k in ["active","idle","iat"])]
OTHER       = [f for f in feat if f not in FLOW_LEVEL + PKT_DERIVED + TCP_FLAGS + TIMING]

# ═══════════════════════════════════════════════════════════════════════════
# MODULE 2: Feature Extraction Registry
# ═══════════════════════════════════════════════════════════════════════════
sec("⚙️ Module 2 — Feature Extraction Registry")

col_reg1, col_reg2 = st.columns(2)

with col_reg1:
    st.markdown(
        "<div style='background:rgba(99,102,241,0.07);border:1px solid rgba(99,102,241,0.2);"
        "border-radius:12px;padding:1rem;margin-bottom:0.8rem'>"
        "<div style='color:#818cf8;font-size:0.68rem;font-weight:700;text-transform:uppercase;"
        "letter-spacing:0.1em;margin-bottom:0.6rem'>🌊 Flow-Level Metrics</div>"
        + "".join(f"<div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;"
                  f"color:#c7d2fe;padding:0.2rem 0;border-bottom:1px solid rgba(51,65,85,0.4)'>{f}</div>"
                  for f in FLOW_LEVEL)
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='background:rgba(168,85,247,0.07);border:1px solid rgba(168,85,247,0.2);"
        "border-radius:12px;padding:1rem'>"
        "<div style='color:#a855f7;font-size:0.68rem;font-weight:700;text-transform:uppercase;"
        "letter-spacing:0.1em;margin-bottom:0.6rem'>⏱️ Timing Metrics</div>"
        + "".join(f"<div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;"
                  f"color:#c7d2fe;padding:0.2rem 0;border-bottom:1px solid rgba(51,65,85,0.4)'>{f}</div>"
                  for f in TIMING)
        + "</div>",
        unsafe_allow_html=True,
    )

with col_reg2:
    st.markdown(
        "<div style='background:rgba(56,189,248,0.07);border:1px solid rgba(56,189,248,0.2);"
        "border-radius:12px;padding:1rem;margin-bottom:0.8rem'>"
        "<div style='color:#38bdf8;font-size:0.68rem;font-weight:700;text-transform:uppercase;"
        "letter-spacing:0.1em;margin-bottom:0.6rem'>📦 Packet-Derived Metrics</div>"
        + "".join(f"<div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;"
                  f"color:#c7d2fe;padding:0.2rem 0;border-bottom:1px solid rgba(51,65,85,0.4)'>{f}</div>"
                  for f in PKT_DERIVED)
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='background:rgba(239,68,68,0.07);border:1px solid rgba(239,68,68,0.2);"
        "border-radius:12px;padding:1rem'>"
        "<div style='color:#ef4444;font-size:0.68rem;font-weight:700;text-transform:uppercase;"
        "letter-spacing:0.1em;margin-bottom:0.6rem'>🚩 TCP Flags & Window</div>"
        + "".join(f"<div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;"
                  f"color:#c7d2fe;padding:0.2rem 0;border-bottom:1px solid rgba(51,65,85,0.4)'>{f}</div>"
                  for f in TCP_FLAGS)
        + "</div>",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
c_f1, c_f2, c_f3, c_f4 = st.columns(4)
with c_f1:
    st.markdown(kpi("Total Features", f"{len(feat)}", "State vector D", "#818cf8"), unsafe_allow_html=True)
with c_f2:
    st.markdown(kpi("Flow-Level", f"{len(FLOW_LEVEL)}", color="#818cf8"), unsafe_allow_html=True)
with c_f3:
    st.markdown(kpi("Packet-Derived", f"{len(PKT_DERIVED)}", color="#38bdf8"), unsafe_allow_html=True)
with c_f4:
    st.markdown(kpi("TCP Flags & Timing", f"{len(TCP_FLAGS) + len(TIMING)}", color="#ef4444"), unsafe_allow_html=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# MODULE 3: 60-Second Time-Window Engine
# ═══════════════════════════════════════════════════════════════════════════
sec("⏳ Module 3 — 60s Time-Window Aggregator")

ts_col = "window_timestamp"
if ts_col in df.columns:
    df_ts = df.copy()
    df_ts[ts_col] = pd.to_datetime(df_ts[ts_col], errors="coerce")
    df_ts = df_ts.dropna(subset=[ts_col])

    col_t1, col_t2 = st.columns([2, 1])

    with col_t1:
        # Windows per date
        df_ts["date"] = df_ts[ts_col].dt.date
        wnd_per_day = df_ts.groupby("date").size().reset_index(name="Windows")
        fig_day = px.bar(wnd_per_day, x="date", y="Windows",
                         title="60s Windows per Day (Real Dataset)",
                         color_discrete_sequence=["#818cf8"])
        fig_day.update_layout(**_BL, height=280,
            title=dict(text="60s Windows per Day", font=dict(size=13, color="#c7d2fe"), x=0.01))
        st.plotly_chart(fig_day, use_container_width=True)

    with col_t2:
        n_windows = len(df_ts)
        n_attack  = int(df_ts["is_attack_window"].sum()) if "is_attack_window" in df_ts.columns else 0
        n_benign  = n_windows - n_attack
        st.markdown(kpi("Total Windows", f"{n_windows:,}", "60s buckets", "#818cf8"), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(kpi("Attack Windows", f"{n_attack:,}", f"{100*n_attack/max(n_windows,1):.1f}%", "#ef4444"), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(kpi("Benign Windows", f"{n_benign:,}", f"{100*n_benign/max(n_windows,1):.1f}%", "#22c55e"), unsafe_allow_html=True)

    # Attack vs Benign timeline
    if "is_attack_window" in df_ts.columns:
        fig_atk = go.Figure()
        fig_atk.add_trace(go.Scatter(
            x=df_ts[ts_col], y=df_ts["is_attack_window"],
            mode="markers", name="Window Label",
            marker=dict(
                color=["#ef4444" if v else "#22c55e" for v in df_ts["is_attack_window"]],
                size=5, opacity=0.7,
            ),
            hovertemplate="<b>%{x}</b><br>Attack=%{y}<extra></extra>",
        ))
        fig_atk.update_layout(
            **get_layout(
                height=180,
                title=dict(text="Window Labels Over Time (1=Attack, 0=Benign)", font=dict(size=12, color="#c7d2fe"), x=0.01),
                yaxis=dict(range=[-0.15, 1.2]),
            )
        )
        st.plotly_chart(fig_atk, use_container_width=True)
else:
    st.info("Timestamp column not found in window dataset.")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# MODULE 4: Network State Vector Inspector
# ═══════════════════════════════════════════════════════════════════════════
sec("🧬 Module 4 — Network State Vector Inspector  S_t = [f₁, f₂, ..., f₃₆]")

col_sv1, col_sv2 = st.columns([1, 2])

with col_sv1:
    window_idx = st.slider("Select Window (S_t)", 0, len(df) - 1, len(df) - 1, 1)

with col_sv2:
    selected_row = df[feat].iloc[window_idx]
    ts_val = df["window_timestamp"].iloc[window_idx] if "window_timestamp" in df.columns else f"Window #{window_idx}"
    is_atk = int(df["is_attack_window"].iloc[window_idx]) if "is_attack_window" in df.columns else "?"
    atk_color = "#ef4444" if is_atk == 1 else "#22c55e"
    atk_label = "⚠️ ATTACK" if is_atk == 1 else "✅ BENIGN"
    st.markdown(
        f"<div style='display:flex;gap:1rem;align-items:center;margin-bottom:0.5rem'>"
        f"<div style='font-family:JetBrains Mono,monospace;font-size:0.8rem;color:#64748b'>{ts_val}</div>"
        f"<div style='background:{atk_color}22;border:1px solid {atk_color}55;color:{atk_color};"
        f"padding:0.15rem 0.7rem;border-radius:20px;font-size:0.7rem;font-weight:700'>{atk_label}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# Feature values as horizontal bars
vec_df = pd.DataFrame({"Feature": feat, "Normalized Value": selected_row.values})
fig_vec = px.bar(vec_df, x="Normalized Value", y="Feature", orientation="h",
                 color="Normalized Value",
                 color_continuous_scale=[[0, "#4f46e5"], [0.5, "#818cf8"], [1, "#f0abfc"]],
                 title=f"State Vector S_t — Window {window_idx} (Z-score normalised)")
fig_vec.update_layout(
    **get_layout(
        height=min(120 + len(feat) * 20, 700),
        title=dict(text=f"State Vector S_t — Window {window_idx}", font=dict(size=13, color="#c7d2fe"), x=0.01),
        yaxis=dict(autorange=True),
        margin=dict(l=180, r=8, t=38, b=8),
        coloraxis_showscale=False,
    )
)
st.plotly_chart(fig_vec, use_container_width=True)

# Feature distribution summary
with st.expander("📋 Feature Statistics Summary (All Windows)", expanded=False):
    summary = df[feat].describe().T.round(4)
    st.dataframe(summary, use_container_width=True)

# Footer
st.divider()
st.markdown(
    "<div style='text-align:center;color:#334155;font-size:0.65rem;padding:0.4rem'>"
    "CyberForeSight AI · Modules 2–4 · Real CIC-IDS2018 Data"
    "</div>",
    unsafe_allow_html=True,
)