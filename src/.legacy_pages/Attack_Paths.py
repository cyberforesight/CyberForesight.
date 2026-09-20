"""
CyberForeSight AI — Attack Paths & MITRE Mapping
Modules 7 (K-Step Attack Paths), 8 (MITRE ATT&CK), 11 (Asset Risk), 12 (Attack Timeline)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src._dashboard_core import (
    MITRE_COLOR, RISK_COLOR, SEQ_LEN,
    inject_css, kpi, page_header, sec, _BL, get_layout,
    load_window_df, load_shap_data,
    get_timeline, get_kstep, get_current_mitre,
    get_rollout_mitre,
)
from src.mitre_map import map_to_mitre_stage

st.set_page_config(page_title="CyberForeSight AI · Attack Paths", page_icon="🕸️", layout="wide")
inject_css()
page_header("🕸️", "Attack Paths & MITRE ATT&CK Mapping",
            "Modules 7 · 8 · 11 · 12 — Real K-Step Rollout + Heuristic Stage Mapping")

# ─── Load ────────────────────────────────────────────────────────────────────
try:
    df, feat = load_window_df()
except FileNotFoundError:
    st.error("Window features not found. Run `python src/features.py` first.")
    st.stop()

shap_data      = load_shap_data()
shap_rankings  = shap_data.get("lstm_deep_explainer", {}).get("feature_rankings", [])
top_shap_names = [r["feature"] for r in shap_rankings[:5]]

ts, probs      = get_timeline()
current_prob   = float(probs[-1])
mitre_current  = get_current_mitre(current_prob, top_shap_names)

with st.sidebar:
    st.markdown("<div class='sec-hdr'>⚙️ Controls</div>", unsafe_allow_html=True)
    threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05, key="ap_thresh")
    k_horizon = st.slider("K-Step Horizon", 3, 10, 5, 1, key="ap_k")

with st.spinner("Running K-step rollout for attack path analysis…"):
    rollout_probs  = get_kstep(k=k_horizon)

current_row   = df[feat].iloc[-1]
rollout_mitre = get_rollout_mitre(rollout_probs, current_row, feat)

# ═══════════════════════════════════════════════════════════════════════════
# MODULE 7: Ranked Probable Attack Paths (K-step rollout)
# ═══════════════════════════════════════════════════════════════════════════
sec("🗺️ Module 7 — Ranked Probable Attack Paths (K-Step Rollout)")

steps_lbl = [f"t+{i}" for i in range(1, k_horizon + 1)]

# Build path card table
RISK_ICON = {"Low": "🟢", "Medium": "🟡", "High": "🔴", "Critical": "🔴"}
path_rows = []
for i, (prob, mr) in enumerate(zip(rollout_probs, rollout_mitre)):
    path_rows.append({
        "Step":        steps_lbl[i],
        "P(Attack)":   f"{prob:.4f}",
        "MITRE Stage": mr.stage,
        "Tactic":      mr.tactic_id,
        "Technique":   f"{mr.technique_id}: {mr.technique_name}",
        "Risk":        f"{RISK_ICON.get(mr.risk_level,'?')} {mr.risk_level}",
        "Confidence":  f"{mr.confidence:.0%}",
    })

path_df = pd.DataFrame(path_rows)
st.dataframe(path_df, use_container_width=True, hide_index=True)

# K-step path chart with colour-coded MITRE stages
stage_cols = [MITRE_COLOR.get(mr.stage, "#818cf8") for mr in rollout_mitre]
fig_path = go.Figure()
fig_path.add_trace(go.Bar(
    x=steps_lbl, y=rollout_probs,
    marker=dict(color=stage_cols, line=dict(width=1, color="#0a0e1a")),
    text=[f"{p:.3f}" for p in rollout_probs],
    textposition="outside",
    textfont=dict(color="#e2e8f0", size=11, family="JetBrains Mono"),
    name="P(Attack)",
    hovertemplate="<b>%{x}</b><br>P(attack)=%{y:.4f}<extra></extra>",
))
fig_path.add_hline(y=threshold, line=dict(color="#ef4444", width=1.5, dash="dot"),
                   annotation_text=f"Threshold {threshold:.2f}",
                   annotation_font_color="#ef4444", annotation_font_size=10)

# MITRE stage labels
for i, mr in enumerate(rollout_mitre):
    fig_path.add_annotation(
        x=steps_lbl[i], y=rollout_probs[i] + 0.08,
        text=mr.stage[:8], showarrow=False,
        font=dict(size=9, color=MITRE_COLOR.get(mr.stage, "#818cf8"), family="Inter"),
    )

fig_path.update_layout(
    **get_layout(
        height=320,
        title=dict(text="K-Step Attack Path — P(Attack) per Horizon Step (colour = MITRE Stage)",
                   font=dict(size=13, color="#c7d2fe"), x=0.01),
        yaxis=dict(range=[-0.02, 1.25]),
        xaxis=dict(title="Forecast Horizon"),
    )
)
st.plotly_chart(fig_path, use_container_width=True)

# Path narratives
col_p1, col_p2, col_p3 = st.columns(3)
sorted_paths = sorted(enumerate(zip(rollout_probs, rollout_mitre)), key=lambda x: x[1][0], reverse=True)
path_colors  = ["#ef4444", "#f59e0b", "#818cf8"]
path_alerts  = [st.error, st.warning, st.info]
for rank, (idx, (prob, mr)) in enumerate(sorted_paths[:3]):
    conf = int(prob * 100)
    msg  = (f"**Path {rank+1} (Conf: {conf}%):** "
            f"{steps_lbl[idx]} → {mr.stage} [{mr.tactic_id}] · {mr.technique_id}: {mr.technique_name} "
            f"| {RISK_ICON.get(mr.risk_level,'?')} {mr.risk_level.upper()}")
    path_alerts[rank](msg)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# MODULE 8: MITRE ATT&CK Stage Mapping — Current Window
# ═══════════════════════════════════════════════════════════════════════════
sec("🎯 Module 8 — MITRE ATT&CK Stage Mapping (Current Window)")

col_m1, col_m2 = st.columns([1, 1], gap="large")
stage_col = MITRE_COLOR.get(mitre_current.stage, "#818cf8")
risk_col  = RISK_COLOR.get(mitre_current.risk_level, "#f59e0b")

with col_m1:
    st.markdown(
        f"""<div style='background:linear-gradient(145deg,#1e293b,#0f172a);
            border:1px solid {stage_col}44;border-radius:14px;padding:1.4rem'>
            <div style='font-size:0.65rem;color:#64748b;text-transform:uppercase;
                letter-spacing:0.1em;margin-bottom:0.8rem;font-weight:700'>Current Window Mapping</div>
            <div style='font-size:1.5rem;font-weight:700;color:{stage_col};margin-bottom:0.5rem'>
                {mitre_current.stage}
            </div>
            <div style='font-size:0.77rem;color:#94a3b8;margin-bottom:0.25rem'>
                <b style='color:#c7d2fe'>{mitre_current.tactic_id}</b> &middot; {mitre_current.technique_id}
            </div>
            <div style='font-size:0.75rem;color:#e2e8f0;margin-bottom:0.7rem'>{mitre_current.technique_name}</div>
            <div style='display:inline-block;background:{risk_col}22;border:1px solid {risk_col}55;
                color:{risk_col};padding:0.2rem 0.7rem;border-radius:20px;font-size:0.68rem;
                font-weight:700;margin-bottom:0.8rem'>
                {mitre_current.risk_level.upper()} RISK &middot; Confidence {mitre_current.confidence:.0%}
            </div>
            <div style='font-size:0.72rem;color:#94a3b8;line-height:1.55'>
                &#128737; {mitre_current.defense_recommendation}
            </div>
        </div>""",
        unsafe_allow_html=True,
    )
    if mitre_current.matched_rules:
        with st.expander("📋 Matched Heuristic Rules", expanded=False):
            for rule in mitre_current.matched_rules:
                st.markdown(f"<small style='color:#94a3b8'>&bull; {rule}</small>", unsafe_allow_html=True)

with col_m2:
    # MITRE stage progression donut
    stage_counts = {}
    for mr in rollout_mitre:
        stage_counts[mr.stage] = stage_counts.get(mr.stage, 0) + 1
    if mitre_current.stage not in stage_counts:
        stage_counts[mitre_current.stage] = 1

    fig_donut = go.Figure(go.Pie(
        labels=list(stage_counts.keys()),
        values=list(stage_counts.values()),
        hole=0.6,
        marker=dict(colors=[MITRE_COLOR.get(s, "#818cf8") for s in stage_counts],
                    line=dict(color="#0a0e1a", width=2)),
        textinfo="label+percent",
        textfont=dict(size=11, color="#e2e8f0"),
        hovertemplate="<b>%{label}</b><br>Steps: %{value}<br>%{percent}<extra></extra>",
    ))
    fig_donut.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter,sans-serif", color="#94a3b8", size=11),
        showlegend=False, height=280,
        title=dict(text="MITRE Stage Distribution (K-Step Rollout)",
                   font=dict(size=12, color="#c7d2fe"), x=0.01),
        annotations=[dict(text=f"<b>{mitre_current.stage[:6]}</b>", x=0.5, y=0.5,
                           font=dict(size=14, color=stage_col), showarrow=False)],
        margin=dict(l=8, r=8, t=38, b=8),
    )
    st.plotly_chart(fig_donut, use_container_width=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# MODULE 11: Asset Criticality Assessment
# ═══════════════════════════════════════════════════════════════════════════
sec("🗄️ Module 11 — Asset Criticality Assessment Matrix")

# Rule-based criticality from real feature values
rst   = float(current_row.get("rst_flag_cnt_sum", 0))
syn   = float(current_row.get("syn_flag_cnt_sum", 0))
byts  = float(current_row.get("flow_byts_s_mean", 0))
ports = float(current_row.get("dst_port_nunique", 0))
pkts  = float(current_row.get("flow_pkts_s_mean", 0))

def _crit_score(asset: str) -> tuple[str, float, str]:
    """Returns (label, score 0-1, reason) for each asset."""
    if asset == "Core Database":
        score = min(1.0, 0.3 + max(rst, 0) * 0.3 + max(byts, 0) * 0.2)
        reason = "High RST flags + elevated byte transfer" if rst > 0.5 else "Moderate exfiltration risk"
        return ("CRITICAL" if score > 0.7 else "HIGH" if score > 0.4 else "MEDIUM", round(score, 2), reason)
    elif asset == "Domain Controller":
        score = min(1.0, 0.2 + max(syn, 0) * 0.35 + max(rst, 0) * 0.25)
        reason = "SYN flood signature detected" if syn > 0.5 else "Brute-force credential risk"
        return ("CRITICAL" if score > 0.7 else "HIGH" if score > 0.4 else "MEDIUM", round(score, 2), reason)
    elif asset == "File Server":
        score = min(1.0, 0.1 + max(byts, 0) * 0.4 + max(pkts, 0) * 0.2)
        reason = "High bandwidth exfiltration risk" if byts > 0.5 else "Elevated packet rate"
        return ("HIGH" if score > 0.5 else "MEDIUM" if score > 0.2 else "LOW", round(score, 2), reason)
    elif asset == "Application Server":
        score = min(1.0, 0.1 + max(ports, 0) * 0.35 + max(pkts, 0) * 0.15)
        reason = "Port scanning activity detected" if ports > 0.5 else "Low exposure"
        return ("HIGH" if score > 0.55 else "MEDIUM" if score > 0.25 else "LOW", round(score, 2), reason)
    else:  # Employee PC
        score = min(0.5, 0.05 + current_prob * 0.3)
        return ("MEDIUM" if score > 0.3 else "LOW", round(score, 2), "Lateral movement endpoint risk")

ASSET_CRIT_COLOR = {"CRITICAL": "#dc2626", "HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#22c55e"}
assets = ["Core Database", "Domain Controller", "File Server", "Application Server", "Employee PC"]
asset_results = {a: _crit_score(a) for a in assets}

col_a1, col_a2 = st.columns([1, 1])
with col_a1:
    for asset, (label, score, reason) in asset_results.items():
        col_color = ASSET_CRIT_COLOR[label]
        st.markdown(
            f"<div style='background:rgba(15,23,42,0.8);border:1px solid {col_color}33;"
            f"border-radius:10px;padding:0.8rem 1rem;margin-bottom:0.5rem;display:flex;"
            f"justify-content:space-between;align-items:center'>"
            f"<div>"
            f"<div style='font-size:0.82rem;font-weight:600;color:#e2e8f0'>{asset}</div>"
            f"<div style='font-size:0.68rem;color:#64748b;margin-top:0.1rem'>{reason}</div>"
            f"</div>"
            f"<div style='background:{col_color}22;border:1px solid {col_color}55;color:{col_color};"
            f"padding:0.2rem 0.7rem;border-radius:16px;font-size:0.7rem;font-weight:700'>{label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

with col_a2:
    fig_asset = go.Figure(go.Bar(
        x=[v[1] for v in asset_results.values()],
        y=list(asset_results.keys()),
        orientation="h",
        marker=dict(color=[ASSET_CRIT_COLOR[v[0]] for v in asset_results.values()],
                    line=dict(width=1, color="#0a0e1a")),
        text=[f"{v[0]}  {v[1]:.2f}" for v in asset_results.values()],
        textposition="inside",
        textfont=dict(color="#fff", size=11, family="JetBrains Mono"),
        hovertemplate="<b>%{y}</b><br>Risk Score=%{x:.2f}<extra></extra>",
    ))
    fig_asset.update_layout(
        **get_layout(
            height=280,
            title=dict(text="Asset Risk Score (Computed from Real Features)", font=dict(size=12, color="#c7d2fe"), x=0.01),
            xaxis=dict(range=[0, 1.1], title="Risk Score (0–1)"),
            yaxis=dict(autorange=True),
            margin=dict(l=140, r=8, t=38, b=8),
        )
    )
    st.plotly_chart(fig_asset, use_container_width=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# MODULE 12: Attack Graph & Timeline
# ═══════════════════════════════════════════════════════════════════════════
sec("📈 Module 12 — Attack Graph & Timeline (Observed + Predicted)")

# Overlay real window labels on the probability timeline
df_plot = df[["window_timestamp", "is_attack_window"]].copy() if "window_timestamp" in df.columns else df[["is_attack_window"]].copy()
df_plot["window_timestamp"] = pd.to_datetime(df_plot.get("window_timestamp", range(len(df))), errors="coerce")
df_plot = df_plot.iloc[SEQ_LEN:]   # align with inference output
df_plot["predicted_prob"] = probs
df_plot["predicted_label"] = (probs >= threshold).astype(int)

fig_atk = go.Figure()
fig_atk.add_trace(go.Scatter(
    x=ts, y=probs, mode="lines", name="P(Attack) [LSTM]",
    line=dict(color="#818cf8", width=2),
    fill="tozeroy", fillcolor="rgba(129,140,248,0.08)",
))
# Mark observed attack windows
atk_mask = df_plot["is_attack_window"].values == 1
atk_ts   = [t for t, m in zip(ts, atk_mask) if m]
atk_p    = probs[atk_mask] if len(atk_mask) == len(probs) else probs[:len(atk_mask)][atk_mask]
if len(atk_ts):
    fig_atk.add_trace(go.Scatter(
        x=atk_ts, y=atk_p, mode="markers", name="Observed Attack",
        marker=dict(color="#ef4444", size=7, symbol="circle", line=dict(width=1.5, color="#dc2626")),
        hovertemplate="<b>%{x}</b><br>Observed Attack · P=%{y:.3f}<extra></extra>",
    ))
# Mark predicted attack windows above threshold
pred_atk_ts = [t for t, p in zip(ts, probs) if p >= threshold]
pred_atk_p  = [p for p in probs if p >= threshold]
if pred_atk_ts:
    fig_atk.add_trace(go.Scatter(
        x=pred_atk_ts, y=pred_atk_p, mode="markers", name="Predicted Attack",
        marker=dict(color="#fbbf24", size=5, symbol="star", opacity=0.8),
        hovertemplate="<b>%{x}</b><br>Predicted · P=%{y:.3f}<extra></extra>",
    ))
fig_atk.add_hline(y=threshold, line=dict(color="#ef4444", width=1.5, dash="dot"),
                  annotation_text=f"Threshold {threshold:.2f}",
                  annotation_font_color="#ef4444", annotation_font_size=10)
fig_atk.update_layout(
    **get_layout(
        height=320,
        title=dict(text="Attack Graph & Timeline — Observed (red) + LSTM Predicted (gold)",
                   font=dict(size=13, color="#c7d2fe"), x=0.01),
        yaxis=dict(range=[-0.02, 1.08]),
    )
)
st.plotly_chart(fig_atk, use_container_width=True)

# Footer
st.divider()
st.markdown(
    "<div style='text-align:center;color:#334155;font-size:0.65rem;padding:0.4rem'>"
    "CyberForeSight AI · Modules 7–8–11–12 · Real CIC-IDS2018 Data"
    "</div>",
    unsafe_allow_html=True,
)