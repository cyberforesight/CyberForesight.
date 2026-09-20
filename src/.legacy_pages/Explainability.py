"""
CyberForeSight AI — Neural Explainability & What-If Simulation Command
Modules 9 (SHAP Feature Impact), 10 (Grounded LLM Threat Prose), 
13 (What-If Sandbox), 14 (Side-by-Side Delta), 15 (Automated Defense Recommendation)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src._dashboard_core import (
    MITRE_COLOR, RISK_COLOR, SEQ_LEN,
    ISOLATION_FEATURES, SMB_FEATURES,
    inject_css, kpi, page_header, sec, _BL,
    load_window_df, load_shap_data,
    get_timeline, get_current_mitre,
    run_intervention, shap_bar_chart, delta_chart,
)

st.set_page_config(page_title="CyberForeSight AI · Explainability & Simulation", page_icon="🧠", layout="wide")
inject_css()
page_header("🧠", "Neural Explainability & What-If Simulation Command",
            "Modules 9 · 10 · 13 · 14 · 15 — Real SHAP Attributions + Real Intervention Rollout")

# ─── Load data ────────────────────────────────────────────────────────────────
try:
    df, feat = load_window_df()
except FileNotFoundError:
    st.error("Window features not found. Run `python src/features.py` first.")
    st.stop()

shap_data = load_shap_data()
shap_rankings = shap_data.get("lstm_deep_explainer", {}).get("feature_rankings", [])
if not shap_rankings:
    st.warning("SHAP feature rankings not found in models/shap_summary.json. Run `python src/explain.py` first.")

ts, probs = get_timeline()
current_prob = float(probs[-1])
top_shap_names = [r["feature"] for r in shap_rankings[:5]] if shap_rankings else []
mitre_current = get_current_mitre(current_prob, top_shap_names)

# ─── Sidebar Controls ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='sec-hdr'>⚙️ Explainability Settings</div>", unsafe_allow_html=True)
    top_n = st.slider("Top SHAP Features", min_value=5, max_value=25, value=12, step=1)
    st.divider()
    st.markdown("<div class='sec-hdr'>🎮 Sandbox Quick Presets</div>", unsafe_allow_html=True)
    sim_host_iso = st.checkbox("Simulate Host Isolation", value=False,
                               help="Zeroes out flow counts, packet volumes, and byte transfer rates")
    sim_smb = st.checkbox("Simulate Restrict SMB", value=False,
                          help="Zeroes out destination port entropy, SYN/RST/PSH flags, and window buffers")
    custom_zero = st.multiselect(
        "Custom Feature Nullification",
        options=feat,
        default=[],
        help="Select additional features to suppress in the world model state vector"
    )

# ═══════════════════════════════════════════════════════════════════════════
# TOP SUMMARY KPI ROW
# ═══════════════════════════════════════════════════════════════════════════
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(kpi("Current Risk P(Attack)", f"{current_prob:.3f}",
                    f"Threshold: 0.50",
                    color=RISK_COLOR.get(mitre_current.risk_level, "#818cf8")),
                unsafe_allow_html=True)
with c2:
    st.markdown(kpi("MITRE ATT&CK Stage", mitre_current.stage,
                    f"Tactic: {mitre_current.tactic_id}",
                    color=MITRE_COLOR.get(mitre_current.stage, "#818cf8")),
                unsafe_allow_html=True)
with c3:
    st.markdown(kpi("Primary Neural Driver",
                    top_shap_names[0] if top_shap_names else "N/A",
                    f"Mean |SHAP|: {shap_rankings[0]['mean_abs_shap']:.2e}" if shap_rankings else "",
                    color="#f0abfc"),
                unsafe_allow_html=True)
with c4:
    num_rules = len(mitre_current.matched_rules)
    st.markdown(kpi("Active Heuristic Rules", f"{num_rules} Triggered",
                    f"Confidence: {mitre_current.confidence * 100:.1f}%",
                    color="#38bdf8"),
                unsafe_allow_html=True)

st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# MODULES 9 & 10: SHAP Feature Importance & Grounded Threat Explanation
# ═══════════════════════════════════════════════════════════════════════════
sec("📊 Modules 9 & 10 — SHAP Feature Impact & Grounded Threat Diagnostic")

col_shap, col_llm = st.columns([1.3, 1.0])

with col_shap:
    st.markdown(
        "<div style='font-size:0.8rem;color:#94a3b8;margin-bottom:0.6rem'>"
        "Global DeepExplainer attributions computed over the PyTorch LSTM World Model "
        f"across {len(feat)} normalized window features.</div>",
        unsafe_allow_html=True
    )
    if shap_rankings:
        fig_shap = shap_bar_chart(shap_rankings, top_n=top_n)
        st.plotly_chart(fig_shap, use_container_width=True)
    else:
        st.info("No SHAP summary data available.")

with col_llm:
    st.markdown(
        "<div style='font-size:0.8rem;color:#94a3b8;margin-bottom:0.6rem'>"
        "Analytical threat narrative synthesized from the current state vector, "
        "MITRE classification heuristics, and top SHAP neural drivers.</div>",
        unsafe_allow_html=True
    )
    
    # Generate deterministic, grounded analytical prose
    stage = mitre_current.stage
    tech_name = mitre_current.technique_name
    tactic_id = mitre_current.tactic_id
    risk = mitre_current.risk_level
    
    lead_feats = top_shap_names[:3]
    lead_str = ", ".join(f"`{f}`" for f in lead_feats) if lead_feats else "traffic anomalies"
    
    if current_prob >= 0.5:
        threat_status = f"<span style='color:#ef4444;font-weight:700;'>CRITICAL ANOMALY DETECTED</span>"
        prose_summary = (
            f"The LSTM World Model forecasts a high likelihood of impending attack progression "
            f"(P={current_prob:.3f}), corroborating an active **{stage}** phase under **{tech_name}** ({tactic_id}). "
            f"Neural attribution analysis confirms that model certainty is predominantly influenced by "
            f"deviations in {lead_str}. Elevated TCP teardown spikes and abnormal window buffer metrics "
            f"indicate active adversary probing or lateral propagation."
        )
    else:
        threat_status = f"<span style='color:#10b981;font-weight:700;'>NOMINAL TELEMETRY</span>"
        prose_summary = (
            f"The network is operating within baseline parameters (P={current_prob:.3f}), "
            f"corresponding to **{stage}**. Feature variance remains dominated by normal background traffic. "
            f"Top SHAP features {lead_str} show minimal adversarial attribution."
        )
        
    st.markdown(
        f"""
        <div style='background:linear-gradient(145deg, #1e293b, #0f172a);
                    border:1px solid rgba(99,102,241,0.3); border-radius:12px;
                    padding:1.2rem; margin-bottom:1rem;'>
            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.8rem;'>
                <span style='font-size:0.75rem; font-weight:700; color:#818cf8; text-transform:uppercase; letter-spacing:0.08em;'>
                    Diagnostic Threat Narrative
                </span>
                <span style='font-size:0.75rem;'>{threat_status}</span>
            </div>
            <div style='font-size:0.85rem; color:#e2e8f0; line-height:1.6; margin-bottom:1rem;'>
                {prose_summary}
            </div>
            <div style='background:rgba(15,23,42,0.8); border-radius:8px; padding:0.8rem; border-left:3px solid {MITRE_COLOR.get(stage, "#818cf8")};'>
                <div style='font-size:0.7rem; color:#94a3b8; text-transform:uppercase; font-weight:600;'>Identified ATT&CK Context</div>
                <div style='font-size:0.85rem; color:#f8fafc; font-weight:600; margin-top:0.2rem;'>
                    {stage} · {tech_name} ({mitre_current.technique_id})
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Detailed attribution breakdown table
    if shap_rankings:
        st.markdown("<div style='font-size:0.75rem;font-weight:600;color:#94a3b8;margin-bottom:0.3rem'>TOP 5 ATTRIBUTION DRIVERS</div>", unsafe_allow_html=True)
        top_df = pd.DataFrame(shap_rankings[:5])
        st.dataframe(
            top_df[["rank", "feature", "mean_abs_shap"]].rename(
                columns={"rank": "Rank", "feature": "Feature", "mean_abs_shap": "Mean |SHAP|"}
            ),
            use_container_width=True,
            hide_index=True
        )

st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# MODULES 13 & 14: What-If Simulation Sandbox Panel & Delta Verification
# ═══════════════════════════════════════════════════════════════════════════
sec("🎮 Modules 13 & 14 — What-If Simulation Sandbox & Defensive Verification")

st.markdown(
    "<div style='font-size:0.82rem;color:#94a3b8;margin-bottom:1rem'>"
    "Simulate prospective countermeasure policies by modifying the underlying network state vectors. "
    "The LSTM World Model evaluates the altered state sequence in real-time, forecasting the revised risk trajectory."
    "</div>",
    unsafe_allow_html=True
)

zero_cols = []
applied_labels = []

if sim_host_iso:
    zero_cols.extend(ISOLATION_FEATURES)
    applied_labels.append("Complete Host Isolation")

if sim_smb:
    zero_cols.extend(SMB_FEATURES)
    applied_labels.append("Restrict SMB (Port 445)")

if custom_zero:
    zero_cols.extend(custom_zero)
    applied_labels.append(f"Custom ({len(custom_zero)} features)")

zero_cols = list(set(zero_cols))
is_simulated = len(zero_cols) > 0

if is_simulated:
    with st.spinner("Computing neural re-inference on modified state tensor…"):
        sim_probs = run_intervention(zero_cols)
    
    # Metrics calculation
    orig_mean = float(np.mean(probs))
    sim_mean = float(np.mean(sim_probs))
    orig_peak = float(np.max(probs))
    sim_peak = float(np.max(sim_probs))
    orig_last = float(probs[-1])
    sim_last = float(sim_probs[-1])
    
    reduction_pct = ((orig_last - sim_last) / (orig_last + 1e-6)) * 100
    
    sim_c1, sim_c2, sim_c3, sim_c4 = st.columns(4)
    with sim_c1:
        st.markdown(kpi("Baseline Risk", f"{orig_last:.3f}", "Current Window", color="#ef4444"), unsafe_allow_html=True)
    with sim_c2:
        st.markdown(kpi("Simulated Risk", f"{sim_last:.3f}", "Post-Intervention", color="#34d399"), unsafe_allow_html=True)
    with sim_c3:
        st.markdown(kpi("Peak Risk Reduction", f"{orig_peak - sim_peak:+.3f}", f"Peak: {orig_peak:.2f} → {sim_peak:.2f}", color="#38bdf8"), unsafe_allow_html=True)
    with sim_c4:
        st.markdown(kpi("Risk Reduction Impact", f"{reduction_pct:.1f}%", f"{len(zero_cols)} features suppressed", color="#f0abfc"), unsafe_allow_html=True)
        
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    
    interv_label = " + ".join(applied_labels)
    fig_delta = delta_chart(ts[-100:], probs[-100:], sim_probs[-100:], label=interv_label)
    st.plotly_chart(fig_delta, use_container_width=True)
    
    st.caption(f"⚡ Active Interventions: {interv_label} · Zeroed features: {', '.join(zero_cols[:6])}{' ...' if len(zero_cols) > 6 else ''}")

else:
    st.info("💡 Select one or more simulation presets in the sidebar (or choose custom features) to trigger real-time neural re-inference and compare before vs after trajectories.")
    fig_base = timeline_chart(ts[-100:], probs[-100:], threshold=0.5, title="Baseline Network Risk Trajectory (Recent 100 Windows)")
    st.plotly_chart(fig_base, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# MODULE 15: Automated Defense Strategy Recommendation
# ═══════════════════════════════════════════════════════════════════════════
sec("🛡️ Module 15 — Automated Defense Strategy & Containment Playbook")

strat_col1, strat_col2 = st.columns([1.2, 1.0])

with strat_col1:
    st.markdown(
        f"""
        <div style='background:rgba(30,41,59,0.7); border:1px solid rgba(99,102,241,0.25);
                    border-radius:12px; padding:1.2rem;'>
            <div style='color:#38bdf8; font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.4rem;'>
                Tailored Playbook Recommendation
            </div>
            <div style='font-size:1.05rem; font-weight:700; color:#f8fafc; margin-bottom:0.8rem;'>
                {mitre_current.defense_recommendation}
            </div>
            <div style='font-size:0.82rem; color:#94a3b8; line-height:1.6;'>
                Based on active heuristic classification for <b>{mitre_current.stage}</b> ({mitre_current.technique_name}) 
                and primary neural attribution weights, the system recommends the following immediate SOC actions:
            </div>
            <ul style='font-size:0.82rem; color:#cbd5e1; margin-top:0.6rem; padding-left:1.2rem; line-height:1.7;'>
                <li><b>Boundary Enforcement:</b> Implement micro-segmentation rules restricting ingress/egress flows on ports associated with {top_shap_names[0] if top_shap_names else 'anomalous flags'}.</li>
                <li><b>Endpoint Quarantine:</b> Initiate automated isolation for hosts exhibiting abnormal RST/SYN flag frequencies or high burst byte rates.</li>
                <li><b>Credential Invalidation:</b> Force session termination and credential rotation for administrative accounts traversing anomalous subnet paths.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

with strat_col2:
    st.markdown(
        """
        <div style='background:rgba(15,23,42,0.8); border:1px solid rgba(51,65,85,0.6);
                    border-radius:12px; padding:1.2rem;'>
            <div style='color:#f59e0b; font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.6rem;'>
                SOC Execution Checklist
            </div>
        """,
        unsafe_allow_html=True
    )
    st.checkbox("Issue EDR Host Containment Command", value=is_simulated)
    st.checkbox("Push Ingress Port ACL to Perimeter Firewalls", value=sim_smb)
    st.checkbox("Generate SIEM Correlation Incident Ticket", value=True)
    st.checkbox("Export Window Attribution Report for Forensics", value=False)
    st.markdown("</div>", unsafe_allow_html=True)