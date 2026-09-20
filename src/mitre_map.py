"""
CyberForeSight AI — MITRE ATT&CK Stage Mapping (Phase 4)
Project: SIH26153 · AI-based Network Attack Forecasting from Network Traffic Data (NTRO)

Module: src/mitre_map.py
Description:
    Implements Phase 4 (Steps 30-31) of PHASES.md:
      1. Rule-based heuristic function `map_to_mitre_stage()` mapping network feature
         metrics and SHAP attributions to MITRE ATT&CK enterprise tactics and techniques.
      2. Maps port-scan patterns -> Reconnaissance (TA0043).
      3. Maps SYN flood & reset patterns -> Initial Access / Brute Force (TA0001 / TA0006).
      4. Maps periodic timing & low-variance IAT -> Command & Control (TA0011).
      5. Maps high-volume byte & packet rates -> Exfiltration (TA0010) or DoS (TA0040).
      6. Provides defensive decision support recommendations (defensive simulation only per RULES.md).

Design & Scoping Note (PHASES.md Step 31):
    MITRE ATT&CK mapping is heuristic/rule-based because CIC-IDS2018 contains flow-level
    telemetry without native matrix tactic annotations. This rule-based translation
    provides interpretable, tactical context for SOC analysts without fabricating labels.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mitre_map")


@dataclass
class MitreStageResult:
    """Structured result of MITRE ATT&CK heuristic stage mapping."""

    stage: str
    tactic_id: str
    technique_id: str
    technique_name: str
    confidence: float
    risk_level: str
    matched_rules: List[str]
    contributing_features: Dict[str, float]
    defense_recommendation: str
    scoping_note: str = (
        "Heuristic rule-based mapping per PHASES.md Step 31 (CIC-IDS2018 has no native stage labels)."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary."""
        return asdict(self)


def _get_val(features: Dict[str, float], key: str, default: float = 0.0) -> float:
    """Safely retrieves float metric value from features dictionary."""
    val = features.get(key, default)
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def map_to_mitre_stage(
    features: Union[Dict[str, float], pd.Series, np.ndarray],
    feature_names: Optional[List[str]] = None,
    attack_probability: Optional[float] = None,
    top_shap_features: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> MitreStageResult:
    """
    Maps network state-vector metrics and SHAP feature attributions to MITRE ATT&CK tactics.

    Args:
        features: Feature dictionary, pandas Series, or 1D array of state variables.
        feature_names: Optional feature column names (required if features is np.ndarray).
        attack_probability: Forecasted attack probability from LSTM World Model or baseline.
        top_shap_features: Optional list of top influential feature names from SHAP.
        threshold: Decision threshold for attack classification (default: 0.5).

    Returns:
        MitreStageResult dataclass detailing stage, tactic, technique, and defense guidance.
    """
    # Normalize features input into a clean string->float dictionary
    feat_dict: Dict[str, float] = {}
    if isinstance(features, pd.Series):
        for k, v in features.to_dict().items():
            try:
                feat_dict[str(k)] = float(v)
            except (ValueError, TypeError):
                continue
    elif isinstance(features, dict):
        for k, v in features.items():
            try:
                feat_dict[str(k)] = float(v)
            except (ValueError, TypeError):
                continue
    elif isinstance(features, np.ndarray):
        if feature_names is None:
            raise ValueError("feature_names must be provided when features is a numpy array.")
        for name, val in zip(feature_names, features):
            try:
                feat_dict[str(name)] = float(val)
            except (ValueError, TypeError):
                continue
    else:
        raise TypeError(f"Unsupported features type: {type(features)}")

    top_shap_set = set(top_shap_features or [])
    matched_rules: List[str] = []
    contributing: Dict[str, float] = {}

    # Extract key normalized metrics (Z-scores where > 0.5 indicates above mean)
    dst_ports = _get_val(feat_dict, "dst_port_nunique")
    syn_flags = _get_val(feat_dict, "syn_flag_cnt_sum")
    rst_flags = _get_val(feat_dict, "rst_flag_cnt_sum")
    psh_flags = _get_val(feat_dict, "psh_flag_cnt_sum")
    flow_duration = _get_val(feat_dict, "flow_duration_mean")
    bytes_s = _get_val(feat_dict, "flow_byts_s_mean")
    pkts_s = _get_val(feat_dict, "flow_pkts_s_mean")
    totlen_fwd = _get_val(feat_dict, "totlen_fwd_pkts_sum")
    totlen_bwd = _get_val(feat_dict, "totlen_bwd_pkts_sum")
    iat_mean = _get_val(feat_dict, "flow_iat_mean")
    pkt_len_max = _get_val(feat_dict, "pkt_len_max_max")
    flow_count = _get_val(feat_dict, "flow_count")

    # If attack probability is supplied and below threshold, classify as Benign/Nominal
    if attack_probability is not None and attack_probability < threshold:
        return MitreStageResult(
            stage="Nominal",
            tactic_id="TA0000",
            technique_id="T0000",
            technique_name="Normal Network Operations",
            confidence=float(1.0 - attack_probability),
            risk_level="Low",
            matched_rules=["Forecasted attack probability is below decision threshold."],
            contributing_features={"attack_probability": attack_probability},
            defense_recommendation="Maintain continuous baseline monitoring; no defensive intervention required.",
        )

    # -----------------------------------------------------------------------
    # Heuristic Rule Scoring Across Candidate Tactics
    # -----------------------------------------------------------------------
    scores: Dict[str, float] = {
        "Reconnaissance": 0.0,
        "Initial Access": 0.0,
        "Command & Control": 0.0,
        "Exfiltration": 0.0,
        "Impact": 0.0,
    }

    # Rule 1: Reconnaissance (Active Scanning / Port Sweep)
    if dst_ports > 0.5 or "dst_port_nunique" in top_shap_set:
        scores["Reconnaissance"] += 3.0
        matched_rules.append(f"Elevated unique destination ports ({dst_ports:+.2f} std dev).")
        contributing["dst_port_nunique"] = dst_ports
    if flow_duration < 0.0 and flow_count > 0.5:
        scores["Reconnaissance"] += 1.5
        matched_rules.append("High volume of rapid, brief flow connections (probing signature).")
        contributing["flow_count"] = flow_count

    # Rule 2: Initial Access & Credential Access (SYN Flood / Brute Force)
    if syn_flags > 0.5 or "syn_flag_cnt_sum" in top_shap_set:
        scores["Initial Access"] += 2.5
        matched_rules.append(f"Elevated SYN handshake flags ({syn_flags:+.2f} std dev).")
        contributing["syn_flag_cnt_sum"] = syn_flags
    if rst_flags > 0.5 or "rst_flag_cnt_sum" in top_shap_set:
        scores["Initial Access"] += 3.0
        matched_rules.append(f"Frequent TCP reset flags ({rst_flags:+.2f} std dev) indicating failed logins.")
        contributing["rst_flag_cnt_sum"] = rst_flags

    # Rule 3: Exfiltration (Heavy outbound transfer)
    if bytes_s > 1.0 or totlen_fwd > 1.0 or totlen_bwd > 1.0 or "flow_byts_s_mean" in top_shap_set:
        scores["Exfiltration"] += 3.5
        matched_rules.append(f"High byte transfer throughput ({bytes_s:+.2f} std dev).")
        contributing["flow_byts_s_mean"] = bytes_s
    if pkt_len_max > 1.0:
        scores["Exfiltration"] += 1.5
        matched_rules.append("Large max packet length distribution.")
        contributing["pkt_len_max_max"] = pkt_len_max

    # Rule 4: Impact (Denial of Service / Flooding)
    if pkts_s > 1.0 or "flow_pkts_s_mean" in top_shap_set:
        scores["Impact"] += 3.0
        matched_rules.append(f"Abnormal packet transmission rate ({pkts_s:+.2f} std dev).")
        contributing["flow_pkts_s_mean"] = pkts_s
    if flow_duration > 1.5:
        scores["Impact"] += 2.0
        matched_rules.append("Prolonged hanging flow duration (Slowloris/resource exhaustion pattern).")
        contributing["flow_duration_mean"] = flow_duration

    # Rule 5: Command & Control (Beaconing / Low Variance IAT)
    if psh_flags > 0.5 and iat_mean < 0.0:
        scores["Command & Control"] += 2.0
        matched_rules.append("Frequent push flags with rapid, steady inter-arrival timing.")
        contributing["psh_flag_cnt_sum"] = psh_flags

    # Determine highest scoring stage
    best_stage = max(scores, key=scores.get)
    max_score = scores[best_stage]

    # Calculate confidence based on score magnitude and attack probability
    base_conf = min(0.95, 0.50 + (max_score / 10.0))
    if attack_probability is not None:
        confidence = float(np.clip((base_conf + attack_probability) / 2.0, 0.50, 0.98))
    else:
        confidence = float(np.clip(base_conf, 0.50, 0.95))

    # Determine Risk Level
    if confidence >= 0.80 or (attack_probability and attack_probability >= 0.80):
        risk_level = "High" if confidence < 0.90 else "Critical"
    else:
        risk_level = "Medium"

    # Stage-specific mapping details
    if best_stage == "Reconnaissance":
        tactic_id = "TA0043"
        technique_id = "T1595"
        technique_name = "Active Scanning"
        recommendation = (
            "Isolate probing host, implement dynamic rate limiting on inbound SYN handshakes, "
            "and review perimeter firewall access control lists (ACLs)."
        )
    elif best_stage == "Initial Access":
        tactic_id = "TA0001"
        technique_id = "T1110"
        technique_name = "Brute Force / Credential Access"
        recommendation = (
            "Enforce account lockout thresholds on authentication services (SSH/FTP), "
            "restrict administrative access to dedicated management subnets, and enable SYN cookies."
        )
    elif best_stage == "Exfiltration":
        tactic_id = "TA0010"
        technique_id = "T1041"
        technique_name = "Exfiltration Over C2 Channel"
        recommendation = (
            "Throttle suspicious outbound data transfers, inspect egress payloads via DLP, "
            "and isolate source endpoints exhibiting abnormal outbound byte volumes."
        )
    elif best_stage == "Impact":
        tactic_id = "TA0040"
        technique_id = "T1498"
        technique_name = "Network Denial of Service"
        recommendation = (
            "Activate reverse proxy connection limits, redirect traffic through DDoS scrubbing "
            "infrastructure, and reduce server keep-alive timeouts."
        )
    elif best_stage == "Command & Control":
        tactic_id = "TA0011"
        technique_id = "T1071"
        technique_name = "Application Layer Protocol"
        recommendation = (
            "Sinkhole destination IP/domain lookups, terminate persistent beaconing sessions, "
            "and inspect encrypted traffic for repetitive packet interval signatures."
        )
    else:
        best_stage = "Reconnaissance"
        tactic_id = "TA0043"
        technique_id = "T1595"
        technique_name = "Active Scanning"
        recommendation = "Investigate source IP and monitor consecutive window progression."

    if not matched_rules:
        matched_rules.append("Elevated general attack probability forecast from temporal World Model.")

    return MitreStageResult(
        stage=best_stage,
        tactic_id=tactic_id,
        technique_id=technique_id,
        technique_name=technique_name,
        confidence=round(confidence, 4),
        risk_level=risk_level,
        matched_rules=matched_rules,
        contributing_features=contributing,
        defense_recommendation=recommendation,
    )


# ---------------------------------------------------------------------------
# CLI & Batch Testing Utility
# ---------------------------------------------------------------------------


def batch_map_dataset(
    data_path: Union[str, Path] = "data/windows/window_features_normalized.csv",
    num_samples: int = 10,
) -> pd.DataFrame:
    """
    Demonstrates heuristic MITRE ATT&CK stage mapping across sample attack windows.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Normalized dataset not found: {path}")

    df = pd.read_csv(path)
    attack_windows = df[df["is_attack_window"] == 1].head(num_samples)

    records: List[Dict[str, Any]] = []
    for idx, row in attack_windows.iterrows():
        res = map_to_mitre_stage(row, attack_probability=0.85)
        records.append(
            {
                "window_timestamp": row.get("window_timestamp", idx),
                "predicted_stage": res.stage,
                "tactic_id": res.tactic_id,
                "technique": f"{res.technique_id}: {res.technique_name}",
                "confidence": res.confidence,
                "risk_level": res.risk_level,
                "top_rule": res.matched_rules[0] if res.matched_rules else "",
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    """CLI execution entrypoint."""
    parser = argparse.ArgumentParser(
        description="CyberForeSight AI — MITRE ATT&CK Stage Mapping"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/windows/window_features_normalized.csv",
        help="Path to normalized window CSV",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of attack windows to map for demonstration",
    )
    args = parser.parse_args()

    results_df = batch_map_dataset(args.data_path, num_samples=args.num_samples)

    print(f"\n{'=' * 75}")
    print("PHASE 4 -- MITRE ATT&CK HEURISTIC STAGE MAPPING DEMONSTRATION")
    print(f"{'=' * 75}")
    for _, r in results_df.iterrows():
        print(f"Window:      {r['window_timestamp']}")
        print(f"MITRE Stage: {r['predicted_stage']} ({r['tactic_id']})")
        print(f"Technique:   {r['technique']}")
        print(f"Confidence:  {r['confidence'] * 100:.1f}% | Risk: {r['risk_level']}")
        print(f"Evidence:    {r['top_rule']}")
        print("-" * 75)
    print("Scoping Note: Rule-based mapping per PHASES.md Step 31.\n")


if __name__ == "__main__":
    main()
