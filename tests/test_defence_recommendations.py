"""
CyberForeSight AI — Defence Recommendations Unit Tests
Validates the prioritized, tiered defence recommendation builder, dynamic data binding,
and card metadata schemas.
"""
from src._dashboard_core import _build_defence_recommendations
from src.mitre_map import MitreStageResult


def test_build_defence_recommendations_tiers_and_keys():
    dummy_mitre = MitreStageResult(
        stage="Exfiltration",
        tactic_id="TA0010",
        technique_id="T1048",
        technique_name="Exfiltration Over Alternative Protocol",
        confidence=0.88,
        risk_level="Critical",
        matched_rules=["High outbound traffic volume"],
        contributing_features={"flow_byts_s_max": 1200000.0},
        defense_recommendation="Enforce egress DLP filtering and sever anomalous connections.",
    )

    path_data = {
        "id": "PATH-01",
        "name": "Lateral SMB Propagation & Exfiltration",
        "target": "Enterprise SQL DB (192.168.1.10)",
        "stages": ["Reconnaissance", "Initial Access", "Command & Control", "Exfiltration"],
        "prob": 0.742,
        "horizon_step": 1,
        "lead_time": "3.5 min",
        "driver": "dst_port_nunique",
    }

    recs = _build_defence_recommendations(
        path_id="PATH-01",
        path=path_data,
        kstep_rollout=[0.742, 0.65, 0.40],
        mitre=dummy_mitre,
        top_shap=["dst_port_nunique"],
        decision_threshold=0.50,
        insufficient_data=False,
    )

    # 1. Verify all 3 tiers exist
    assert "IMMEDIATE" in recs
    assert "HIGH" in recs
    assert "PLANNED" in recs

    # 2. Verify all tiers have actions
    assert len(recs["IMMEDIATE"]) >= 1
    assert len(recs["HIGH"]) >= 1
    assert len(recs["PLANNED"]) >= 1

    # 3. Verify card metadata and footers schema
    for tier_name, actions in recs.items():
        for action in actions:
            assert "title" in action and len(action["title"]) > 0
            assert "desc" in action and len(action["desc"]) > 0
            assert action["related"] == "PATH-01"
            assert action["effort"] in ["Low", "Medium", "High"]
            assert "eta" in action and len(action["eta"]) > 0
            # Data binding
            assert action["target"] == "Enterprise SQL DB (192.168.1.10)"
            assert action["driver"] == "dst_port_nunique"
            assert action["prob"] == "74.2%"


def test_build_defence_recommendations_path_02_and_03():
    dummy_mitre = MitreStageResult(
        stage="Privilege Escalation",
        tactic_id="TA0004",
        technique_id="T1558",
        technique_name="Steal or Forge Kerberos Tickets",
        confidence=0.82,
        risk_level="High",
        matched_rules=["Kerberoasting request spike"],
        contributing_features={},
        defense_recommendation="Audit Active Directory Kerberos requests.",
    )

    for pid in ["PATH-02", "PATH-03"]:
        path_data = {
            "id": pid,
            "name": f"Scenario {pid}",
            "target": "Domain Controller (192.168.1.1)" if pid == "PATH-02" else "Ingress Router (10.0.0.1)",
            "stages": ["Reconnaissance", "Initial Access"],
            "prob": 0.615,
            "horizon_step": 2,
            "lead_time": "4.8 min",
            "driver": "syn_flag_cnt_sum",
        }
        recs = _build_defence_recommendations(
            path_id=pid,
            path=path_data,
            kstep_rollout=[0.5, 0.615],
            mitre=dummy_mitre,
            top_shap=["syn_flag_cnt_sum"],
            decision_threshold=0.50,
            insufficient_data=False,
        )
        assert len(recs["IMMEDIATE"]) >= 1
        assert recs["IMMEDIATE"][0]["related"] == pid
