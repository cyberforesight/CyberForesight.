## PRD.md — CyberForeSight AI

Problem Statement ID: IH26153 / 26153

Title: AI based Network Attack Forecasting from Network

Traffic Data

Organization: National Technical Research Organisation

(NTRO)

Category: Software

Theme: Blockchain & Cybersecurity

Event: Smart India Hackathon 2026

This is a defensive cybersecurity decision-support system. It MUST NOT automatically modify real firewalls, isolate real hosts, execute commands on real infrastructure, or perform offensive actions of any kind.

## 1. Executive Summary

CyberForeSight AI is an explainable, offline-capable system that learns the temporal evolution of network behaviour and forecasts how an in-progress cyberattack is likely to progress, before it completes. Instead of classifying individual flows as benign/malicious (traditional IDS), the system learns a World Model of network-state transition

dynamics — P(S_t+1 | S_t) — using an LSTM, and

performs K-step forward simulation to estimate infiltration probability, likely attack stage (mapped to MITRE ATT&CK), and probable attack paths. Predictions are explained via


SHAP feature attribution, optionally verbalized by an LLM. A What-If simulator lets a defender test hypothetical interventions (e.g., "Restrict SMB", "Isolate Host") entirely in simulation, comparing forecasts before/after. Results are presented in an offline Streamlit SOC dashboard and benchmarked against a Logistic Regression baseline.

## 2. Problem Statement

Traditional network intrusion detection treats each flow independently and outputs a static benign/malicious label. This discards the temporal and causal structure of an infiltration — the order in which ports are probed, the timing between reconnaissance and lateral movement, etc. NTRO's problem statement (26153) asks for an AI system that learns network behaviour as a process unfolding over time, forecasts attacker progression before compromise completes, and gives defenders interpretable, forward- looking decision support — applicable to enterprise and Critical Information Infrastructure environments.

## 3. Goals

- G1: Learn temporal network-state transition dynamics from flow-level (and optionally packet-derived) traffic features.

- G2: Forecast K steps into the future and estimate infiltration/attack-progression probability per future window.

- G3: Map predicted behaviour to MITRE ATT&CK stages (Reconnaissance, Initial Access, Lateral Movement, C2,


## Exfiltration).

- G4: Generate multiple ranked probable future attack paths.

- G5: Explain every prediction with SHAP (or equivalent) feature attribution; optionally verbalize with a grounded LLM.

- G6: Provide a What-If simulator for hypothetical defensive interventions, purely in simulation.

- G7: Benchmark the World Model against a Logistic Regression baseline (Precision/Recall/F1/FPR + forecasting-specific metrics).

- G8: Run fully offline with no mandatory cloud dependency.

- G9: Present all of the above through a Streamlit SOC dashboard.

- G10: Be reproducible: versioned data, config, model artifacts, and training/inference pipelines.

## 4. Non-Goals

- NG1: No blockchain component (theme mention does not imply a requirement).

- NG2: No autonomous response — no real firewall/host modification.

- NG3: No offensive capability — no scanning, exploitation, or credential theft.

- NG4: No malware development.

- NG5: Not a SIEM replacement.


- NG6: No authentication/login system unless technically necessary for the demo.

- NG7: No unnecessary microservices or cloud infrastructure.

- NG8: No claim that packet-level features exist when only flow-level CSV data is available.

- NG9: No claim of guaranteed/certain outcomes — outputs are probabilistic estimates.

## 5. Target Users

- SOC analysts / defenders who need early warning of attack progression.

- NTRO / CII evaluators assessing forecasting and explainability capability.

- SIH judges evaluating technical depth, correctness, and safety posture.

## 6. User Stories

- As a SOC analyst, I want to see the probability that current traffic will escalate to lateral movement in the next N minutes, so I can act before compromise.

- As a defender, I want to know why the model believes an attack is progressing (which flags/ports/features), so I can validate the alert.

- As a defender, I want to simulate "Restrict SMB" and see how the forecast changes, so I can decide whether the intervention is worth applying for real.


- As an evaluator, I want to compare the LSTM World Model against a simple baseline, so I can judge whether temporal modelling adds real value.

- As an operator, I want the system to run without internet access, so it is deployable in an air-gapped CII environment.

## 7. Functional Requirements

| ID Requirement |   | Testable Criterion |
| --- | --- | --- |
|   |   | System parses the |
| FR- | Ingest CIC-IDS2018 | dataset without error |
| 1 | CSV flow records | and reports |
|   |   | row/column counts |
|   |   | Missing/invalid |
| FR- |   | columns produce a |
|   | Validate input schema |   |
| 2 |   | structured validation |
|   |   | report, not a crash |
|   |   | Output feature matrix |
| FR- | Extract flow-level | contains all listed flow |
| 3 | features | features, correctly |
|   |   | typed |
|   |   | Feature matrix marks |
|   | Extract packet-derived |   |
| FR- |   | packet features as |
|   | features when input | unavailable when |
| 4 |   |   |
|   | supports it (PCAP) |   |
|   |   | input is CSV-only |
| FR- | Aggregate features | Given known |
| 5 | into time windows | timestamps, records |


| ID Requirement | Testable Criterion |
| --- | --- |
|   | map to the correct |
|   | window |
|   | deterministically |
|   | State vector shape |
| FR- Build network state |   |
|   | matches configured |
| 6 vectors per window |   |
|   | feature list |
|   | Sequences and targets |
| FR- Build temporal |   |
|   | align correctly (see |
| 7 sequences of length L |   |
|   | TESTING.md) |
|   | Model trains and |
| FR- Train LSTM World | serializes; loss |
| 8 Model on sequences | decreases over epochs |
|   | on sample data |
| Train Logistic | Baseline trains and |
| FR- |   |
| Regression baseline | serializes |
| 9 |   |
| on equivalent features | independently of LSTM |
|   | Forecast output |
| FR- Perform K-step |   |
|   | contains exactly K |
| 10 forward forecasting |   |
|   | future steps |
| Estimate infiltration |   |
| FR- | Output probabilities are |
| probability per future |   |
| 11 | finite and within [0,1] |
| window |   |
|   | Every stage output |
| FR- Map predicted state to | cites the |
| 12 MITRE ATT&CK stage | evidence/features it is |
|   | based on |


| ID Requirement | Testable Criterion |
| --- | --- |
|   | Paths are deduplicated |
| FR- Generate ranked |   |
|   | and ordered by |
| 13 candidate attack paths |   |
|   | probability/risk |
| Produce SHAP | Attribution references |
| FR- |   |
| explanations for each | only real input features, |
| 14 |   |
| prediction | ranked correctly |
|   | System functions |
| FR- Optional LLM | identically (minus |
| 15 explanation layer | prose explanation) with |
|   | LLM disabled |
| Identify potentially |   |
|   | No asset criticality is |
| FR- affected assets from |   |
|   | invented beyond |
| 16 configured |   |
|   | configured data |
| topology/evidence |   |
|   | Simulated state differs |
| What-If simulation: |   |
| FR- | from real observed |
| Restrict SMB / Isolate |   |
| 17 | state; original state is |
| Host |   |
|   | preserved |
|   | Before/after |
| FR- Re-forecast after | comparison is |
| 18 simulated intervention | produced from the |
|   | same starting state |
|   | Every dashboard panel |
| FR- Present all outputs in a | renders from pipeline |
| 19 Streamlit dashboard | outputs, not from UI- |
|   | embedded logic |


| ID Requirement | Testable Criterion |
| --- | --- |
|   | Report includes |
| FR- Evaluate LSTM vs |   |
|   | Precision, Recall, F1, |
| 20 baseline |   |
|   | FPR for both models |
|   | Core pipeline succeeds |
| FR- |   |
| Run fully offline | with network access |
| 21 |   |
|   | disabled |

## 8. Data Requirements

Primary dataset: CIC-IDS2018 (CSV flow records).

## Flow-level features (must be supported):

source/destination IP, source/destination port, protocol, TCP flag bitmask (SYN/ACK/FIN/RST/PSH/URG), bytes transferred, packets per flow, flow duration, inter-arrival time statistics (mean/variance/max), bidirectional flow ratios.

Packet-derived features (supported via an extensible extraction layer, only when PCAP input is available): TTL and TTL variance, TCP window size, IP fragmentation indicators, payload-size distribution, port-scan signatures, retransmission counts.

Feature availability contract: every feature must be tagged

as one of directly_available_flow_feature , packet_derived_feature , unavailable_for_current_input , or optional_future_extension . The system must never

fabricate a value for an unavailable feature.


## 9. ML / World Model Requirements

- The primary World Model is an LSTM that learns P(S_t+1 | S_t) over time-windowed state sequences — not a static per-flow classifier.

- Input: sequences of length L of state vectors [S_t- L+1, ..., S_t] . Output: predicted next state(s) and/or progression probability.

- Must generalize beyond memorized training signatures (validated via held-out temporal split).

- Preprocessing/scaling configuration must be saved alongside the model and reused identically at inference.

- Logistic Regression baseline trained on the same feature representation (flattened/aggregated as appropriate) for fair comparison.

## 10. Forecasting Requirements

- Support configurable K-step forward rollout (e.g., K=1,3,5,10).

- Output per future step: infiltration/progression probability, predicted attack stage, contributing features.

- Report lead time: how many windows before an observed/labelled event the model raises a comparable-confidence prediction.

- Rollout must never consume ground-truth future information (no leakage — see Non-Goals/Rules).

## 11. Attack Stage / MITRE Requirements


- Map predicted states to: Reconnaissance, Initial Access, Lateral Movement, Command & Control, Exfiltration.

- Mapping is a separate, documented layer on top of model output — not claimed to be directly predicted by the LSTM.

- Low-evidence or conflicting signals must not produce a falsely confident stage label.

## 12. Explainability Requirements

- Every prediction must carry a SHAP (or equivalent) feature attribution identifying top contributing features.

- Optional LLM explanation layer only verbalizes structured, grounded SHAP evidence — it never determines probability, stage, path ranking, or asset risk.

- If SHAP/LLM is unavailable, core inference still produces predictions (without prose explanation).

## 13. What-If Simulation Requirements

- Supported simulated interventions (initial set): Restrict SMB, Isolate Host.

- Simulation operates on a copy of the current state; the real observed state is never mutated.

- Each intervention's effect on state/features must be explicitly documented (assumptions), not implicit.

- Re-forecast after intervention must be comparable (same model, same horizon) to produce a valid


before/after delta.

- All simulation outputs are visibly labelled "Simulated" in the UI and logs.

## 14. Dashboard Requirements

Streamlit dashboard must expose: current network risk, forecast probability timeline, predicted attack stage, probable attack paths with probabilities, contributing features / SHAP explanation, asset risk, observed timeline, predicted timeline, attack graph, What-If simulator, before/after comparison, defence recommendation, and model/baseline evaluation metrics. UI must call into a service/application layer — no model logic embedded in UI code.

## 15. Evaluation Requirements

- Standard classification metrics (Precision, Recall, F1, False Positive Rate) for both LSTM World Model and Logistic Regression baseline, on the same evaluation protocol.

- Forecasting-specific metrics where feasible: prediction lead time, future-state prediction quality, stage- prediction quality, calibration/uncertainty (if implemented).

- Evaluation must respect temporal ordering (no shuffled train/test split for time-series evaluation).

## 16. Security Requirements


- No offensive capability of any kind (scanning, exploitation, credential theft, malware).

- What-If simulator is sandboxed from any real network control plane — no code path exists that could issue a real firewall/host command.

- All simulated actions are clearly and persistently labelled as simulated in data model, logs, and UI.

## 17. Offline Requirements

- Core pipeline (ingestion features forecasting SHAP dashboard) must run with no internet access.

- Any LLM explanation layer is optional; its absence must not break or degrade core functionality beyond removing prose explanations.

## 18. Performance Requirements

- Feature extraction and inference must complete within demo-acceptable latency on a laptop-class CPU for representative sample sizes (exact thresholds to be measured and recorded during implementation, see TESTING.md).

- Dashboard interactions should respond without a full pipeline re-run unless input data changes.

## 19. Reproducibility Requirements

- Deterministic preprocessing given fixed config and seed.


- Model artifacts, scaler/config, and evaluation reports are versioned together.

- README with exact setup and run instructions.

- Training, inference, and evaluation are separate, independently runnable scripts.

## 20. Constraints

- No blockchain component unless a concrete, justified requirement emerges later (ADR required).

- No GNN/Transformer as the primary MVP model (LSTM is primary; others are future extensions).

- No autonomous/real-world response capability.

- No unnecessary microservices, message queues, or cloud infrastructure.

- No login/auth system unless technically required for the demo.

## 21. Acceptance Criteria

- A1: Given a CIC-IDS2018 sample, the pipeline produces a valid feature matrix with correct availability tags.

- A2: The LSTM model trains, saves, and reloads deterministically; sequence shapes are correct.

- A3: K-step forecast returns exactly K steps with probabilities in [0,1].

- A4: Every forecast includes a MITRE-mapped stage with supporting evidence.

- A5: Every prediction includes a SHAP-based explanation referencing real features.


- A6: What-If simulation produces a distinct, correctly- labelled before/after comparison without mutating the original state.

- A7: Evaluation report shows LSTM vs Logistic Regression on Precision/Recall/F1/FPR.

- A8: The full pipeline (excluding optional LLM) runs successfully with network access disabled.

- A9: The dashboard renders all required panels from pipeline outputs.

## 22. Future Enhancements

- Transformer- or GNN-based World Model variants.

- Additional datasets (CIC-IDS2017, UNSW-NB15, CTU-13, CICIoT2023).

- Calibrated uncertainty estimation.

- Richer asset-risk modelling with real topology integration.

- Additional What-If intervention types.

- Justified blockchain use-case (e.g., tamper-evident audit log) if a concrete need is identified — subject to a new ADR.
