# ARCHITECTURE.md — CyberForeSight AI

Project: CyberForeSight AI · SIH 2026 PS 26153 · NTRO · Software · Blockchain & Cybersecurity

## 1. Architecture Overview

CyberForeSight AI is a modular, offline, Python-based pipeline that turns network traffic into a temporal World Model forecast of attack progression, explains that forecast, lets a defender simulate interventions, and presents everything through a Streamlit dashboard, benchmarked against a non-temporal baseline.

## 2. Architectural Principles

Modularity over monolith · offline-first · explainability by construction · simulation strictly isolated from real control paths · minimal dependencies · testable boundaries between layers · no fabricated data.

## 3. System Context

Inputs: CIC-IDS2018 CSV (primary), optional PCAP files, optional asset/topology configuration. Output consumer: a

human defender/evaluator via the Streamlit dashboard. No external system integration is required for core operation.

# 4. Input Layer

Accepts CSV flow records and, optionally, PCAP files. Normalizes both into a common internal "raw record" representation tagged with source type.

# 5. Data Ingestion Layer

Reads files from local disk, streams/batches large CSVs, and hands raw records to the Validation Layer. No network fetch is part of ingestion.

# 6. Validation Layer

Schema, type, timestamp, NaN/Inf, and duplicate checks. Produces a validation report object consumed by both the pipeline (to decide whether to proceed) and the dashboard (to display data-quality status).

# 7. Feature Extraction Layer

Two sub-components: FlowFeatureExtractor (CSV/NetFlow input) and PacketFeatureExtractor (PCAP input, via Scapy/PyShark). Both emit features plus availability metadata into a shared feature-matrix format.

# 8. Time-Window Layer

Buckets raw/feature records into configurable, non- overlapping time windows and aggregates per-window statistics.

# 9. Network State Layer

Converts each window's aggregated features into a fixed- schema state vector, S_t, per the configuration-driven feature list.

# 10. Temporal Sequence Layer

Builds (sequence_length, feature_dim) training/inference examples from consecutive state vectors, preserving chronological order.

# 11. World Model Layer

Hosts the LSTM model definition, training loop, and inference interface. Learns P(S_t+1 | S_t) via supervised dynamics learning over sequences.

# 12. Forecasting Engine

Performs K-step autoregressive rollout using the World Model, producing a per-step probability/state forecast.

# 13. Attack Progression Engine

Aggregates the K-step forecast into a progression trajectory (escalating/plateauing/receding) and a per-step infiltration

probability.

# 14. Attack Path Engine

Combines progression trajectories with configured topology/asset relationships to generate candidate attack paths.

# 15. MITRE Mapping Layer

Independent rule/heuristic layer that interprets predicted state/feature evidence into one of the five defined MITRE ATT&CK-aligned stages, with the evidence retained alongside the label.

# 16. Explainability Layer

Computes SHAP (or equivalent) attribution against the exact feature vector used for the corresponding prediction; ranks and exposes top contributing features.

# 17. Optional LLM Explanation Layer

Consumes only the structured SHAP evidence object and produces prose. Stateless with respect to the prediction pipeline — cannot write back into probability/stage/path/risk fields. Feature-flagged; disabled by default in offline mode.

# 18. Asset Risk Layer

Combines predicted path/target information with configured asset criticality data, if supplied; otherwise reports risk as not evaluated.

# 19. Timeline and Attack Graph Layer

Builds the observed/predicted/simulated timeline and the node/edge attack graph structure consumed by the dashboard.

# 20. What-If Simulation Layer

Applies a documented transformation to a copy of the current state per selected intervention (Restrict SMB, Isolate Host). Has no dependency on any real network control API — this is a hard architectural boundary, not just a policy.

# 21. Re-Forecasting Layer

Re-runs the Forecasting Engine on the simulated state using the same horizon and model, producing a comparable output object.

# 22. Recommendation Layer

Ranks modeled interventions by simulated risk reduction and surfaces the top recommendation(s), labelled as simulated estimates.

# 23. Dashboard Layer

Streamlit application composed of independent page modules, each calling into a thin service layer ( services/* ) that wraps the pipeline. No model/feature logic lives in page code.

# 24. Evaluation Layer

Runs both the LSTM World Model and the Logistic Regression baseline over the same held-out, time- respecting split, computing Precision/Recall/F1/FPR and forecasting-specific metrics (lead time, future-state quality, stage-prediction quality).

# 25. Storage / Artifacts

Local filesystem: raw/processed data cache, trained model weights + config + scaler, evaluation reports, logs. No database server required for the MVP.

# 26. Configuration

A single config file (YAML/JSON) drives window size, sequence length, feature list, horizon K, hyperparameters, dataset paths, and optional-component flags (LLM, packet extraction).

# 27. Offline Architecture

Every layer except the optional LLM layer uses only locally installed libraries and locally stored artifacts. The system is validated with network access disabled (see TESTING.md).

# 28. Security Boundaries

A hard boundary exists around the What-If Simulation Layer: it can read state and produce a simulated copy, but has zero code path to any real firewall/host-control interface. This boundary is enforced architecturally (no such interface exists in the codebase), not just documented as a rule.

# 29. Failure Handling

Each layer returns a typed result or structured error. Optional layers (SHAP, LLM, packet extraction) degrade gracefully without breaking the core forecast → dashboard path.

# 30. Scalability / Future Extensions

The modular boundaries (extraction / windowing / state / sequence / model / forecast / explain / simulate / dashboard) allow swapping the LSTM for a Transformer or GNN, or adding datasets, without redesigning the rest of the pipeline.

# World Model Detail

S_t = current network state (time-windowed feature vector). Transition: |

||S_t → S_t+1, conceptually learning||P(S_t+1|
|---|---|---|---|
|S_t).|S_t|→ predicted|→|

## Forward simulation: S_t+1

predicted S_t+2 →... → K-step forecast. Attack probability at each future step is derived from the model's progression/probability head applied to each predicted state; this is not a simple binary classifier — it is a sequential, autoregressive forecast over a learned transition model.

# Feature Groups

Flow (always available from CSV/NetFlow): IPs, ports, protocol, TCP flags, bytes, packets, duration, IAT statistics, bidirectional ratios. Packet-derived (only when PCAP is processed): TTL & variance, TCP window, fragmentation, payload-size statistics, port-scan signatures, retransmissions. Unavailable packet features are never claimed as present.

# Technology

Core: Python, Pandas/NumPy, Scikit-learn (baseline + metrics), PyTorch or TensorFlow (LSTM), SHAP, Streamlit. Optional: Scapy/PyShark (PCAP extraction), Matplotlib/Plotly (visualization), a local/offline LLM (explanation prose). No Kubernetes, no microservices, no blockchain, no mandatory cloud dependency, no message queues — none are required by this architecture.

# Dashboard–Pipeline Interaction

function →

|Streamlit|page|→ service|layer|
|---|---|---|---|
|pipeline|(feature/model/forecast/explain/simulate)|||
|→ typed|result|→ page|rendering. Model logic never|

lives inside a Streamlit page module.

# LLM Boundary

object

|SHAP evidence|→ structured|explanation|
|---|---|---|
|→ optional text|local/offline|LLM → explanation|

The LLM never decides attack probability, attack stage, path ranking, asset risk, or which intervention to recommend. The core system functions fully without it.

# Evaluation Architecture

|flowchart|LR||||
|---|---|---|---|---|
|subgraph|Baseline||||
|A1[Same Regression] end|Features]|--> A2[Logistic|||
|subgraph|WorldModel||||
|B1[Temporal Model] end|Sequences]|-->|B2[LSTM|World|
|A2--> Precision/Recall/F1/FPR]|C[Common|Evaluation:|||
|B2-->|C||||
|B2-->|D[Forecasting|Metrics:|Lead|Time,|
|Future-State|Quality,|Stage Quality]|||

# Diagrams

1. System Architecture flowchart

||TD||
|---|---|---|
|Input[CSV/PCAP||--> Ingest[Ingestion]|
|Ingest|--> Valid[Validation]||
|Valid|--> Feat[Feature|Extraction]|
|Feat-->|Win[Time-Window|Engine]|
|Win-->|State[Network|State]|
|State|--> Seq[Temporal|Sequence|
|Seq-->|Model[LSTM|World Model]|
|Model|--> Forecast[K-Step|Forecasting|
|Forecast|--> Prog[Attack|Progression|
|Prog-->|Path[Attack|Path Engine]|
|Prog-->|Mitre[MITRE|Mapping]|
|Prog-->|Shap[SHAP|Explainability]|
|Shap-->|LLM[Optional|LLM Explanation]|
|Path-->|Asset[Asset|Risk]|
|Path-->|Sim[What-If|Simulator]|
|Sim-->|Reforecast[Re-Forecasting]||
|Reforecast|-->||
|Compare|--> Rec[Recommendation]||
|Mitre|--> Dash[Streamlit|Dashboard]|
|Asset|--> Dash||
|Rec-->|Dash||
|Dash-->|Eval[Evaluation/Benchmarking]||

Input]

Builder]

Engine] Engine]

Compare[Before/After Comparison]

2. Data Flow — see High-Level Data Flow diagram in DESIGN.md §3 (identical flow).
3. World Model Forecasting Loop

|flowchart|LR||
|---|---|---|
|St[S_t|observed]|--> LSTM1[LSTM|
|LSTM1|--> Sp1[predicted|S_t+1]|
|Sp1-->|LSTM2[LSTM|step]|
|LSTM2|--> Sp2[predicted|S_t+2]|
|Sp2-->|LSTMk[...|K steps]|
|LSTMk|--> SpK[predicted|S_t+K]|

step]

4. What-If Simulation

|flowchart|TD||||
|---|---|---|---|---|
|Obs[Observed|State]|-->||State]|
|Copy|--> Apply[Apply|Simulated|Intervention]||
|Apply|--> SimState[Simulated||State]||
|SimState|--> Reforecast[Re-run||World|Model]|
|Obs-->|ForecastOrig[Original||Forecast]||
|Reforecast Forecast]|-->|ForecastSim[Simulated|||
|ForecastOrig Comparison]|-->|Diff[Before/After|||
|ForecastSim|-->|Diff|||

Copy[Copy

5. Evaluation Architecture — see Evaluation Architecture diagram above.
# 2-Page SIH Architecture Summary (condensed)

Core pipeline: Traffic (CSV/PCAP) → Validation → Flow + Packet-Derived Feature Extraction → Time-Window Engine → Network State Vector → Temporal Sequence Builder → LSTM World Model ( P(S_t+1|S_t) ) → K-Step Forecast →

Attack Progression Probability + MITRE Stage Mapping → SHAP Explainability (+ optional grounded LLM prose) → Attack Path & Asset Risk → What-If Simulator (simulation- only) → Re-Forecast → Before/After Comparison → Defence Recommendation → Streamlit SOC Dashboard → Evaluation vs Logistic Regression baseline (Precision/Recall/F1/FPR + lead time).

Key architectural guarantees: offline-core, no autonomous real-world action, explicit observed/predicted/simulated distinction, explainability is evidence-grounded (SHAP first, LLM only verbalizes), baseline comparison is mandatory, no fabricated packet-level data.