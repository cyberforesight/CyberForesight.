DESIGN.md — CyberForeSight AI

Project: CyberForeSight AI · SIH 2026 PS 26153 · NTRO
Primary dataset: CIC-IDS2018 · World Model: LSTM ·
Baseline: Logistic Regression
Dashboard: Streamlit · Explainability: SHAP · Optional:
grounded LLM · Execution: offline

This document describes how the system behaves
internally. It does not repeat the PRD's requirements list and
does not contain source code.



1. Design Goals

Learn realistic temporal state-transition dynamics; forecast
attack progression before compromise; explain every
prediction; support safe hypothetical intervention testing;
remain fully offline and reproducible.



2. Design Principles

Offline-first · modular · reproducible · explainable · temporal ·
defensive · testable · data-driven · no fabricated telemetry ·
no autonomous real-world response.



3. High-Level Data Flow


  flowchart TD
       A[Traffic Input: CSV/PCAP] --> B[Validation]
       B --> C[Feature Extraction]
       C --> D[Time-Window Engine]
       D --> E[Network State Vector]
       E --> F[Temporal Sequence Builder]
       F --> G[LSTM World Model]
       G --> H[K-Step Forecast]
       H --> I[Attack Progression + Stage Mapping]
       I --> J[SHAP Explainability]
       J --> K[Optional LLM Explanation]
       I --> L[Attack Path + Asset Risk]
       L --> M[What-If Simulator]
       M --> N[Re-Forecast]
       N --> O[Before/After Comparison]
       O --> P[Streamlit Dashboard]
       P --> Q[Evaluation & Benchmarking]




4. Input Data Design

Accepted inputs: CIC-IDS2018 CSV flow records (primary),
PCAP files (optional, for packet-derived features), other
schema-compatible flow/NetFlow exports. Each input is
tagged with its type so downstream stages know which
feature groups are obtainable.



5. Data Validation Design

Validation checks: file readability, required-column presence,
column types, timestamp parseability, NaN/Inf detection,
duplicate-record detection, empty/oversized input handling.
Validation produces a structured report (pass/fail per
check) rather than raising uncaught exceptions; invalid rows


are quarantined and logged, never silently dropped in bulk
without a count being reported.



6. Feature Extraction Design

Two independent extractors: a flow-feature extractor
(always active for CSV/NetFlow input) and a packet-feature
extractor (active only when PCAP input is supplied). Each
extractor outputs values plus a per-feature availability tag.
The two outputs are merged into one feature matrix at the
state-vector construction stage.



7. Flow-Level Feature Design

Derived directly from flow records: source/destination IP
and port, protocol, TCP flag bitmask decomposition,
bytes/packets per flow, flow duration, inter-arrival-time
statistics (mean/variance/max), bidirectional flow ratios.
These are always available whenever flow-level input is
present.



8. Packet-Derived Feature Design

Derived from PCAP parsing (e.g., Scapy/PyShark): TTL and
TTL variance, TCP window size, IP fragmentation flags,
payload-size distribution, port-scan signatures (sequential
vs randomized access), retransmission counts. These
require raw packet capture; they are unavailable from flow-
only CSV input.



9. Feature Availability Handling


Every feature carries metadata: {name, group:
flow|packet, extraction_method, availability:
available|unavailable|optional, unit,
preprocessing} . Downstream components (state builder,
model, dashboard) must read this metadata and never treat
an unavailable feature as zero or default without explicit
configuration — absence is represented distinctly from a
legitimate zero value.



10. Time-Window Design

Traffic is aggregated into configurable, non-overlapping time
windows (default: 1 minute), e.g. W1: 09:00–09:01 , W2:
09:01–09:02 . A record with timestamp t belongs to the
window [floor(t/window_size)*window_size,
+window_size) . Boundary timestamps map to exactly one
window (start-inclusive, end-exclusive).



11. Network State Representation

Each window produces one state vector aggregating all
flows observed in that window, e.g. (illustrative, final list is
configuration-driven): flow_count, syn_ratio, ack_ratio,
rst_ratio, bytes_rate, packets_rate, unique_src_count,
unique_dst_count, port_activity summary, mean_IAT, plus
any available packet-derived aggregates. The exact vector is
defined in configuration, not hard-coded in documentation.



12. Temporal Sequence Construction


Given a sequence of per-window states S_1 … S_n and a
configured sequence length L , training examples are built
as [S_t-L+1 … S_t] → target derived from S_t+1
(and/or further steps for multi-step targets) .
Sequences are built respecting chronological order only; no
sequence may span windows out of order.



13. LSTM World Model Design

The LSTM ingests a (sequence_length, feature_dim)
tensor per example, maintains a hidden state across the
sequence, and outputs either a predicted next-state vector, a
progression/probability score, or both (architecture-
configurable). It is described as a learned temporal
approximation of network-state transition dynamics, not a
complete causal simulation of the physical network.



14. State Transition Learning

The model is trained with supervised dynamics learning:
ground-truth "next state" and/or progression labels are
derived from the dataset's attack timeline annotations. Loss
combines (as configured) next-state reconstruction error
and/or progression classification loss.



15. K-Step Forecasting Design

Forecasting rolls the model forward autoregressively: the
predicted state at step t+1 becomes part of the input
context for predicting t+2 , up to configured K . Each step's
output records whether it was produced from real observed


input or from a previously predicted state, so predicted-on-
predicted uncertainty growth is visible and not hidden.



16. Future State Probability Design

Each future step yields an infiltration/progression
probability derived from model output (e.g., a calibrated
sigmoid/softmax head). Probabilities are reported per
window across the K-step horizon, forming a probability
timeline rather than a single number.



17. Attack Progression Design

Progression is represented as a monotonically-trackable
trajectory across predicted states/stages over the forecast
horizon, so a defender can see whether the trend is
escalating, plateauing, or receding.



18. Attack Stage Mapping

A separate mapping layer converts predicted state/feature
evidence into one of the five defined stages
(Reconnaissance, Initial Access, Lateral Movement,
Command & Control, Exfiltration). This mapping is
documented as inferred, not directly predicted by the LSTM
— the model outputs state/progression signals; the
mapping layer interprets them against stage-indicative
feature patterns.



19. Attack Path Generation


Candidate paths (e.g., PC → File Server → Database )
are generated from predicted state trajectories combined
with configured network topology/asset relationships.
Multiple candidate paths may be generated per forecast.



20. Path Probability Ranking

Each candidate path is scored using the corresponding
progression probabilities along its nodes; paths are
deduplicated and ranked descending by score. Only the top-
K paths are surfaced to the dashboard.



21. Asset Risk Design

Asset risk combines predicted traffic relationships with
configured asset metadata (if provided). If no asset
criticality configuration exists, the system reports "asset
risk: not evaluated" rather than inventing a criticality value.



22. SHAP Explainability Design

For each prediction, SHAP (or an equivalent attribution
method suited to the model type) computes per-feature
contribution values against the same feature vector used
for inference. Output is a ranked list of top contributing
features with attribution weights — never text invented
independently of these values.



23. LLM Explanation Design


  Prediction → SHAP evidence → structured
  evidence object → (optional) LLM → human-
  readable explanation


The LLM receives only the structured evidence object
(feature names, attribution weights, predicted
stage/probability) and produces prose. It cannot alter the
probability, stage, path ranking, or asset risk — those are
fixed before the LLM is invoked. If the LLM is
disabled/unavailable, the structured evidence is shown
directly instead of prose.



24. Timeline Reconstruction

The dashboard reconstructs a timeline combining: observed
windows (actual data), predicted windows (model forecast),
and simulated windows (What-If output) — each visually and
structurally distinguished.



25. Attack Graph Design

A graph view renders nodes (hosts/assets) and edges
(observed or predicted communication/progression) with
edge weights from path probabilities. Observed edges and
predicted edges are styled differently.



26. What-If Defence Simulation

Given the current state, a simulated intervention (e.g.,
"Restrict SMB") applies a documented transformation to a
copy of the relevant features (e.g., zeroing/reducing SMB-


port flow volume), producing a new hypothetical state.
Example — "Isolate Host": communication features
associated with the target host are reduced/removed in the
copied state. The original observed state is never mutated.



27. Before-vs-After Re-Forecasting

The same World Model is run on both the original state and
the simulated state, using the same horizon K . Outputs
(probability timeline, stage, paths) are diffed to produce a
before/after comparison, quantifying the estimated effect
of the intervention.



28. Defence Recommendation Logic

Recommendations are derived by ranking candidate
interventions by the magnitude of risk reduction they
produce in simulation, subject to the interventions actually
modeled (initially: Restrict SMB, Isolate Host).
Recommendations are presented as simulated estimates,
not guarantees.



29. Streamlit Dashboard Design

Pages/sections: Current Risk, Forecast Timeline, Attack
Paths, MITRE Stage, SHAP Explanation, Asset Risk,
Observed/Predicted Timeline, Attack Graph, What-If
Simulator, Before/After Comparison, Defence
Recommendation, Baseline Benchmark. Each section calls
a service-layer function that wraps the pipeline; no page
computes predictions inline.


User interaction flow: load/select dataset sample →
validation summary → view current state & risk → view
forecast & stage & paths → inspect SHAP explanation
(optionally LLM prose) → optionally select a What-If
intervention → view before/after comparison and
recommendation → view benchmark metrics.



30. Offline Execution Design

All core components (feature extraction, LSTM inference,
SHAP, dashboard) use locally installed libraries and locally
stored model artifacts. No component in the core path
performs a network call. The LLM layer, if used, is the only
optional network-dependent (or local-model) component
and is feature-flagged off by default in offline mode.



31. Model Persistence

Model artifacts saved together: model weights,
architecture/config, feature list + ordering,
scaler/normalization parameters, training metadata
(dataset version, date, git commit if available). Loading
validates that the saved feature list matches the current
inference feature list before running inference.



32. Configuration Management

A single configuration source (e.g., YAML/JSON) controls:
window size, sequence length, feature list, forecast horizon
K, model hyperparameters, dataset paths, and enabled


optional components (LLM, packet extraction). No
configuration value is hard-coded inside business logic.



33. Error Handling

Each pipeline stage returns either a result or a structured
error (not a bare exception surfaced to the UI). The
dashboard displays a graceful degraded state (e.g., "SHAP
unavailable") rather than crashing when an optional
component fails.



34. Logging

Preprocessing decisions (dropped rows, imputed values,
detected schema issues), model training runs, and
simulation actions are logged with timestamps for
traceability and reproducibility review.



35. Performance Considerations

Feature extraction and windowing are vectorized where
possible; the LSTM inference and K-step rollout are bounded
in cost by sequence length × K; dashboard recomputation is
triggered only when inputs change, not on every UI
interaction.



36. Security Considerations

The What-If simulator has no dependency on, or code path
to, any real network control API. All simulated-state objects
are structurally distinct from observed-state objects so they


cannot be accidentally treated as real telemetry
downstream.



37. Data Leakage Prevention

Time-based (not random) train/validation/test splits;
scalers fit only on training-period data; sequence
construction never includes a target window's own features
in its input; label-derived fields are excluded from the
feature set used for prediction.



38. Reproducibility

Fixed random seeds where applicable; versioned config and
model artifacts; deterministic preprocessing given the same
input and config; separate, independently runnable
training/inference/evaluation scripts.



39. Design Limitations

The LSTM learns a practical, supervised approximation of
state-transition dynamics, not a full causal simulation of
network physics. Attack-stage mapping is an inferred layer,
not a directly-labelled model output. Packet-derived features
are only as good as the availability of PCAP input. Asset risk
is only as good as the configured asset metadata provided.



40. Future Extensions

Graph Neural Network or Transformer variants of the World
Model; calibrated uncertainty quantification; richer topology-


aware asset risk; additional datasets; additional What-If
intervention types; a justified, separately-ADR'd tamper-
evident logging mechanism if a real need for it is identified.


