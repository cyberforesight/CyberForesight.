# DECISIONS.md — CyberForeSight AI

Architecture Decision Record (ADR) log. Records why decisions were made, alternatives considered, and consequences — not just what the architecture is.

Rule for future developers: when an important architectural decision changes, add a new ADR — never silently rewrite an old one. Never delete a past decision. If a decision is superseded, mark its Status as Superseded and reference the new ADR number.

## ADR-001: LSTM as the Primary World Model

Status: Accepted Date: 2026 (SIH cycle) Context: PS 26153 requires temporal state-transition modelling of network behaviour rather than static classification. Decision: Use an LSTM as the primary World Model. Alternatives Considered: Transformer, Graph Neural Network (GNN), GRU, static ML classifier. Reasoning: LSTM is well-suited to sequential state data and is practical to implement, train, and explain within the prototype's scope and timeline.

Consequences: Implementation must preserve temporal sequences and state progression; must not collapse to per- flow classification. Implementation Notes: Sequence length, hidden size, and output head are configuration-driven (see DESIGN.md §13).

# ADR-002: Logistic Regression as Baseline

Status: Accepted Decision: Use Logistic Regression as the non-temporal baseline. Reasoning: Simple, interpretable, fast to train; a meaningful reference point for whether temporal modelling adds value. Consequences: Must be evaluated on an equivalent feature representation using Precision, Recall, F1, and False Positive Rate, on the same protocol as the LSTM.

# ADR-003: CIC-IDS2018 as Primary Dataset

Status: Accepted Decision: Use CIC-IDS2018 as the primary dataset. Reasoning: Publicly available, widely used, and contains network traffic/attack data suitable for the prototype. Consequences: The pipeline must handle CIC-IDS2018's actual schema and its limitations (e.g., no raw packet capture, only flow-level CSV).

# ADR-004: Flow-Level + Packet-Derived Architecture

Status: Accepted Decision: Support both flow-level and packet-derived features via two independent extraction paths. Reasoning: The problem statement explicitly requires both levels (aggregate behaviour and timing/sequencing patterns). Consequences: Packet-level fields must never be fabricated when unavailable in the selected input; availability metadata is mandatory (see RULES.md §3–4).

# ADR-005: One-Minute Time Windows (Configurable)

Status: Accepted Decision: Use configurable time windows, initially defaulting to one minute. Reasoning: Balances interpretability with computational manageability. Consequences: Window size must be a configuration value, not hard-coded.

# ADR-006: Streamlit Dashboard

Status: Accepted Decision: Use Streamlit for the SOC dashboard. Reasoning: Fast development, strong Python/ML integration, sufficient for offline demonstration purposes.

# ADR-007: SHAP for Explainability

Status: Accepted Decision: Use SHAP (or an equivalent attribution method) for explaining predictions. Reasoning: PS 26153 explicitly requires interpretable decision support; black-box output is not acceptable.

# ADR-008: Optional LLM Explanation Layer

Status: Accepted Decision: An LLM may convert grounded SHAP evidence into human-readable prose. Critical Constraint: The LLM is not the decision-maker. The model predicts; SHAP explains evidence; the LLM only verbalizes that evidence. Consequences: Core system must function correctly without the LLM enabled.

# ADR-009: Simulation-Only Defensive Actions

Status: Accepted Decision: What-If actions (Restrict SMB, Isolate Host) are simulations only. Reasoning: The prototype is a defensive decision-support tool and must never modify real infrastructure. Consequences: No code path may exist from the simulator to any real network control interface (architectural boundary, see ARCHITECTURE.md §28).

# ADR-010: Offline-First Core

Status: Accepted Decision: Core functionality must run without any cloud API dependency. Reasoning: PS 26153 expects a fully open-source/offline- capable prototype suitable for CII environments.

# ADR-011: No Blockchain Component in Initial Architecture

Status: Accepted Decision: Do not add a blockchain component merely because the SIH theme is "Blockchain & Cybersecurity." Reasoning: Blockchain is not required by PS 26153 and would add unjustified complexity without a concrete use

case. Consequences: Any future blockchain addition requires its own ADR with a specific justified use case (e.g., tamper- evident audit logging).

# ADR-012: No GNN/Transformer in Initial MVP

Status: Accepted Decision: Do not implement GNN or Transformer as the primary model initially. Reasoning: The MVP's goal is to demonstrate the core temporal World Model concept using LSTM within the hackathon timeline. Consequences: GNN/Transformer variants are recorded as future extensions (see PRD.md §22, DESIGN.md §40).

# ADR-013: Separate ML Logic from Streamlit UI

Status: Accepted Decision: Keep model/pipeline logic independent of dashboard/UI code, connected via a service layer. Reasoning: Improves testability, reuse, and maintainability; prevents logic drift between demo and pipeline.

# ADR-014: Prevent Temporal Data Leakage

Status: Accepted Decision: Training and evaluation must respect temporal ordering; no random shuffling that breaks chronological structure for time-series splits. Reasoning: Forecasting evaluated with future information leaking into the past would produce invalid, overstated results.

# ADR-015: Predicted vs Observed vs Simulated States

Status: Accepted Decision: The system must explicitly and persistently distinguish observed, predicted, and simulated data everywhere they appear (data model, logs, UI). Reasoning: Prevents users or judges from mistaking a model forecast or a hypothetical simulation for an actual observed event.

# Decision Index

ADR Title Status

001 LSTM as the Primary World Model Accepted

|ADR|Title|Status|
|---|---|---|
|002|Logistic Regression as Baseline|Accepted|
|003|CIC-IDS2018 as Primary Dataset|Accepted|
|004|Flow-Level + Packet-Derived Architecture|Accepted|
|005|One-Minute Time Windows (Configurable)|Accepted|
|006|Streamlit Dashboard|Accepted|
|007|SHAP for Explainability|Accepted|
|008|Optional LLM Explanation Layer|Accepted|
|009|Simulation-Only Defensive Actions|Accepted|
|010|Offline-First Core|Accepted|
|011|No Blockchain Component in Initial Architecture|Accepted|
|012|No GNN/Transformer in Initial MVP|Accepted|
|013|Separate ML Logic from Streamlit UI|Accepted|
|014|Prevent Temporal Data Leakage|Accepted|
|015|Predicted vs Observed vs Simulated States|Accepted|