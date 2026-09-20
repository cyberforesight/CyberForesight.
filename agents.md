# AGENTS.md — CyberForeSight AI

(This file is also intended to be used as CLAUDE.md if the project is developed with Claude Code.)

1. Project Context Project: CyberForeSight AI Event: Smart India Hackathon 2026 Problem Statement: 26153 — AI based Network Attack Forecasting from Network Traffic Data Organization: National Technical Research Organisation (NTRO) Category: Software | Theme: Blockchain & Cybersecurity The project is an offline defensive cybersecurity research prototype. It is NOT a traditional intrusion classifier. It is a World Model that learns temporal network-state transition dynamics and forecasts future attack progression from network traffic data. Conceptual pipeline: Traffic → validation → feature extraction → time windows → network state → temporal sequences → LSTM World Model → K- step forecasting → attack progression probability → attack-stage mapping → SHAP explanation → optional grounded LLM explanation → asset/path risk → simulated intervention → re-forecast → Streamlit dashboard → evaluation

2. Mission Build a working LSTM-based World Model that learns P(S_t+1 | S_t) over time-windowed network states, performs K-step forward simulation, maps predicted progression to MITRE ATT&CK stages, explains predictions with SHAP, supports simulation-only What-If defence testing, and benchmarks against a Logistic Regression baseline — all running offline, exposed through a Streamlit dashboard.
3. Required Reading (in order, before acting)
1. PRD.md — before any implementation decision.
2. DESIGN.md and ARCHITECTURE.md — before modifying architecture.
3. RULES.md — before changing project conventions.
4. DECISIONS.md — before proposing architecture changes.
5. TESTING.md — before implementing or modifying testable modules.
Preserve the distinction between requirements (PRD), architecture/design (ARCHITECTURE/DESIGN), decisions (DECISIONS), and implementation. Do not blur these into one document.

4. Development Workflow Do not invent unsupported features. Do not silently change the model architecture. Do not replace the LSTM World Model with a static classifier. Do not remove temporal sequencing merely to simplify implementation.

Keep Logistic Regression as the baseline. Keep core functionality offline. Avoid unnecessary dependencies; prefer simple, explainable implementations. Do not introduce a new framework when an existing dependency already solves the problem. Do not rewrite working code unnecessarily — make minimal, targeted changes. Before modifying shared interfaces, inspect all callers. Keep data schemas explicit; validate data before model processing. Handle missing, invalid, NaN, infinite, malformed, and unexpected columns safely — never silently corrupt or drop data. Log important preprocessing decisions; make preprocessing reproducible. Prevent train/test data leakage; respect temporal ordering in splits. Never let future information leak into past-state features. Keep training and inference pipelines consistent; save preprocessing/scaling config with the model. Make model loading deterministic and version-aware.

5. Architecture Rules Follow the modular flow: input adapters → preprocessing → feature engineering → temporal dataset builder → model service → forecast engine → explainability → simulation → dashboard. No unnecessary microservices, message queues, Kubernetes, or cloud dependencies.

No blockchain unless explicitly justified by a new ADR. UI (Streamlit) must consume a service/application layer — never manipulate the model or pipeline directly.

6. ML / World Model Rules LSTM is the primary temporal World Model; preserve S_t → S_t+1 → S_t+2 →.... Support K-step forecasting. Clearly distinguish: current observed state, predicted future state, attack probability, attack stage, confidence/uncertainty. Do not use language such as "certainty", "guaranteed", "confirmed attacker", or "causal proof" unless technically justified by the actual method. Logistic Regression remains the baseline for comparison — do not remove it.
7. Data Rules Primary dataset: CIC-IDS2018. Do not assume packet-level fields are available in CSVs; maintain explicit feature metadata: source feature, feature type, extraction method, availability, units, preprocessing. If packet-level data is unavailable, do not fabricate it — mark it unavailable.
8. Cybersecurity Safety This project is strictly defensive. Never add code that: attacks external systems or scans public targets,

||9. Explainability Rules inputs — never invented.|clearly bounded simulation layer.|steals credentials or deploys malware, performs unauthorized exploitation, have no code path to a real network control plane.|modifies real firewall rules or isolates real hosts,|executes real defensive commands without an explicit, The What-If simulator must remain purely simulated — it must SHAP/feature attribution must be tied to actual model||||
|---|---|---|---|---|---|---|---|---|
||Explanation pipeline:|||ML|prediction|→ SHAP|evidence|→|
|structured|human-readable||explanation|explanation.|input|→ (optional)|LLM →||
|Never:|ML (minus prose explanation). 10. Dashboard Rules Streamlit, modular pages.|prediction|||→ LLM If the LLM is unavailable, the system must still function Business/model logic must not be embedded in UI code.|decides|prediction.||
|Prefer|UI → model/pipeline, never manipulation.||||service/application UI →|layer → direct low-level|model||
||11. Testing Rules|||validation, preprocessing, feature extraction, time-window|Every important module should have tests, at minimum covering:||||

creation, state construction, sequence generation, model loading, forecasting, probability output, SHAP integration, stage mapping, path generation, simulation, and before/after comparison.

12. Code Quality Before considering a task complete:
1. Run relevant tests.
2. Run lint/type checks if configured.
3. Verify imports and configuration.
4. Verify the pipeline works on a small sample.
5. Verify no obvious data leakage.
6. Verify output schema.
7. Update documentation when behaviour changes.
When something fails: identify root cause, make the smallest appropriate fix, rerun the relevant test. Do not hide errors with broad exception handling.

13. Documentation When implementing a significant feature, update the relevant documentation. Do not duplicate architecture information unnecessarily across multiple files — each file (PRD/DESIGN/ARCHITECTURE/RULES/DECISIONS/TESTING/MEMORY) owns its own concern.
14. Dependency Rules Avoid unnecessary dependencies.

Do not add a package that duplicates functionality already available in an existing dependency. Justify any new core dependency in DECISIONS.md.

15. Git / Change Rules Make minimal, targeted commits tied to a single concern. Do not mix refactors with behaviour changes in the same change set. Reference the relevant PRD/RULES/DECISIONS section in commit messages where useful.
16. Definition of Done A task is done only when: it satisfies the relevant PRD requirement, follows RULES.md, has passing tests, has updated documentation where behaviour changed, and has been verified end-to-end on a small sample without fabricated data or hidden failures.
17. Conflict Resolution If instructions conflict, apply this priority order:
1. Explicit user request
2. PRD.md
3. RULES.md
4. ARCHITECTURE.md
5. DESIGN.md
6. DECISIONS.md
7. Existing working implementation
8. Agent assumptions

If an important requirement is ambiguous, explain the ambiguity before making a major architectural change.

Output Style for Reporting Work

When reporting implementation work, state: what changed, which files changed, why, what tests were performed, and any limitations. Do not claim a feature is complete if it is only stubbed.