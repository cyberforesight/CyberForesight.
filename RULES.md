RULES.md — CyberForeSight AI

Non-negotiable engineering, ML, data, cybersecurity,
architecture, and documentation rules.
Project: CyberForeSight AI · SIH 2026 PS 26153 · AI based
Network Attack Forecasting from Network Traffic Data

Legend: MUST = mandatory, MUST NOT = forbidden,
SHOULD = strong default, MAY = optional.



1. Core Project Rules

    The system MUST remain a defensive decision-support
    prototype.

    The system MUST NOT perform any real-world
    offensive or autonomous-response action.

    Contributors MUST read PRD.md , RULES.md ,
     ARCHITECTURE.md , DESIGN.md , and DECISIONS.md
    relevant to their change before implementing it.



2. Architecture Rules

    The pipeline MUST follow: input adapters →
    preprocessing → feature engineering → temporal
    dataset builder → model service → forecast engine →
    explainability → simulation → dashboard.

    The system MUST NOT introduce Kubernetes,
    microservices, message queues, or mandatory cloud


  infrastructure without a new ADR justifying the need.

  The system MUST NOT add a blockchain component
  unless a new ADR documents a concrete, justified
  requirement.

  UI code MUST NOT contain model or business logic; it
  MUST call a service layer.



3. Data Rules

  The primary dataset MUST be CIC-IDS2018.

  The system MUST NOT assume every packet-level
  feature exists in the CSV.

  Every feature MUST be tagged as available, packet-
  derived, unavailable, or optional-future.

  Input MUST be validated for file type, schema, required
  columns, missing values, NaN/Inf, and malformed
  timestamps before processing.

  The system MUST NOT silently discard large amounts
  of data; drops MUST be logged with counts and
  reasons.



4. Feature Engineering Rules

  Feature names and ordering MUST be identical
  between training and inference.

  Feature availability metadata MUST travel with the
  feature matrix, not be dropped after extraction.

  The system MUST NOT fabricate a value for an
  unavailable feature.


5. Time-Series Rules

  Timestamp ordering MUST be preserved throughout
  preprocessing and modelling.

  Time windows MUST be deterministic and boundary-
  tested (start-inclusive, end-exclusive).

  The system MUST NOT use random shuffling that
  destroys temporal order for any split used in
  forecasting evaluation.



6. World Model Rules

  LSTM MUST remain the primary World Model.

  The system MUST NOT reduce the World Model to a
  static per-flow classifier.

  Transformer/GNN architectures MUST NOT become the
  primary model without an explicit new ADR approving
  the change.

  The model MUST preserve and expose the distinction
  between observed, predicted, and simulated states.

  Language claiming certainty ("guaranteed", "confirmed
  attacker", "causal proof") MUST NOT be used unless
  technically justified.



7. Forecasting Rules

  K-step forecasting MUST return exactly K steps for a
  given configured K.

  Forecasting MUST NOT consume ground-truth future
  information during rollout.


   Each forecast step MUST report whether it was derived
   from real input or from a previously predicted state.



8. Baseline Rules

   Logistic Regression MUST remain the baseline model.

   The baseline MUST be trained and evaluated on an
   equivalent feature representation using the same
   evaluation protocol as the LSTM.

   The baseline MUST NOT be removed to simplify the
   codebase.



9. Explainability Rules

   Every prediction MUST carry a SHAP (or equivalent)
   attribution referencing real model inputs.

   The LLM explanation layer MUST NOT override, invent,
   or modify probability, stage, path ranking, or asset risk.

   The LLM MUST NOT invent features or attack stages.

   If SHAP is unavailable, the system MUST NOT fabricate
   an explanation — it MUST show structured evidence or
   a clear "explanation unavailable" state instead.



10. MITRE Mapping Rules

   Stage mapping MUST be documented as an inferred
   layer on top of model output, not a directly predicted
   label.

   Low-evidence or conflicting signals MUST NOT produce
   a falsely confident stage label.


11. Attack Path Rules

   Paths MUST be deduplicated before ranking.

   Path probabilities MUST be valid (finite, within a defined
   range) before being surfaced.

   Only top-K paths SHOULD be shown to avoid
   overwhelming the dashboard.



12. What-If Simulation Rules

   What-If actions MUST be simulation-only.

   The system MUST NOT modify real firewall
   configurations, isolate real hosts, or execute real
   defensive commands.

   Every simulated intervention MUST be explicitly labelled
   "Simulated" in data, logs, and UI.

   Simulation MUST operate on a copy of state; the
   original observed state MUST remain unchanged.



13. Dashboard Rules

   The dashboard MUST fail gracefully when an optional
   component (SHAP/LLM/packet extraction) is
   unavailable.

   The dashboard MUST visibly distinguish observed,
   predicted, and simulated data.

   The dashboard MUST NOT present predictions as
   guaranteed outcomes.



14. Offline Rules


  The core pipeline MUST run with no internet access.

  Any LLM component MUST be optional and MUST NOT
  break core functionality when disabled or unavailable.



15. Cybersecurity Safety Rules

  The system MUST NOT implement offensive
  capabilities: credential theft, malware, exploit
  deployment, unauthorized scanning, persistence,
  evasion, or destructive actions.

  The What-If simulator MUST have no code path to any
  real network control interface.



16. Testing Rules

  A feature MUST NOT be considered complete until
  appropriate tests exist and pass.

  Tests MUST cover, at minimum: validation,
  preprocessing, feature extraction, windowing, state
  construction, sequencing, model loading, forecasting,
  SHAP, stage mapping, path generation, simulation, and
  before/after comparison.



17. Reproducibility Rules

  Preprocessing and training MUST be reproducible given
  fixed configuration and seed.

  Model artifacts MUST be saved with their
  preprocessing/scaler configuration.

  Model artifacts MUST be versioned.


18. Dependency Rules

   New dependencies MUST be justified; duplication of
   existing dependency functionality MUST be avoided.

   Unnecessary packages MUST NOT be introduced for
   one-off convenience.



19. Documentation Rules

   When architecture or behaviour changes, the relevant
   documentation (PRD/DESIGN/ARCHITECTURE/RULES)
   MUST be updated in the same change.

   Significant architectural decisions MUST be recorded in
   DECISIONS.md .



20. Code Change Rules

   Changes MUST be minimal and targeted; unrelated
   working modules MUST NOT be rewritten incidentally.

   Shared interfaces MUST NOT be changed without
   inspecting all callers first.

   Existing functionality MUST NOT be duplicated by new
   code.



21. Git Rules

   Commits SHOULD be scoped to a single concern.

   Refactors and behaviour changes SHOULD NOT be
   mixed in the same commit.


22. Performance Rules

    Feature extraction and inference SHOULD complete
    within demo-acceptable latency on commodity
    hardware for representative sample sizes; actual
    thresholds MUST be measured and recorded, not
    assumed.



23. Error Handling Rules

    Errors MUST NOT be hidden behind broad exception
    handling.

    Root cause MUST be identified before applying a fix; the
    smallest appropriate fix MUST be applied and re-tested.



24. Definition of Done

A change is done only when it: satisfies the relevant PRD
requirement, complies with this file, has passing tests, has
updated documentation where behaviour changed, and has
been verified end-to-end on a small sample without
fabricated data or silently hidden failures.


