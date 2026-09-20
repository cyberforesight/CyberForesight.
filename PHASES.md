# CyberForeSight AI — Build Phases (SIH26153)
**AI-based Network Attack Forecasting from Network Traffic Data — NTRO**
Timeline: 1–2 weeks | Dataset: CIC-IDS2018 (CSV, 81 columns)

---

## Scope Decisions (read this first)

- **No raw PCAP files needed.** Your CIC-IDS2018 CSVs are CICFlowMeter-extracted and already contain packet-derived features (packet length stats, TCP window size, active/idle timing). This satisfies the problem statement's flow-level + packet-level requirement without a 40GB PCAP download.
- **Core build = Phases 0–5 + 7.** Fully working, fully integrated.
- **Phase 6 (LLM explanations) and Phase 6b (standalone PCAP demo) are optional bonus work** — only attempt if the core is done early. Neither affects the core pipeline if skipped.
- **Tools:** Python, PyTorch, Scikit-learn, Streamlit, Claude Code (VS Code or terminal). Antigravity is optional/skippable.

---

## PHASE 0 — Environment Setup

1. Install Python 3.11 (python.org) — check "Add to PATH"
2. Install VS Code (code.visualstudio.com) + Python extension
3. Install Git (git-scm.com)
4. Create shared GitHub repo `cyberforesight-ai`, everyone clones it
5. Create + activate virtual environment: `python -m venv venv`
6. Install packages: `pip install pandas numpy scikit-learn torch streamlit shap matplotlib seaborn jupyter`
7. Install Node.js (nodejs.org) — needed for Claude Code
8. Install Claude Code: `npm install -g @anthropic-ai/claude-code`
9. Get Anthropic API key at console.anthropic.com → paste in on first `claude` run
10. Learn Claude Code basics: `cd` into project, activate venv, run `claude`, give one specific task at a time, review before trusting
11. (Optional, later) Google AI Studio API key at aistudio.google.com — only needed for Phase 6
12. (Optional, skippable) Antigravity IDE — if used, run Claude Code in the same project folder as a separate tool

**Blocks everything else — finish before starting Phase 1.**

---

## PHASE 1 — Data Pipeline

13. Confirm your 4 CSVs load correctly; inspect columns and label distribution
14. Clean data: drop inf/NaN rows, strip whitespace from column names
15. Build time windows: floor timestamps to 60-second buckets, aggregate flow+packet-derived features per window → this is your network "state vector" S_t
16. Label each window `is_attack_window` (any non-Benign flow in that window)
17. Normalize features with `StandardScaler`, save the scaler (`joblib.dump`) for reuse at inference time
18. Output: `data/windows/window_features_normalized.csv`

**Deliverable:** a clean, windowed, normalized CSV, ready for model training.

---

## PHASE 2 — Baseline Model

19. Train a Logistic Regression classifier on the windowed features (`is_attack_window` as target)
20. Record Precision, Recall, F1, AUC-ROC — this is your comparison baseline for Phase 7
21. Save model with `joblib.dump`

---

## PHASE 3 — LSTM World Model

22. Convert windowed data into sequences (last 5 windows → predict next window's attack probability)
23. Define `LSTMWorldModel` in PyTorch (LSTM layer → linear → sigmoid)
24. Train for ~20 epochs, evaluate on held-out test set (same metrics as baseline)
25. Save model weights (`torch.save`)
26. Build K-step rollout function: feed last sequence, predict forward K windows, output a probability at each step
27. Document clearly: rollout uses a simplified state approximation, not full next-state vector prediction — acceptable for MVP, note as future work

**Deliverable:** trained LSTM + baseline, both evaluated, numbers recorded for Phase 7.

---

## PHASE 4 — Explainability & MITRE Mapping

28. Run SHAP (`LinearExplainer`) on the baseline model, save a summary plot
29. Run SHAP (`KernelExplainer`) on the LSTM for a small sample of sequences (slower — don't run on full test set)
30. Build a rule-based MITRE ATT&CK stage mapping function (port-scan pattern → Reconnaissance, SYN flood pattern → Initial Access, high-volume transfer → Exfiltration, rapid beaconing → Command & Control)
31. Document: MITRE mapping is heuristic/rule-based since CIC-IDS2018 has no native stage labels — state this explicitly, it's a defensible scoping choice

**Deliverable:** SHAP plots + a working `map_to_mitre_stage()` function.

---

## PHASE 5 — Streamlit Dashboard & Integration

32. Build basic `app.py` skeleton: file uploader, placeholder chart, placeholder MITRE label
33. Run locally: `streamlit run app.py`
34. **Integration:** wire in the real scaler, LSTM model, SHAP output, and MITRE mapping function from Phases 1–4 — replace all placeholders with real pipeline output
35. Display: attack probability timeline (line chart), predicted MITRE stage, SHAP feature importance chart

**Budget a full day for integration — this always takes longer than expected.**

---

## PHASE 6 — Optional: LLM Explanation Layer (only if time remains)

36. Call Google AI Studio's Gemini API with SHAP top features + predicted stage + probability
37. Generate a 2-sentence plain-English explanation for the dashboard

**Skip entirely if behind schedule — not required by the problem statement.**

---

## PHASE 6b — Optional: Standalone PCAP Proof-of-Concept (only if time remains, after core is done)

38. Download a small (not 40GB) public PCAP sample — e.g. from malware-traffic-analysis.net, CTU-13 individual scenarios, or Wireshark's sample captures
39. Write a separate script (`pcap_extractor_demo.py`, kept isolated from the main pipeline) using Scapy to extract TTL variance, retransmission counts, fragment flags
40. Present as a bonus "we validated raw-PCAP extraction separately" segment — does not need to feed into the main LSTM

**Fully decoupled — do this last, and only with spare time. Nothing else depends on it.**

---

## PHASE 7 — Evaluation & Submission

41. Compile final metrics table: Logistic Regression vs LSTM (Precision, Recall, F1, AUC-ROC)
42. Write Architecture Document (max 2 pages): pipeline diagram, scoping decisions (flow+packet-derived features instead of raw PCAP; probability-only rollout; rule-based MITRE mapping), metrics table
43. Record Demo Video (max 2 min): upload CSV → show timeline → SHAP chart → MITRE stage output
44. Build 5-slide Technical Presentation: Problem → Architecture → World Model & K-step Forecasting → Explainability & MITRE Mapping → Results + Future Work
45. Push final code to GitHub with README (condensed setup steps 1–11, 32–33)
46. Submit

---

## Rough Timeline

| Days | Focus |
|---|---|
| 1 | Phase 0 |
| 2–4 | Phases 1–4 (can run in parallel across the team) |
| 5 | Phase 5 integration |
| 6–7 | Phase 7 docs, video, slides; Phase 6/6b only if ahead of schedule |

**Golden rule:** don't start Phase 5 integration until Phases 1–4 each produce a working, tested output on their own. Integration bugs are much worse when you're also still debugging the individual pieces.
