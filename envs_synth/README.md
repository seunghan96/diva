# Controlled environments

Synthetic decisions in four domains (equity investment, loan approval, chest-pain triage, litigation) with stated check reliabilities, so that the decision value, EIG, and the DP-optimal policy are computed exactly.

- `domains_bin.py`, `domains_ms.py`: Domain libraries (binary and multi-state tracks).
- `gen_binary.py`, `gen_ms.py`: Scenario generators; `core.py`, `sim.py`, `ref_policies.py`: Exact VOI, EIG, and DP / myopic optima.
- `qc_roundtrip.py`: LLM round-trip check of the verbalized scenarios (used only as a filter).
- `eval_api.py`: Scoring API shared by all methods.
- `data/`: Development and test scenarios (`*_dev.jsonl.gz`, `*_test.jsonl.gz`).

Regenerate with `bash run_all.sh`.
