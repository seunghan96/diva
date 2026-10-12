# Knowledge-required environments: UCI records

Approve / decline and skip / refer decisions on four UCI datasets (Taiwan credit default, German credit, CDC diabetes indicators, heart disease). The LLM sees check descriptions, costs, payoffs, and the training base rate, but no reliability; decision values come from a reference model fitted on the training records.

- `scripts/prep.py`, `scripts/features.py`: Splits (60/20/20) and binarized findings.
- `scripts/fit_ref.py`: Reference models; `scripts/gen.py`, `scripts/gen_fresh.py`, `scripts/gen_big.py`: Scenarios.
- `raw/`, `clean/`: UCI data (original licenses); `models/`: Fitted reference models; `data/`: Scenarios.
