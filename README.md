# DIVA: Decision-Informed Value-of-information Acquisition

Code and environments for the paper *Not All Information Is Worth Asking For: Decision-Informed Information Seeking with LLMs*.

## Layout
- `envs_synth/`: Controlled environments (four domains, stated reliabilities): generators, test/dev scenarios (`data/`), and exact DP / myopic reference policies.
- `envs_tab/`: Knowledge-required environments on UCI records (Taiwan credit, German credit, CDC diabetes, heart disease): feature binarization, reference models, and scenarios.
- `envs_real/`: Knowledge-required environments on SEC filings (return and volatility targets): scenario construction and reference models. Raw EDGAR downloads are not included; `envs_real/scripts` re-creates them from the public SEC EDGAR APIs.
- `baselines/`: LLM baselines (Direct, CoT, SC, ReAct, Ask-P, VoI-P, VoI-Ask, BED-LLM, PAL, and further controls) and the episode runner.
- `scripts/`: DIVA, DIVA-U, and DIVA-DP (elicitation, decision value, stopping), evaluation and aggregation, and the scripts that generate every table and figure of the paper.
- `serving/`: vLLM serving helper and client used for all open-weight LLMs.
- `results/`: Aggregated results behind the tables (large per-scenario files are omitted).
- `PREREG_*.md`, `ERRATA.md`: Pre-registration amendments and deviations (Appendix H of the paper).

## Usage
1. Serve an open-weight LLM with vLLM (see `serving/serve_one.sh`; checkpoints in Appendix C.5 of the paper).
2. Run DIVA and the baselines on an environment, e.g., `python scripts/eval_test.py` (controlled), `python scripts/eval_tab.py` (UCI records), `python scripts/eval_real.py` (SEC filings), and `python baselines/run.py`.
3. Aggregate and build the tables and figures with `python scripts/make_tables.py`, `python scripts/make_tables_app.py`, and `python scripts/figs_all.py`.

Paths and server names were removed for anonymity; set the model endpoints in `serving/client.py`.

## License
Code: For research use only during review. UCI datasets are redistributed under their original licenses; SEC filings are public U.S. government data.
