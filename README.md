# DIVA: Decision-Informed Value-of-information Acquisition

Code for *Not All Information Is Worth Asking For: Decision-Informed Information Seeking with LLMs*.

DIVA lets an LLM specify a small decision model through short numeric questions and acquires a check only if its decision value exceeds its cost; DIVA-U averages the decision value over plausible strengths of each check, and DIVA-DP plans every remaining acquisition with exact dynamic programming.

## Structure
```
envs_synth/   controlled environments (stated reliabilities, exact DP optimum)
envs_tab/     knowledge-required environments on UCI records
envs_real/    knowledge-required environments on SEC filings
scripts/      DIVA (elicitation, decision value, stopping), evaluation, and aggregation
baselines/    LLM baselines (Direct, CoT, SC, ReAct, Ask-P, VoI-P, VoI-Ask, BED-LLM, PAL, ...)
serving/      vLLM serving script and client
```

## Setup
```
pip install vllm numpy pandas scipy scikit-learn lightgbm pyarrow matplotlib
bash serving/serve_one.sh   # serve an open-weight LLM (checkpoints in Appendix C.5 of the paper)
```
Set the served endpoints in `serving/client.py`.

## Running DIVA
```
# 1. Elicit the decision model (short numeric questions)
python scripts/run_factored.py      # controlled environments
python scripts/run_factored_k.py    # UCI records
python scripts/run_factored_real.py # SEC filings
# 2. Score DIVA, DIVA-U, DIVA-DP and the references
python scripts/eval_test.py         # controlled
python scripts/eval_tab.py          # UCI records
python scripts/eval_real.py         # SEC filings
python scripts/episodes_know.py     # full acquire-then-decide episodes
```

## Baselines
```
python baselines/run.py --model <served model> --methods all --env envs_synth/data/binary/finance_test.jsonl.gz
```

## Tables and figures
```
python scripts/agg_bin.py && python scripts/agg_know_main.py && python scripts/agg_episodes.py
python scripts/make_tables.py && python scripts/make_tables_app.py && python scripts/figs_all.py
```

## Data
UCI datasets are redistributed under their original licenses. SEC filings are public; `envs_real/scripts` rebuilds the records from SEC EDGAR.
