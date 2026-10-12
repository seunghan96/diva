# envs_real: market-grounded finance slice (InfoSeek, NAACL 2027 sprint)

Owner: envs_real helper. The interfaces follow `../DESIGN.md` §3. The rows use the same schema as `../envs_synth` (binary
track), so the lead's harness and `envs_synth/eval_api.py` run on them unchanged. `track` is `"real"`.

## What this slice is
- **Unit.** One S&P 500 firm-quarter at its **earnings announcement**: the first 8-K Item 2.02 filed after the fiscal-quarter end
  and no later than that quarter's 10-Q/10-K. The universe is every S&P 500 member at any time from 2015-01 to 2026-09, taken
  point-in-time from the r1 membership layer, so firms that later left the index are included.
- **Decision.** Binary act/pass under a payoff (g, l), exactly as in envs_synth §1.
  - `ret_h`: **BUY** and hold for h trading days, or PASS. The adverse case is Y = 1 when the market-model abnormal
    return CAR[0, h-1] < 0.
  - `vol_h`: **SELL VOLATILITY**, or PASS. The adverse case is Y = 1 when the log ratio of post-announcement abnormal
    volatility (over h days) to pre-announcement abnormal volatility is above its training-period median.
  - Horizons h are 1, 5, 20 and 60 trading days. Day 0 is the first session whose close-to-close return can reflect the
    release.
- **Candidate items.** There are 19 items: 17 real and 2 placebo. Each is verbalized with its actual value and binned (terciles from
  the training years, or fixed categories):

  | item | source |
  |---|---|
  | rev_growth, gm_change, om_change, eps_sue (time-series EPS surprise proxy), fcf_margin, accruals, lev_change, capex_growth, buyback_yield, rnd_intensity | SEC XBRL companyfacts |
  | analyst_surprise (EPS vs consensus) | Yahoo earnings calendar; coverage is limited to currently listed names |
  | guidance (raised / maintained-or-new / lowered-or-withdrawn / none), segments (mostly growing / mixed / mostly declining / not reported), tone | LLM extraction from the masked 8-K EX-99 press release |
  | mom60, rv_pre, size_dvol | prices, using sessions ≤ day0-2 only |
  | placebo_weekday, placebo_code | irrelevant by construction (VOI = EIG = 0 exactly) |
- **Splits** (by day 0):
  - train: 2015–2022
  - calib: 2023 (Platt recalibration and early stopping)
  - dev: 2024
  - **test: 2025-01-01 → 2026-07 (post-cutoff)**

  60-day outcomes need prices up to 2026-09-30. They are NaN when unavailable, and those rows are dropped for that target.

## Point-in-time and masking rules
- **XBRL vintages.** Every XBRL value is the *first-filed vintage* of its (concept, start, end) fact. Restatements are never used.
- **Prior-period values** (q−4, lags, balance sheets a year earlier) must have been filed on or before the announcement date.
- **Current-quarter values** come from that quarter's 10-Q/10-K, which is filed 0–75 days after the release. The release discloses
  the headline numbers at time t, so this is a **release-time proxy**. `scripts/05_validate.py` measures how often the
  XBRL revenue and EPS literally appear in the release text (`voi/validation.json → proxy_*`).
- **Announcement time** is the earliest of two timestamps: the 8-K acceptance time (EDGAR, UTC) and the Yahoo earnings
  timestamp within the 3 days before it. Day 0 is that session if the time is before 16:00 ET, otherwise the next session.
- **Abnormal returns** use a market model against SPY. Beta is estimated on [−250, −21], shrunk as 0.67·β + 0.33, and needs at
  least 100 observations. Prices are LSEG total-return daily data from the r1 equity layer.
- **Masking** (`scripts/mask.py`) is applied to every text an LLM sees (the extraction inputs and all prompts):
  - company names and tickers become "the Company" / [TICKER];
  - executives become [PERSON];
  - datelines become [DATELINE];
  - calendar dates become [DATE];
  - years become relative tokens ([Y+0] is the release year);
  - URLs, phones and e-mails become [URL] / [CONTACT].

  Prompts state only the GICS sector. Segment or brand names inside quotes can still identify mega-caps, so treat masking as
  *reduced*, not blind.

## Reference decision model and empirical VOI
- **Outcome model** p(Y | E): LightGBM per target.
  - Features are item bins plus the sector.
  - It is trained on 2015–2022 with *random feature masking*: 6 copies per row, each keeping a random 0–5 available items,
    or all of them with probability 0.15. This lets one model condition on any observed subset.
  - Early stopping and Platt recalibration use 2023. Placebos are hidden from the model, so their VOI is 0 by construction.
- **Value model** p(v_e | E): one LightGBM multiclass model per item, trained the same way with item e removed.
- **Preposterior.** For instance i with observed set E (sector plus 0–2 observed items; P = .4/.4/.2) and item e:
  - p~(Y | E, e=v) is p(Y | E, e=v) shifted by one log-odds constant δ so that Σ_v p(v|E) p~(Y|E,v) = p(Y|E). This gives a
    coherent martingale. |δ| is reported as `coherence_abs_delta_logodds_mean`.
  - **VOI(e|E) = Σ_v p(v|E) max_a EU(a; p~(Y|E,v)) − max_a EU(a; p(Y|E))**
  - **EIG(e|E) = H(Y|E) − Σ_v p(v|E) H(Y|E,v)** in bits. Under the model this equals the expected log-score gain.
- **Held-out realized ("empirical") VOI** on dev and test:
  - `real_util_gain` = U(a*(E, e = actual), y) − U(a*(E), y)
  - `real_logscore_gain` = log p~(y | E, e = actual) − log p(y | E)

  Both are averaged over firm-quarters with standard errors. These are market-grounded checks of the model VOI.

## Files
| path | content |
|---|---|
| `data/records_base.parquet` | PIT firm-quarter records: ids, raw item values, outcomes car_h, lvr_h, split |
| `data/items.parquet` | records plus LLM items plus bins (`b_{item}`) plus sector_code |
| `data/bins.json` | tercile cuts, bin representative values (1/6, 1/2, 5/6 quantiles), coverage, vol medians |
| `data/llm/{qwen3_32b,mistral24b}/{cik}.jsonl.gz` | LLM extraction outputs per 8-K (accn → guidance, metrics, horizon, quote, segments, tone) |
| `data/ref_models/` | LightGBM reference models (`{target}.txt` + `.platt.json`, `v_{item}.txt`) |
| `data/{target}_{split}.jsonl.gz` | **scenario rows** (calib/dev/test) for the 8 targets |
| `voi/voi_table_{target}_{split}.csv` | per-item VOI table (model VOI/EIG at E=∅ with g=l=1 and at the scenario E, realized gains ± se, argmax shares) |
| `voi/voi_matrix_{split}.csv` | item × target matrix of the same quantities |
| `voi/item_stats_{target}.parquet` | per (instance, item) VOI / EIG / realized gains |
| `voi/summary.json` | per target: AUC/log-loss/ECE (full vs sector-only), strata counts, gold-stop rate, coherence |
| `voi/validation.json`, `voi/llm_audit_sample.jsonl` | LLM-extraction validation and release-time proxy validation |
| `raw/` | EDGAR raw: universe.csv, facts/, filings/, ex99/ (press-release text, first 80k chars) |

## Scenario row schema (`data/{target}_{split}.jsonl.gz`)
These are the fields of envs_synth's binary track, plus a few specific to this slice:
- `id`, `track="real"`, `domain="finance_real"`, `family="real"`, `target`, `split`, `stratum`, `prior` (= p(Y|E)), `g`, `l`,
  `cost_regime`, `budget`, `costs{label}`
- `checks[]`. Each check is `{label, item, name, ask, cost, is_placebo, kind, bins[{bin,value,level,text}], p_bins, p_y_given_bin, actual_bin,
  actual_text, finding_text}`. `bins` provides the imagined values for IO-VOI (verbalized tercile medians).
- `observed[]`: `{item, name, text, bin}` (already known), `label_order`, `actions` ([BUY|SELL VOLATILITY, PASS]), `states`
- `text_parts{context,payoff,base_rate,known,options,option_lines,stop_line,budget}` and `text` (the full prompt). The base rate is the
  training-period adverse rate for the sector.
- `model`: the exact discrete model used by `envs_synth/core.Model` (prior, utility, items with lik[y][v], cost, outcomes)
- `exact`: posterior, eu_act/eu_pass, best_action, threshold, voi/eig/eig_dec/net/q{label}, gold_choice, gold_ties, dp_value,
  dp_first_action, dp_first_ties, myopic_value, value_of_stopping, stratum, all_items{item: {voi, eig}}
- `episodes`: a single episode holding the **actual** outcome y and the actual bin of every offered item
- `realized{y, car{h}, lvr{h}}`, `sector`, `day0`
- `_meta_do_not_show{rid, cik, ticker, period_end, accn_8k}`

Notes that differ from synth:
- **Payoffs.** (g, l) is drawn from {1,2,3} × {1,2,3,5,8}, among the pairs whose threshold g/(g+l) lies within ±0.08 of
  p(Y|E). Real signals are weak, so a payoff far from the belief would make every VOI 0.
- **Costs.** The cost regimes are {0, .005, .01, .02, .04} × min(g, l), with a per-item jitter of ×{0.5, 1, 2}. This grid is
  finer than synth's, as real VOIs are small.
- **Strata.** S1 and S3 use synth's definitions. S2 ("already sufficient") uses an EIG threshold of 0.005 bits instead of
  0.05, as real EIGs are small.
- **Exactness.**
  - First-step VOI, EIG and net values are exact with respect to the reference model.
  - DP and multi-step policy metrics assume the items are conditionally independent given (Y, E), i.e. naive Bayes over
    the per-item preposteriors. This is an approximation; report it as such.

## Usage
```python
import sys; sys.path.insert(0, './envs_real')
import eval_api as R
rows = R.load('ret_5', 'test')                 # list of dicts
env = R.Env(rows[0]); print(env.prompt()); print(env.request('B')); res = env.decide('BUY')
R.evaluate(row, ['B', 'D'], 'PASS')            # exact first-step + NB-approximate policy metrics (envs_synth semantics)
R.realized_metrics(row, ['B'], 'BUY')          # market-grounded: realized utility / cost-adjusted utility / log score
R.rank_metrics(row, {'A': .1, 'B': .3, ...})  # Spearman of model scores vs exact VOI / net / EIG
```
CLI (same prediction format as envs_synth):
`python eval_api.py --data data/ret_5_test.jsonl.gz --preds preds.jsonl.gz --out summary.json`.
Each prediction row is `{"id", "requests": [labels], "decision": action, "scores": {label: float}}`.

## Rebuild
```
R1=python; PC=python
$R1 scripts/01_fetch_sec.py A    # companyfacts + submissions (EDGAR UA declared, <= 8 req/s, shared-budget aware)
$R1 scripts/01_fetch_sec.py B    # 8-K Item 2.02 EX-99 texts
$PC scripts/02_build_records.py  # PIT firm-quarter records + outcomes
$PC scripts/03_llm_extract.py qwen3_32b Qwen/Qwen3-32B-AWQ                              # primary extractor (shared fleet)
$PC scripts/03_llm_extract.py mistral24b casperhansen/mistral-small-24b-instruct-2501-awq  # second extractor (validation)
$PC scripts/04_build_env.py      # reference models, VOI tables, scenario rows
$PC scripts/05_validate.py       # extraction agreement/grounding/market validity + release-time proxy check
```
The LLM calls go through `infoseek/serving/client.FleetClient`, which load-balances over the shared fleet. The calls use
JSON-schema constrained decoding at temperature 0, and Qwen3 runs with thinking off.

## Progress log (KST)
- 2026-10-08 23:55 Read DESIGN.md, NOVELTY_T1 and STATUS. Prices come from the internal r1 equity layer
  (`r1_fdr/data/equity`: LSEG total-return daily panel with PIT S&P 500 membership, 2005-01..2026-09-30) instead of
  Stooq. Stooq now serves a JavaScript proof-of-work page to scripts, while the r1 layer accounts for survivorship.
- 2026-10-08 23:58 Started the SEC fetch: companyfacts and submissions for 724 CIKs, i.e. all S&P 500 members since 2015.
- 2026-10-09 00:03–00:13 EDGAR returned HTTP 429 for this host's IP. The lead's `scripts/data/fetch_panel.py` was fetching
  companyfacts from the same IP at the same time.
  - Fix: the SEC client now backs off for up to about 25 min instead of giving up. It also drops to 2.5 req/s whenever the
    lead's fetch log shows SEC activity, which keeps the combined rate below 10 req/s.
  - 9 CIKs have empty facts files. Some of these may come from the throttle window; they are re-fetched after phase B.
- 2026-10-09 00:11 First partial build: 278 firms, 11,861 firm-quarters.
  - Sanity check: day-0 abnormal return has a Spearman correlation of 0.28 with the consensus EPS surprise and 0.10 with
    the time-series SUE.
- 2026-10-09 00:18 Pipeline smoke test of 04_build_env on the partial data (ret_1, no LLM items yet).
  - The model VOI ranks analyst_surprise first: E=∅, g=l=1, test mean VOI 0.081.
  - Its realized utility gain on test is 0.074 ± 0.017.
  - A placebo had a small positive model VOI (0.0025) from overfitting. Fix: placebos are now hidden from the outcome model.
- 2026-10-09 00:20 LLM extractors started; they stream over the EX-99 texts as phase B writes them.
  - Smoke test on 8 releases: Qwen3-32B and Mistral-24B agree on 7/8 guidance labels.
