# envs_synth: generative information environments with EXACT decision value (VOI), EIG and DP stopping

Owner: envs_synth helper. Consumers: the lead's harness, baselines/. Follows `../DESIGN.md` section 2. The binary track
is that spec's interface. The multi-state (ms) track is a superset that adds factored hidden states, 3 actions,
decision contexts (horizons), multi-outcome evidence, and qualitative priors.

All labels come from exact enumeration in `core.py`. No LLM is involved in any label. The LLM QC round-trip
(`qc_roundtrip.py`) only checks the verbalization and acts as a filter.

## 1. What is ready

| path | content |
|---|---|
| `data/binary/{finance,credit,clinical,legal}_{dev,test}.jsonl.gz` | Binary track: 2000 dev rows and 3000 test rows per domain |
| `data/binary/*_pilot.jsonl.gz`, `*_test1000.jsonl.gz` | First 150 rows of dev and first 1000 rows of test. These are the lead's DESIGN sizes. Rows are shuffled by group, so prefixes stay balanced and a group is never cut. |
| `data/ms/{finance,credit,clinical,legal}_{dev,test}.jsonl.gz` (+ pilot, test1000) | Multi-state track, same sizes |
| `data/STATS.json` | Per-file counts: strata, families, gold-stop rate, myopic/DP agreement, EIG≠VOI argmax, qualitative robustness |
| `data/qc/{track}_{file}.{model}.jsonl.gz` | QC round-trip results keyed by `id`. Take the union over files, because pilot ⊂ dev and test1000 ⊂ test. |
| `results/ref_policies_{track}_test.json` | Exact reference policies (oracle DP, myopic VOI, EIG-greedy, …) scored by the evaluation API |

Code:
- `core.py` is the engine: beliefs, VOI, EIG, EIG-about-the-optimal-action, DP and episodes.
- `gen_binary.py` and `gen_ms.py` are the generators. `domains_bin.py` (a copy of the lead's draft library) and `domains_ms.py` hold the domain specs.
- `eval_api.py` is the evaluation API and the interactive `Env`.
- `sim.py` is the DESIGN.md simulator API.
- `qc_roundtrip.py` is the LLM extraction check.
- `ref_policies.py` runs the reference policies.
- `selftest.py` holds the independent correctness checks.
- `run_all.sh` regenerates everything; `run_qc.sh` runs QC over all splits.

Python: `python` (numpy). CPU only, except QC, which calls a served LLM.

## 2. Formal model (both tracks)

A scenario is a finite decision problem:
- A hidden state s ∈ {0..K-1} has a prior π.
- Actions a have utilities U(a,s).
- Candidate items i have likelihoods P(o | s) over outcomes o and a cost c_i. Items are conditionally independent given s.
- There is an optional request budget B.

Exact quantities at a history h (belief b):

- `EU(a|b) = Σ_s b(s) U(a,s)`, `V0 = max_a EU`
- `VOI_i = Σ_o P(o|b) max_a EU(a|b,o) − V0` (myopic and cost-free; this is the project definition)
- `net_i = VOI_i − c_i`
- `EIG_i = H(S|b) − Σ_o P(o|b) H(S|b,o)` in bits. This is information about the hidden state.
- `EIGdec_i`: the same quantity for the full-information optimal action a\*(S). This is an Entropy-Search-style decision-aware information measure.
- DP: `V*(h) = max( V0(h), max_{i unobserved, budget left} [ −c_i + Σ_o P(o|h) V*(h+(i,o)) ] )`. It is exact by memoised enumeration.

Two golds, both stored:
- `exact.gold_choice` is myopic: argmax net if it is > 0, else "S". Ties are in `gold_ties`.
- `exact.dp_first_action` is the non-myopic optimum. Ties are in `dp_first_ties`. When stop ties with acquiring, stop wins.

The two golds differ in about 17–20% of binary rows and about 10–17% of ms rows (see STATS.json). Most of these are cases where no single check pays for itself but a pair does. The lead's strict "gold stop" is **stratum S2** = (max net ≤ 0) ∧ (DP stops) ∧ (some EIG > 0.05 bits).

Strata. Each row gets one stratum, computed from exact values in this priority order:
1. **S2 (already sufficient):** gold = stop, but the checks are still informative.
2. **S1 (EIG-VOI divergent):** argmax-EIG and argmax-VOI sets are disjoint, VOI(argmax EIG) ≤ 0.5·max VOI, and max VOI > 0.
3. **S3 (standard decision-flip):** argmax EIG = argmax VOI and max net > 0.
4. **other:** everything else.

Placebo items have identical likelihood rows, so VOI = EIG = 0. Every scenario has at least one.

## 3. Binary track (`data/binary`, DESIGN.md section 2 fields)

- Y ∈ {0 benign, 1 adverse}. Actions [act, pass]. U(act,0)=+g, U(act,1)=−l, U(pass,·)=0.
- π ∈ {0.10, …, 0.60} in steps of 0.05; g ∈ {1,2,3}; l ∈ {2,3,5,8}.
- Each check i has s1 = P(finding|Y=1) ≥ s0 = P(finding|Y=0) on a 0.05 grid.
- Check kinds: strong, moderate, weak, `one_high` (finding common either way), `one_low` (rare either way) and placebo. The one-sided kinds are informative but often cannot flip the decision.
- Costs: regime ∈ {0, .05, .1, .2, .4} × min(g,l) × a per-check jitter in {.5, 1, 1.5, 2}.
- There are 6 checks plus 0–2 already-observed checks with stated reliabilities and realized values.
- All numbers are stated exactly in the text, in mixed surface forms ("about 70%", "in 7 out of 10 cases", "roughly 70 percent").

Families (`family`):

| family | share | design |
|---|---|---|
| `base` | 62% | Rejection-sampled to hit the split-level strata quotas S1 38%, S2 27%, S3 27%, other 8% (achieved exactly) |
| `cost_variant` | 12% | Groups of 3 (same `group_id`) with costs ×0.25 / ×1 / ×4 |
| `payoff_pair` | 10% | Groups of 2: same evidence, different (g,l), so the decision threshold and VOI ranking shift |
| `many_item` | 16% | 10–12 checks with budget 3–4 (DP is exact under the budget) |

Row fields:
- `id`, `track`="binary", `domain`, `split`, `family`, `group_id`, `stratum`, `prior`, `g`, `l`, `cost_regime`, `budget`, `costs{label:c}`.
- `checks[{label,name,finding_text,s1,s0,cost,is_placebo,kind}]`.
- `observed[{name,finding_text,value,s1,s0}]`, `label_order`, `actions` ([act, pass]), `states`.
- `text_parts{context,payoff,base_rate,known,options,option_lines{label:line},stop_line,budget}`. Use `option_lines` to re-assemble prompts in a permuted label order.
- `text`, the default assembled prompt body. It has no answer-format instruction; the harness adds one.
- `model`: the full numeric model used by core.py (`prior`, `utility[A][K]`, `items[{label,lik[K][O],cost,outcomes}]`, `observed`, `budget`).
- `exact`:
  - posterior: P(Y=1) after the known findings
  - `posterior_vec`, `eu_act`, `eu_pass`, `best_action`, `threshold` = g/(g+l)
  - `voi`, `eig`, `eig_dec`, `net` and `q` (DP Q-value of acquiring each label), all as {label: value}
  - `gold_choice`, `gold_ties`, `dp_value`, `dp_first_action`, `dp_first_ties`, `myopic_value`, `value_of_stopping`, `stratum`
- `episodes`: 8 stored realizations `{state, y, outcomes{label: 0/1}}`. The state is sampled from the posterior after the known findings, and every check's outcome is sampled given that state.

## 4. Multi-state track (`data/ms`)

**Hidden state.** It has 3 a-priori independent factors per domain. Joint K is 12 (clinical, legal) or 18 (finance, credit). The text states each factor's prior separately. **Utility** is additive over factors: U(a,s) = const[a] + Σ_f W[a][f][level_f(s)]. The baseline action is worth 0.

**Decision context.** It sets W. In each context, one or more factors have W ≡ 0. An item that only informs about such a factor has **VOI = 0 exactly while EIG > 0**. This is the designed information trap, and it holds by construction: factors are independent and U does not depend on that factor. `selftest.py` asserts it.

| domain | actions | factors (levels) | contexts (irrelevant factor) |
|---|---|---|---|
| finance (equity around earnings) | BUY / HOLD / SELL (short) | revenue surprise (3), guidance (3), earnings quality (2) | horizon 1 day (quality), 3 months (surprise), 12 months (surprise) |
| credit (loan approval) | APPROVE / APPROVE_REDUCED / DECLINE | capacity (3), collateral (2), application accuracy (2) | secured (none), unsecured (collateral), government-guaranteed (capacity, collateral) |
| clinical (diagnosis/treatment) | SUPPORTIVE / ANTIBIOTICS / ADMIT | cause viral/non-infectious/bacterial (3), severity (2), drug-interaction risk (2) | amoxicillin clinic (risk), fluoroquinolone-only (none), remote no-admission (severity, risk) |
| legal (4th domain) | TRIAL / MEDIATE / SETTLE | liability case (3), provable damages (2), document retention (2) | uncapped (none), damages capped (damages), bench ruling (retention) |

**Payoffs.**
- Finance: a per-factor return model in %, BUY = +size·r, SELL = −size·r, plus trading costs, with size ∈ {$0.5M, 1M, 2M, 5M}.
- Credit and legal: $k contributions relative to DECLINE or SETTLE.
- Clinical: outcome points relative to SUPPORTIVE.

Base values are jittered ×U(0.7, 1.3) per scenario, rounded, and stated exactly.

**Items.** There are 7–11 domain items per domain (two or three outcomes, ordinal likelihoods whose strength is weak, moderate or strong) and 4–5 "salient" placebos (e.g. "Social-media sentiment scrape (trending, high volume)"). Each placebo nominally references a factor but has identical rows. Likelihood tables are stated per level of the target factor on a 0.05 grid. Costs are scale × regime {0, .25, .5, 1, 2, 3} × jitter.

**Two presentations.**
- `text_numeric` states priors as percentages.
- `text_qualitative` states priors with verbal phrases: very unlikely, unlikely, a real possibility, about as likely as not, likely, very likely. Everything else is the same.

**Robustness flag for qualitative priors.** `qual_robustness` redraws every prior 24 times uniformly inside the range of its verbal phrase and re-solves. It reports the fraction of draws that agree on `gold_choice`, `best_action`, `top_voi` and `dp_first` (DP is skipped for many_item). `robust` = all of these are ≥ 0.9. **Report qualitative-text results on `qual_robustness.robust` rows** (about 38–52% of rows, see STATS). For the qualitative condition, exact gold applies only to those rows.

Families (`family`, share of split):

| family | share | design |
|---|---|---|
| `trap` | 17% | Rejection-sampled until the argmax-EIG item has VOI = 0 (EIG > 0.05 bits) while some item has VOI > 0 |
| `sufficient` | 13% | Concentrated priors and/or high costs; accepted only if stratum S2 (gold = stop) |
| `flip` | 13% | Accepted only if S3 (argmax EIG = argmax VOI, net > 0) |
| `context_pair` | 15% | Groups of 3: the same priors, items, likelihoods and costs under all 3 contexts, with unjittered base W. The same item changes VOI across horizons. |
| `cost_variant` | 9% | Groups of 3 with costs ×0.25 / ×1 / ×4 |
| `placebo_heavy` | 6% | 3 salient placebos |
| `many_item` | 12% | 11–14 items (including 2 placebos), budget 3–4 |
| `base` | rest | Quota-sampled over S1, S2, S3 and other |

Row fields:
- Shared with the binary track: `id`, `track`="ms", `domain`, `split`, `family`, `group_id`, `stratum`, `cost_regime`, `budget`, `actions`, `costs`, `label_order`, `model`, `episodes`.
- Context and payoffs: `context`, `prior_mode`, `size_musd`, `factors[{key,name,levels,prior}]`, `W[action][factor][level]`, `const[action]`, `ret` (finance returns in %).
- `checks[{label,name,factor,outcomes,lik_by_level[level][outcome],cost,strength,is_placebo,decision_relevant_factor}]`.
- `text_parts{header,situation,beliefs_numeric,beliefs_qualitative,payoff,options,option_lines,stop_line,budget}`.
- `text` (= `text_numeric`), `text_numeric`, `text_qualitative`, `qual_robustness`.
- `flags{has_voi0_eigpos, argmax_eig_voi0, gold_stop, dp_stop, decision_flip, n_placebo}`.
- `model.states`: the factor levels of each joint state index.
- `exact`: as in the binary track, except `posterior` is the joint vector and `eu` is {action: value}.
- `episodes`: `{state, state_levels, outcomes{label: outcome index}}`.

## 5. Evaluation API (`eval_api.py`)

```python
from eval_api import evaluate, aggregate, rank_metrics, Env, read_jsonl_gz
rows = read_jsonl_gz("data/ms/finance_test.jsonl.gz")
r = evaluate(rows[0], requests=["C", "A"], decision="BUY", episode=0)   # one trajectory
agg = aggregate(list_of_results)                                        # overall + by domain/family/stratum
rank_metrics(rows[0], scores={"A": .3, "B": .1, ...})                   # Spearman vs VOI/net/EIG, top-1 checks
env = Env(rows[0], episode=0, mode="numeric")   # or "qualitative"
env.prompt(); env.request("C")  # -> "Result of (C) ...: 'firm'."
env.decide("BUY")               # -> same dict as evaluate()
```

CLI:

```
python eval_api.py --data data/ms/*_test.jsonl.gz --preds preds.jsonl.gz --out summary.json [--per_item per_item.jsonl.gz]
```

The preds rows look like this:

```
{"id", "requests":[labels in order], "decision":"BUY", "episode":0, "scores":{label: float} (optional)}
```

Outcomes revealed to the model must come from `episodes[episode]` (use `Env`) so that trajectories match the scoring.

Per-trajectory output:
- First step:
  - `voi_regret_first` (and `_norm`, divided by the utility range): max VOI − VOI(first request). If the model stops first, this is max VOI.
  - `net_regret_first`
  - `match_gold_first` and `match_dp_first`
- Policy:
  - `policy_regret`: Σ_t [V\*(h_t) − Q\*(h_t, move_t)] plus [V\*(h_T) − EU(decision)]. This is an unbiased per-episode estimate of V\*(root) − V^π(root) for any policy that is deterministic given observations. Average it over episodes. Oracle = 0, which is asserted.
- Utility:
  - `cau_expected` = EU(decision | final history) − total cost
  - `cau_realized` = U(decision, true state) − cost
  - `v_star_root`
- Stopping: `model_stop_first` vs `gold_stop`, and `under_ask` (DP would still acquire at the stop node).
- Decision: `decision_correct` (Bayes action at the final history) and `decision_regret`.
- Per request step: `chose_argmax_voi`, `chose_argmax_eig`, `divergent_node`, `eig_trap`, `chose_voi0_eigpos`, `chose_placebo`, `over_ask_dp`, `voi_rank`, `eig_rank`.
- Validity: invalid, repeated and over-budget requests are recorded and skipped. An unparseable decision is scored as the worst action.

`aggregate` adds:
- `stop_recall` / `over_ask_first`, computed on gold-stop rows
- `under_ask_first`, computed on gold-acquire rows
- `eig_trap_rate`, the share of steps at divergent nodes that chose argmax EIG
- `voi_choice_rate_divergent`, `chose_voi0_eigpos`, `chose_placebo`, `mean_voi_rank` and `mean_eig_rank`

For single-shot (non-interactive) protocols, pass `requests=[first_choice]` (or `[]` for stop) and score with the `*_first` metrics.

## 6. Simulator API (`sim.py`, DESIGN.md)

The functions are:
- `sample_y(scn, rng)`
- `reveal(scn, label, rng, y)`
- `posterior(scn, observed)`: P(Y=1) in the binary track, the joint vector in the ms track
- `eu`, `voi`, `eig`, `eig_decision`
- `dp_policy(scn, observed)`, which returns `{"move": "S", "action", "value"}` or `{"move": label, ...}`
- `best_action`

`observed` is a list of (label, value) pairs or a dict.

## 7. QC round-trip (`qc_roundtrip.py`, filter only)

A served LLM, by default Qwen3-32B-AWQ, reads the text at temperature 0 and returns JSON. It is checked against the generator.

- Binary: base rate, g, l, and per check its cost, s1 and s0.
- ms: all costs, all prior percentages, one probed likelihood cell per item and the budget. On the qualitative text, it also returns the verbal phrase for every factor level.

The client is the lead's `dcfr_lib.Client`. `render` uses the model's chat template with thinking disabled. Endpoints are read from `serving/endpoints.json` merged with `logs/servers_launched.tsv`. Runs are resumable.

```
./run_qc.sh                       # pilot -> test1000 -> test -> dev, both tracks; QC_MODEL=..., CONC=... to override
```

Use `pass` as an optional filter (for example, report on QC-pass rows as a robustness check). A QC failure means the reader model misread the text. It does not mean the label is wrong.

## 8. Regenerating and checking

```
./run_all.sh                       # all splits, both tracks (about 7 min on 96 cores), then make_aux.py (pilot/test1000/STATS)
python selftest.py data/ms/*_test.jsonl.gz --n 400     # independent checks (plain-loop VOI/EIG, Monte Carlo VOI,
                                                        # placebo/irrelevant-factor VOI=0, utilities re-derived from the
                                                        # stated numbers, oracle policy regret = 0 on every episode)
python ref_policies.py --files data/ms/*_test.jsonl.gz --out results/ref_policies_ms_test.json
```

Seeds are fixed per (domain, split, family, index), so generation is deterministic. Dev and test use disjoint seed streams.

## 9. Known limitations (state them in the paper)

- Factors are independent and utilities additive in the ms track. This keeps VOI = 0 exact for the trap items, but it simplifies real dependence. The binary track has no factor structure; there, VOI = 0 arises from one-sided checks or from a prior far from the threshold.
- The likelihood and payoff numbers are stated exactly. Only the ms priors also come in a qualitative version. "Realistic" means filing-style or clinical-style framing and naming; the item names are not real documents.
- In the binary track the myopic gold says stop in about 47% of rows, but the DP says stop in only about 34%. In the ms credit domain stopping is optimal in about 52–56% of rows, because concentrated priors are common there. Use the stratum or `dp_stop`, and report per-family results.

## 10. Progress log (KST)

- 2026-10-08 23:56: Read DESIGN.md, NOVELTY_T1, STATUS, NEW_IDEA §1 and the lead's domains.py. Planned the binary track (DESIGN interface) plus the ms superset.
- 2026-10-09 00:05: core.py, gen_binary.py, domains_ms.py, gen_ms.py, eval_api.py, sim.py and selftest.py written. Self-test on 360 sample rows: 2390 items and 2880 oracle episodes, 0 failures.
- 2026-10-09 00:16: Full generation done: binary and ms × 4 domains × (dev 2000, test 3000) = 40,000 rows, plus pilot and test1000 subsets and STATS.json. 50 MB on disk.
- 2026-10-09 00:24: QC smoke test with Qwen3-32B-AWQ: binary finance 18/20 pass, ms credit 19/20 pass. The full QC run (`run_qc.sh`), the full self-test (400 rows per file) and the reference policies were launched in the background.
