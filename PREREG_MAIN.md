# InfoSeek main pre-registration (hashed before any TEST-split output) — 2026-10-09 00:55 KST

Frozen method (primary): FIO-VOI = factored imagined-observation decision VOI. The LLM specifies the decision model by
greedy short numeric answers (base rate; per check P(finding|adverse), P(finding|not adverse); values clipped to
[0.01, 0.99]); exact Bayes for current and imagined beliefs; decision value from the stated payoffs; acquire
argmax_e (VOI_hat(e) - cost_e) if > 0 else stop (tau = 0, myopic). Secondary: FIO-VOI-DP (exact DP on the elicited
model). Final decision: argmax expected utility under the elicited model. No test-split tuning; dev splits only.

Test splits: envs_synth binary test (4 domains x 3000), envs_synth multi-state test (if ready), the verbal-reliability
variant (if ready), and the real market-grounded slice (filings >= 2025-01-01). Baselines: random, salience, direct,
CoT, self-consistency CoT, uncertainty-stopping, EIG/BED-style (same LLM beliefs and same factored model), VoI-prompting,
FinRisk-Ask-style, ReAct, oracle.

Hypotheses (cell = model x domain; models = all served families):
H1 (gap) Direct LLM policies are no better than always-stop: first-step normalized regret >= 0.8 x always-stop regret in
   >= 2/3 of cells.
H2 (method, first step) FIO-VOI reduces normalized regret by >= 50% relative to the best LLM baseline in >= 80% of cells.
H3 (information trap) On S1 scenarios, EIG on the same elicited model chooses the argmax-EIG (non-argmax-VOI) check in
   >= 70% of cases; FIO-VOI in <= 35%.
H4 (episodes) FIO-VOI's mean cost-adjusted episode utility exceeds every LLM baseline in >= 80% of cells, and its
   budget-utility AUC (k = 0..K forced acquisitions) exceeds EIG's in >= 80% of cells.
H5 (real slice) On post-cutoff firm-quarters, FIO-VOI's empirical-VOI regret is lower than direct and EIG in >= 2/3 of
   (model, target) cells.
H6 (no stated numbers) With verbal reliabilities, FIO-VOI reduces regret by >= 30% vs the best LLM baseline in >= 2/3
   of cells.
Reporting: all cells reported in the appendix; deviations disclosed.
