# Knowledge-required tracks: frozen DIVA configuration and hypotheses (before any TEST scoring of these tracks)
Frozen 2026-10-09 ~01:00 KST. Selection used ONLY dev splits (envs_tab *_dev, envs_real *_dev): 26 (env, model) cells,
script scripts/select_knowledge.py, output results/select_knowledge.json, log logs/select_k1.log.

Selected configuration (lowest mean DP-regret ratio to always-stop on dev, 0.539, better than always-stop in 26/26 cells):
- elicitation form for binary checks: conditional risks (P(adverse | finding present / absent)) + finding prevalence,
  converted to likelihoods by Bayes; binned SEC checks: conditional risk per bin (uniform tercile prior);
- shrinkage lambda = 0.75 of every elicited likelihood pair toward its mean (conditional risk toward the base rate);
- single greedy specification (no ensemble), stop rule tau = 0, myopic acquisition (DIVA-DP reported as a variant).
Secondary (reported, not primary): robust spec-ensemble stopping q = 0.5, lambda = 0.75 (dev ratio 0.545); unshrunk
DIVA (lambda = 1).
Note: the controlled-track configuration (PREREG_MAIN.md: likelihood form, no shrinkage) is unchanged; lambda is used
only where likelihoods are not stated (disclosed as a knowledge-required adaptation chosen on dev).

Test hypotheses (cell = model x env; envs = taiwan, cdc, german, heart, SEC ret_1, ret_5, vol_5, vol_20):
K1 DIVA's DP-regret is lower than always-stop in >= 75% of cells, mean ratio <= 0.75.
K2 DIVA's DP-regret is lower than direct LLM choice and than Fac-EIG in >= 80% of cells.
K3 DIVA's over-acquisition rate is lower than direct's in >= 80% of cells.
