# Multi-state track amendment (dev-diagnosed; before scoring the fresh ms test slice)
Date 2026-10-09 ~02:35 KST. Disclosure: an interim ms TEST aggregate (Mistral-7B, R1-Qwen-7B; rows 0-499) was viewed and
showed DIVA-ms underperforming direct/Fac-EIG. Diagnosis on ms DEV only (scripts/diag_ms.py, logs/diagdev_ms_*.log,
3 models x 4 domains x 150): with LLM-elicited beliefs/likelihoods but the STATED payoff structure, DP-regret falls from
0.007-0.082 to 0.0005-0.0033; elicited payoff contributions (additive action x factor-level amounts) are the error source,
while elicited beliefs are near-correct (true beliefs + elicited payoffs stays at 0.007-0.080).
Amended DIVA-ms: the payoff table is taken from the task specification (exactly as g and l are in the binary and
real-data tracks); the LLM elicits only uncertain quantities (current factor probabilities and check likelihoods).
LLM-elicited payoffs are reported as an ablation.
Confirmation: fresh ms test slice rows 500-999 per domain (never scored), all served models; rows 0-499 are reported too,
flagged as viewed. Hypothesis M1: DIVA-ms DP-regret below direct, Fac-EIG and always-stop in >= 80% of (model, domain) cells
on rows 500-999.
