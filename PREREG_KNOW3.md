# Amendment K3: uncertainty-aware decision value for knowledge-required tracks (before scoring it on any TEST data)
Date 2026-10-09 ~07:55 KST. Motivation (disclosed): interim test views showed DIVA under-acquiring on SEC volatility
targets (direct LLM better) and small margins over never-ask on some records. Diagnosis on DEV only: elicited evidence
strengths are compressed toward the base rate (SEC) and noisy (records), so a point specification undervalues checks whose
strength the LLM cannot state reliably.
Method (DIVA-U): the elicited log-odds spread of every check (binary: logit P(f|Y=1) - logit P(f|Y=0); binned: log-odds of
the extreme bins) is treated as uncertain with a Gaussian of scale sigma; the acquisition value is the expectation of VOI
over this uncertainty (3-point Gauss-Hermite), minus cost; stop if <= 0. Everything else as PREREG_KNOW (conditional form,
lambda = 0.75, greedy answers). Controlled tracks (stated reliabilities): sigma = 0 (no specification uncertainty).
Selection on DEV only (scripts/dev_uncertain_tab.py, scripts/dev_uncertain_voi.py; sigma grid 0, .25, .5, 1, 1.5, 2, 3):
sigma = 1.0. Dev ratio to never-ask: records 0.519 -> 0.360 (29 cells); SEC returns 0.416 -> 0.348 (30 cells);
SEC volatility 0.636 -> 0.178 (33 cells; direct LLM ~0.20).
Confirmation on TEST: all tab test rows (DIVA-U never scored on test) and SEC rows >= 1000 (primary confirmation slice),
all served models. Hypotheses: K3a DIVA-U regret lower than frozen DIVA in >= 2/3 of cells; K3b DIVA-U lower than never-ask
in >= 80% of cells; K3c on SEC volatility, DIVA-U not worse than direct in >= 1/2 of cells.
If K3a fails, frozen DIVA remains primary; both are reported either way.
