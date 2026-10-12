# Amendment K2 (SEC binned checks): joint outcome elicitation — 2026-10-09 ~03:30 KST (before scoring any joint TEST output)
Disclosure: interim SEC TEST aggregates under PREREG_KNOW (rows 0-999 for Mistral-7B, Qwen2.5-7B, R1-Qwen-7B; partial for
gpt-oss/Qwen3-8B/Qwen2.5-14B) were viewed. DIVA beat always-stop and Fac-EIG everywhere and direct on return targets, but
lost to direct on volatility targets. Diagnosis on SEC DEV only: single-question conditional risks per bin collapse to the
base rate (e.g. 0.52/0.52/0.52 for pre-announcement volatility, true 0.76/0.64/0.44), so DIVA sees no value and stops.
Redesign chosen on DEV (scripts/select_real_joint.py; 26 model x target cells): JOINT elicitation that asks for the adverse
share of all outcome bins of a check in one answer (contrastive), lambda = 1.0. Dev mean DP-regret ratio to always-stop:
joint 0.456 vs single 0.554 (single lambda 0.75: 0.584); direct 0.432.
Confirmation: SEC test rows >= 1000 per target (never scored by any DIVA variant), all served models; rows 0-999 also
reported with the viewing flag. Hypothesis K2a: joint-DIVA DP-regret below always-stop and Fac-EIG in >= 80% of cells and
below single-question DIVA in >= 2/3 of cells. No claim of superiority over direct is pre-registered for volatility targets.
