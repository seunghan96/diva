"""Short captions (user rule: one-column <= 5 lines, two-column table <= 3 lines). Details removed from captions are kept in
the body (Section 4.1 'Table conventions') or in the appendix text next to each table. Applied to paper/tables/*.tex after
every table generator run (rebuild_all)."""
import glob, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from complete import REASON_KNOW  # F4: reasoning-LLM set size in captions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
IS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = {
    "tab:know1000": r"\textbf{Knowledge-required environments on the first 1,000 test scenarios}, DP optimum, with the LLM baselines run on them, averaged over the 20 LLMs, to check that the first 300 scenarios are representative.",
    "tab:know_myo": r"\textbf{Myopic regret on knowledge-required environments.} Regret of the first decision against the myopic optimum, on the same scenarios and LLMs as Table~\ref{tab:know}. Avg.\ is the mean over the eight environments.",
    "tab:know": r"\textbf{DP regret} of the first decision, averaged over 20 LLMs. \diva{}-U equals \diva{} in the controlled environments, where reliabilities are stated. Wins: \diva{}-U beats every external baseline.",
    "tab:know_detail": r"\textbf{Stopping behavior on knowledge-required environments.} Regret against the DP / myopic optimum and stopping rates at the first decision (all test scenarios, 20 LLMs), with the tuned \diva{}-$\beta$ and \diva{}-$\kappa$.",
    "tab:confirm": r"\textbf{Held-out SEC filings.} DP / myopic regret beyond the first 1,000 scenarios per target, over 20 LLMs.",
    "tab:episodes_know_prereg": r"\textbf{Pre-registered episodes} on test scenarios 300--999 with the final action from the reference posterior, averaged over the same 20 LLMs. A $\rightarrow$ \diva{} takes the first decision of A and then follows \diva{}.",
    "tab:episodes_know_exact": r"\textbf{Full episodes of every exact policy} on test scenarios 300--999, with the final action from the reference posterior, averaged over the same 20 LLMs. Generic DP gives every check the same strength.",
    "tab:episodes_know_own": r"\textbf{Full episodes with the policy's own final action} on scenarios 300--999, averaged over the 20 LLMs.",
    "tab:episodes_know300": r"\textbf{Exploratory full episodes} on the first 300 test scenarios, with the final action from the reference posterior / the policy's own belief, averaged over the 20 LLMs. Acq.: Mean number of acquired checks.",
    "tab:ms": r"\textbf{Multi-state environments.} Regret on held-out scenarios, normalized by half the payoff range.",
    "tab:episodes": r"\textbf{Episode regret} of full acquire-then-decide episodes, averaged over environments and 20 LLMs.",
    "tab:episodes_full": r"\textbf{Episode regret per environment}, averaged over the same 20 LLMs, on test scenarios 0--149 with the reference-posterior final action (knowledge-required) and on identical pre-sampled episodes (controlled).",
    "tab:controls": r"\textbf{Additional controls.} Left: \diva{} vs.\ \diva{} w/ EIG$^\ddagger$ (controlled). Right: Real records, all test scenarios.",
    "tab:altgold": r"\textbf{Naive Bayes reference model.} DP regret under the main reference model (Main) and naive Bayes (NB) on the first 1,000 test scenarios, over 20 LLMs.",
    "tab:ci_know": r"\textbf{Paired differences with 95\% CIs on knowledge-required environments}, averaged over the 20 LLMs. Negative values favor \diva{}, and a CI excluding zero marks a significant difference.",
    "tab:const": r"\textbf{Generic versus elicited specification.} Regret against the DP / myopic optimum of a generic and an elicited specification (\diva{}, \diva{}-U). The generic one gives every check the same strength, selected on development data as one value for all environments. First 300 test scenarios, as in Table~\ref{tab:know}.",
    "tab:cost": r"\textbf{Cost of the first acquisition decision} per scenario on the controlled finance environment.",
    "tab:decomp": r"\textbf{Error decomposition on controlled environments}, all test scenarios. Lik.: Likelihoods. MAE: Mean absolute error vs.\ true numbers.",
    "tab:extra_bl": r"\textbf{Additional baselines on controlled environments.} Regret against the DP optimum per LLM.",
    "tab:full_know": r"\textbf{Knowledge-required environments on all available test scenarios}, tabular records.",
    "tab:full_know_1": r"\textbf{Knowledge-required environments on all available test scenarios}, tabular records (continued).",
    "tab:full_know_sec": r"\textbf{Knowledge-required environments on all available test scenarios}, SEC filings.",
    "tab:full_know_sec_1": r"\textbf{Knowledge-required environments on all available test scenarios}, SEC filings (continued).",
    "tab:full_bin": r"\textbf{Controlled environments on all available test scenarios per domain.} $n$ per row.",
    "tab:full_bin_1": r"\textbf{Controlled environments on all available test scenarios per domain} (continued).",
    "tab:k4": r"\textbf{Sampled specifications.} DP regret of \diva{} with greedy and averaged specifications (20 LLMs).",
    "tab:large": r"\textbf{Larger LLMs on controlled environments.}",
    "tab:ms_diag": r"\textbf{Error sources in the multi-state environments} (diagnosed on the development split only).",
    "tab:para": r"\textbf{Paraphrased elicitation questions.} DP regret per LLM.",
    "tab:realized": r"\textbf{Realized utility on real records}, net of costs, per LLM and dataset (higher is better).",
    "tab:realized_1": r"\textbf{Realized utility on real records}, net of costs (continued).",
    "tab:voiask_know": r"\textbf{VoI-Ask \citep{dong2026value} on knowledge-required environments}, averaged over the 20 LLMs.",
    "tab:think_know": r"\textbf{Native reasoning on knowledge-required environments}, averaged over the " + str(len(REASON_KNOW)) + " reasoning LLMs.",
    "tab:ablation": r"\textbf{Ablation study.} DP regret on 496 controlled scenarios, averaged over the same 20 LLMs.",
    "tab:abl_eig": r"\textbf{Decision value versus EIG on the same specification.} DP regret on the full test sets, averaged over the same 20 LLMs. $^\ddagger$: Cost trade-off and threshold tuned on development scenarios with ground-truth decision values.",
}

# number of LLMs averaged per column of tab:know / tab:know_myo (all Table-3 methods run on s300)
try:
    import json as _json
    _R = _json.load(open(f"{IS}/results/know_main.json"))
    _M = ["direct", "cot_need", "bed_llm", "voi_prompt", "pal", "never", "always_top", "diva", "diva_u"]
    def _n(envs):
        ns = [sum(all(f"s300:{m}" in rec for m in _M) for rec in _R.get(e, {}).values()) for e in envs]
        return (str(min(ns)) if min(ns) == max(ns) else f"{min(ns)}--{max(ns)}")
    _tab, _sec = _n(["tab:taiwan", "tab:german", "tab:cdc", "tab:heart"]), _n(["sec:ret_1", "sec:ret_5", "sec:vol_5", "sec:vol_20"])
    _cnt = (f"{_tab} per environment" if _tab == _sec else f"{_tab} per tabular and {_sec} per SEC environment")
except Exception:
    _cnt = "see Appendix~\\ref{app:full}"
C = {k: v.replace("__LLMCOUNT__", _cnt) for k, v in C.items()}


def caption_span(s, start):
    k = start + len("\\caption{"); depth = 1
    while depth and k < len(s):
        depth += {"{": 1, "}": -1}.get(s[k], 0)
        if s[k] == "\\":
            k += 2; continue
        k += 1
    return start, k


for f in glob.glob(f"{IS}/paper/tables/*.tex"):
    if os.path.basename(f) == "diff.tex":  # user-owned Table 1, never touched
        continue
    s = open(f).read(); changed = False
    for m in list(re.finditer(r"\\label\{([^}]+)\}", s))[::-1]:
        lab = m.group(1)
        if lab not in C:
            continue
        j = s.rfind("\\caption{", 0, m.start())
        if j < 0:
            continue
        a, b = caption_span(s, j)
        cap = C[lab]
        if "{\\textbf{Qwen3}}" in s[s.rfind("\\begin{table", 0, j):j]:  # family-grouped LLM rows: name the families
            from complete import FAMILY_NOTE
            cap = cap + " " + FAMILY_NOTE
        new = "\\caption{" + cap + "}"
        if s[a:b] != new:
            s = s[:a] + new + s[b:]; changed = True
    if changed:
        open(f, "w").write(s)
