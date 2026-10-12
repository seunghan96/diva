"""Tables of the F8 analyses (results/f8_summary.json from an_f8_agg.py, results/f8_divaus_{dev,test}.json from
an_f8_divaus.py): paper/tables/f8_stop.tex (cost- and stop-aware baselines), f8_mis.tex (misspecified payoffs and costs),
f8_cost.tex (cost regimes on the knowledge-required environments), f8_us.tex (DIVA-U with shrinking uncertainty, episodes).
Cells are DP regret (x100) averaged over the same 20 LLMs; TBD unless every LLM is complete."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE); sys.path.insert(0, HERE)
from complete import fmt_best, PH, ENVN, KNOW_LLMS
from an_common import TAB, SEC, CDOMS

OUT = f"{IS}/paper/tables"
S = json.load(open(f"{IS}/results/f8_summary.json"))
KN, CT = TAB + SEC, list(CDOMS)
ROW = "\\rowcolor{oursrow} "
CITE = {"Direct": "Direct", "Ask-P": "\\begin{tabular}[c]{@{}l@{}}Ask-P\\\\\\citep{zhong2026finriskatlas}\\end{tabular}"}


def env_header(lead, nk=8, nc=4):
    return ("\\multirow{3}{*}[-1.4ex]{" + lead + "} & \\multicolumn{" + str(nk) + "}{c|}{Knowledge-required environments} & \\multicolumn{" + str(nc) + "}{c}{Controlled environments} \\\\\n"
            "\\cmidrule(lr){2-" + str(1 + nk) + "}\\cmidrule(lr){" + str(2 + nk) + "-" + str(1 + nk + nc) + "}\n"
            " & \\multicolumn{2}{c}{Credit} & \\multicolumn{2}{c}{Clinical} & \\multicolumn{4}{c|}{SEC filings (2025--2026)} & "
            + " & ".join("\\multirow{2}{*}[-0.6ex]{" + ENVN[d] + "}" for d in CT) + " \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-9}\n"
            " & " + " & ".join(ENVN[e] for e in KN[:-1]) + " & \\multicolumn{1}{c|}{" + ENVN[KN[-1]] + "} & & & & \\\\\n\\midrule\n")


def cells(vals_by_row, envs):
    """vals_by_row: list of {env: value or None}; returns per-row cell strings, best bold / second underlined per column"""
    cols = [fmt_best([r.get(e) for r in vals_by_row], scale=100) for e in envs]
    return [[c[i] for c in cols] for i in range(len(vals_by_row))]


def merge_pairs(rows, pairs, cidx):
    """print a DIVA / DIVA-U value once, centered over both rows, where they are identical (controlled columns)"""
    for a, b in pairs:
        for c in cidx:
            if rows[a][c] == rows[b][c]:
                rows[b][c] = "\\multirow{-2}{*}{" + rows[a][c] + "}"; rows[a][c] = ""
    return rows


def wrap(spec, head, body, cap, label, star=True):
    env = "table*" if star else "table"
    return ("\\begin{" + env + "}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{" + spec + "}\n\\toprule\n" + head
            + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\caption{" + cap + "}\n\\label{" + label + "}\n\\end{" + env + "}\n")


def t_stop():
    E = KN + CT; G = S["G"]; C = S["C"]
    names = ["Direct", "Direct + tuned stop", CITE["Ask-P"], "Ask-P + tuned stop", "Cost-aware EIG", "\\diva{}", "\\diva{}-U"]
    vals = [{e: G[e]["direct"] for e in E}, {e: G[e]["thr_direct"] for e in E}, {e: G[e]["ask_p"] for e in E}, {e: G[e]["thr_ask"] for e in E},
            {e: C[e]["eigc"] for e in E}, {e: C[e]["diva"] for e in E}, {e: C[e]["diva_u"] for e in E}]
    rows = merge_pairs(cells(vals, E), [(5, 6)], range(8, 12))
    body = []
    for i, (n, r) in enumerate(zip(names, rows)):
        if i in (4, 5):
            body.append("\\midrule")
        body.append((ROW if i >= 5 else "") + n + " & " + " & ".join(r) + " \\\\")
    cap = ("\\textbf{Cost- and stop-aware baselines.} DP regret on the first 300 test scenarios, averaged over 20 LLMs. "
           "Tuned stop: Stop threshold of each LLM, cross-fitted over scenario groups. Cost-aware EIG: EIG minus $\\lambda c/l$ on the elicited specification, "
           "with one $\\lambda=" + f"{S['lam']:g}" + "$ selected on development data.")
    return wrap("l|cccccccc|cccc", env_header("Method"), body, cap, "tab:f8_stop")


def t_mis():
    E = KN + CT; M = S["E"]; C = S["C"]
    names = ["\\diva{}", "\\diva{}-U", "\\diva{}, $\\pm$25\\%", "\\diva{}-U, $\\pm$25\\%", "\\diva{}, $\\pm$50\\%", "\\diva{}-U, $\\pm$50\\%"]
    vals = [{e: C[e]["diva"] for e in E}, {e: C[e]["diva_u"] for e in E}, {e: M[e]["diva_0.25"] for e in E}, {e: M[e]["divau_0.25"] for e in E},
            {e: M[e]["diva_0.5"] for e in E}, {e: M[e]["divau_0.5"] for e in E}]
    rows = merge_pairs(cells(vals, E), [(0, 1), (2, 3), (4, 5)], range(8, 12))
    body = []
    for i, (n, r) in enumerate(zip(names, rows)):
        if i in (2, 4):
            body.append("\\midrule")
        body.append(ROW + n + " & " + " & ".join(r) + " \\\\")
    cap = ("\\textbf{Misspecified payoffs and costs.} DP regret when \\diva{} decides with $g$, $l$, and every cost multiplied by independent factors drawn uniformly "
           "from $[1-p,1+p]$ and is scored with the true values, averaged over 20 LLMs.")
    return wrap("l|cccccccc|cccc", env_header("Specification"), body, cap, "tab:f8_mis")


def t_cost():
    F = S["F"]; sc = [0.25, 0.5, 1.0, 2.0, 4.0]
    grp = lambda envs, k: (float(np.mean([F[e][k] for e in envs])) if all(F[e][k] is not None for e in envs) else None)
    names = ["Never", "\\diva{}", "\\diva{}-U"]; keys = ["never", "diva", "divau"]
    cols = []
    for envs in (TAB, SEC):
        for s in sc:
            cols.append(fmt_best([grp(envs, f"{k}_{s}") for k in keys], scale=100))
    body = [(ROW if i else "") + n + " & " + " & ".join(c[i] for c in cols) + " \\\\" for i, n in enumerate(names)]
    st = [grp(envs, f"dpstop_{s}") for envs in (TAB, SEC) for s in sc]
    body.append("\\midrule\nStopping is DP-optimal (\\%) & " + " & ".join(PH if x is None else f"{100 * x:.0f}" for x in st) + " \\\\")
    head = ("\\multirow{2}{*}[-0.6ex]{Method} & \\multicolumn{5}{c|}{Records (Taiwan, German, CDC, Heart)} & \\multicolumn{5}{c}{SEC filings (four targets)} \\\\\n"
            "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\n & " + " & ".join(f"$\\times${s:g}" for s in sc) + " & " + " & ".join(f"$\\times${s:g}" for s in sc) + " \\\\\n\\midrule\n")
    cap = ("\\textbf{Cost regimes on the knowledge-required environments.} DP regret when every check cost is scaled for the decision and for the "
           "reference DP optimum, averaged over each group and 20 LLMs ($\\times$1: Table~\\ref{tab:know}).")
    return wrap("l|ccccc|ccccc", head, body, cap, "tab:f8_cost")


def t_us():
    T = json.load(open(f"{IS}/results/f8_divaus_test.json")); Dv = json.load(open(f"{IS}/results/f8_divaus_dev.json"))
    sched = ["us_sqrt", "us_inv", "us_first"]
    m = lambda R, e, p, k: (float(np.mean([R[e][t][p][k] for t in KNOW_LLMS])) if e in R and all(t in R[e] and p in R[e][t] for t in KNOW_LLMS) else None)
    dev = {p: np.mean([m(Dv, e, p, "regret_ref") for e in KN]) for p in sched}
    sel = min(dev, key=dev.get)
    pols = ["never", "diva", "diva_u"] + sched
    lab = {"never": "Never", "diva": "\\diva{}", "diva_u": "\\diva{}-U ($\\sigma$ fixed)", "us_sqrt": "\\diva{}-U, $\\sigma/\\sqrt{1+n}$",
           "us_inv": "\\diva{}-U, $\\sigma/(1+n)$", "us_first": "\\diva{}-U, first step only"}
    per = {p: {e: m(T, e, p, "regret_ref") for e in KN} for p in pols}
    avg = {p: (float(np.mean(list(per[p].values()))) if all(v is not None for v in per[p].values()) else None) for p in pols}
    own = {p: (float(np.mean([m(T, e, p, "regret") for e in KN])) if all(m(T, e, p, "regret") is not None for e in KN) else None) for p in pols}
    acq = {p: (float(np.mean([m(T, e, p, "n_acq") for e in KN])) if all(m(T, e, p, "n_acq") is not None for e in KN) else None) for p in pols}
    cols = [fmt_best([per[p][e] for p in pols], scale=100) for e in KN] + [fmt_best([avg[p] for p in pols], scale=100), fmt_best([own[p] for p in pols], scale=100)]
    body = []
    for i, p in enumerate(pols):
        if i == 1:
            body.append("\\midrule")
        name = lab[p] + ("$^\\star$" if p == sel else "")
        body.append((ROW if i else "") + name + " & " + " & ".join(c[i] for c in cols) + " & " + (PH if acq[p] is None else f"{acq[p]:.2f}") + " \\\\")
    head = ("\\multirow{2}{*}[-0.6ex]{Policy} & \\multicolumn{2}{c}{Credit} & \\multicolumn{2}{c}{Clinical} & \\multicolumn{4}{c|}{SEC filings (2025--2026)} & "
            "\\multicolumn{2}{c|}{Avg.} & \\multirow{2}{*}[-0.6ex]{Acq.} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-11}\n"
            " & " + " & ".join(ENVN[e] for e in KN[:-1]) + " & \\multicolumn{1}{c|}{" + ENVN[KN[-1]] + "} & Ref. & \\multicolumn{1}{c|}{Own} & \\\\\n\\midrule\n")
    cap = ("\\textbf{\\diva{}-U with shrinking uncertainty.} Episode regret on test scenarios 0--149, averaged over 20 LLMs, with the reference-posterior "
           "(Ref.) or own (Own) final action. Acq.: Acquisitions per episode; $n$: Checks acquired so far; $^\\star$: Selected on development data.")
    return wrap("l|cccccccc|cc|c", head, body, cap, "tab:f8_us"), sel, dev


if __name__ == "__main__":
    open(f"{OUT}/f8_stop.tex", "w").write(t_stop())
    open(f"{OUT}/f8_mis.tex", "w").write(t_mis())
    open(f"{OUT}/f8_cost.tex", "w").write(t_cost())
    t, sel, dev = t_us(); open(f"{OUT}/f8_us.tex", "w").write(t)
    print("ok; selected schedule", sel, {k: round(100 * v, 2) for k, v in dev.items()},
          "TBD cells:", sum(open(f"{OUT}/{f}").read().count(PH) for f in ["f8_stop.tex", "f8_mis.tex", "f8_cost.tex", "f8_us.tex"]))
