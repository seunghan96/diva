"""Analysis Table tab:generic: what the LLM's knowledge contributes. First-decision DP regret
(x100) of the Table-3 cells (first 300 test scenarios, mean over the same 20 LLMs), averaged over environment groups:
records (Taiwan, German, CDC, Heart), SEC filings (4 targets), controlled (4 domains). Rows: the best LLM baseline (per
environment the lowest mean regret among the external LLM baselines of Table tab:know, then averaged over the group),
never acquiring, the LLM-free generic specification (one strength for every check, scripts/generic_main.py), and DIVA /
DIVA-U. Reads results/know_table_means.json written by make_tables.py know2."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from complete import fmt_best, PH, DOM_ORDER  # noqa: E402
from make_tables import merge_diva_rows, OURS_ROW  # noqa: E402

IS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = json.load(open(f"{IS}/results/know_table_means.json"))
EXTERNAL = ("Prompting", "Decision-aware prompting", "Information gain", "Program-aided")
GROUPS = [("know", ["taiwan", "german", "cdc", "heart"]), ("know", ["ret_1", "ret_5", "vol_5", "vol_20"]), ("ctrl", list(DOM_ORDER))]
ext = [m for m, g in M["group"].items() if g in EXTERNAL]


def gmean(xs):
    return float(np.mean(xs)) if xs and all(x is not None for x in xs) else None


def cell(m, part, envs):
    return gmean([M[part].get(m, {}).get(e) for e in envs])


def best_baseline(part, envs):
    per = []
    for e in envs:
        v = [M[part][m].get(e) for m in ext]
        per.append(None if any(x is None for x in v) else min(v))
    return gmean(per)
ROWS = [("best", None, None), ("never", None, None),
        ("const_diva", "\\diva{}", "One shared"), ("diva", None, "From LLM"),
        ("const_diva_u", "\\diva{}-U", "One shared"), ("diva_u", None, "From LLM")]
LAB = {"best": "Best LLM baseline", "never": "Never"}
cols = []
for part, envs in GROUPS:
    cols.append([best_baseline(part, envs) if k == "best" else cell(k, part, envs) for k, _, _ in ROWS])
colf = [fmt_best(c, scale=100) for c in cols]
lines = []
for i, (k, meth, src) in enumerate(ROWS):
    vals = " & ".join(c[i] for c in colf)
    if src is None:
        lines.append("\\multicolumn{2}{@{}l|}{" + LAB[k] + "} & " + vals + " \\\\")
    else:
        if meth:
            lines.append("\\midrule")
        lines.append(("\\multirow{2}{*}{" + meth + "}" if meth else "") + " & " + src + " & " + vals + " \\\\")
tex = ("\\begin{table}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}ll|cc|c@{}}\n\\toprule\n"
       "\\multirow{2}{*}[-0.7ex]{Method} & \\multirow{2}{*}[-0.7ex]{\\begin{tabular}[c]{@{}l@{}}Check\\\\strengths\\end{tabular}} & \\multicolumn{2}{c|}{Knowledge-required} & \\multirow{2}{*}[-0.7ex]{Controlled} \\\\\n"
       "\\cmidrule(lr){3-4}\n & & Records & SEC & \\\\\n\\midrule\n"
       + "\n".join(lines)
       + "\n\\bottomrule\n\\end{tabular}\n"
       "\\caption{\\textbf{What the LLM's knowledge contributes.} DP regret of the first decision, averaged over 20 LLMs. "
       "Best LLM baseline: Best of Table~\\ref{tab:know} per environment.}\n"
       "\\label{tab:generic}\n\\end{table}\n")
open(f"{IS}/paper/tables/generic_main.tex", "w").write(tex)
print("ok tab:generic", "TBD" if PH in tex else "complete", [[None if v is None else round(100 * v, 2) for v in c] for c in cols])
