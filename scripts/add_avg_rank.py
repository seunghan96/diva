"""Add an average (mean regret over environments) column to Table tab:know (paper/tables/main_know.tex).

Ranks are computed within each block (DP optimum / myopic optimum) over the method rows only
(Generic reference rows and the Wins row are not ranked); ties share the average rank. Idempotent.
"""
import re, sys
import numpy as np
from scipy.stats import rankdata

P = sys.argv[1] if len(sys.argv) > 1 else "./paper/tables/main_know.tex"
s = open(P).read()
if "{Avg.}" in s:
    sys.exit(0)
if "% merged tab:" in s:  # merged knowledge-required | controlled table: no Avg. columns (user rule 2026-10-11)
    sys.exit(0)

num = lambda c: float(re.sub(r"\\(textbf|underline)\{([^}]*)\}", r"\2", c).strip())
lines = s.split("\n")
blocks, cur = [], []
for i, ln in enumerate(lines):
    if "Myopic optimum" in ln:
        blocks.append(cur); cur = []
    body = re.sub(r"^\\rowcolor\{oursrow\}\s*", "", ln)
    if body.rstrip().endswith("\\\\") and "&" in body and not body.startswith(("Method", " &", "Wins", "Generic", "\\multicolumn")) \
            and "Myopic optimum" not in body:
        cells = [c.strip() for c in body.rstrip()[:-2].split("&")]
        try:
            vals = [num(c) for c in cells[1:]]
        except ValueError:
            continue
        cur.append((i, vals))
blocks.append(cur)

avg = {}
for blk in blocks:
    if not blk:
        continue
    M = np.array([v for _, v in blk])
    a = M.mean(1)  # arithmetic mean of the regrets over the environments (same unit: regret x100 / l)
    order = sorted(set(np.round(a, 2)))
    for (i, _), x in zip(blk, a):
        t = f"{x:.2f}"
        if round(x, 2) == order[0]:
            t = f"\\textbf{{{t}}}"
        elif len(order) > 1 and round(x, 2) == order[1]:
            t = f"\\underline{{{t}}}"
        avg[i] = t

out = []
for i, ln in enumerate(lines):
    if ln.startswith("\\begin{tabular}{l|cccccccc}"):
        ln = ln.replace("{l|cccccccc}", "{l|cccccccc|c}")
    elif ln.startswith("Method &"):
        ln = ln.rstrip()[:-2].rstrip() + " & \\\\"
    elif ln.startswith(" & \\multicolumn{2}{c}{Credit}"):
        ln = ln.replace("\\multicolumn{4}{c}{Financial", "\\multicolumn{4}{c|}{Financial")
        ln = ln.rstrip()[:-2].rstrip() + " & \\multirow{2}{*}[-0.6ex]{Avg.} \\\\"
    elif "\\multicolumn{9}{l}{\\textit{Myopic optimum}}" in ln:
        ln = ln.replace("\\multicolumn{9}", "\\multicolumn{10}")
    elif i in avg:
        ln = ln.rstrip()[:-2].rstrip() + f" & {avg[i]} \\\\"
    elif ln.rstrip().endswith("\\\\") and "&" in ln and (ln.startswith("Generic") or "Wins of" in ln):
        ln = ln.rstrip()[:-2].rstrip() + " & -- \\\\"
    out.append(ln)
open(P, "w").write("\n".join(out))
print("added avg rank to", len(avg), "rows")
