"""Tables for the reviewer-gap experiments of F4 (2026-10-11):
  paper/tables/realized_big.tex  higher-powered realized utility on Taiwan / CDC (results/realized_big.json)
  paper/tables/para_bl.tex       prompt-paraphrase robustness of the baselines vs DIVA (results/para_bl.json)
Cells are gray TBD until every LLM of the fixed 20-LLM set is complete."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from complete import KNOW_LLMS, PH

OUT = f"{IS}/paper/tables"
COT = r"\begin{tabular}[c]{@{}l@{}}CoT\\\citep{wei2022chain}\end{tabular}"
PAL = r"\begin{tabular}[c]{@{}l@{}}PAL\\\citep{gao2023pal}\end{tabular}"
SH = r"\rowcolor{oursrow} "


def realized():
    p = f"{IS}/results/realized_big.json"
    R = json.load(open(p)) if os.path.exists(p) else {"cells": {}, "ref": {}}
    dss = [("taiwan", "Taiwan"), ("cdc", "CDC")]
    rows = [("direct", "Direct", ""), ("never", "Never", ""), ("diva", r"\diva{}", SH), ("diva_u", r"\diva{}-U", SH)]
    cik = {"direct": "ci_direct", "diva": "ci_diva", "diva_u": "ci_diva_u", "ref": "ci_ref"}
    n = {ds: R["ref"].get(ds, {}).get("n") for ds, _ in dss}
    body = []
    for key, name, sh in rows:
        cells = []
        for ds, _ in dss:
            C = [R["cells"].get(f"{t}:{ds}", {}) for t in KNOW_LLMS]
            need = "direct" if key == "direct" else "diva"
            if not all(need in c for c in C):
                cells += [PH, PH]; continue
            if key == "ref":
                u = R["ref"][ds]["ref_dp"]
            else:
                u = sum(c[key] for c in C) / len(C)
            cells.append(f"{100 * u:.2f}")
            if key == "never":
                cells.append("--")
            else:
                b = sum(c[cik[key]][1] > 0 for c in C); w = sum(c[cik[key]][2] < 0 for c in C)
                cells.append(f"{b} / {w}")
        body.append(sh + name + " & " + " & ".join(cells) + r" \\")
    # LLM-free reference rows: the reference model deciding at once, and its DP policy (significance vs deciding at once)
    body.append(r"\midrule")
    for key, name in (("ref_stop", "Reference: Never"), ("ref_dp", "Reference: DP")):
        cells = []
        for ds, _ in dss:
            rr = R["ref"].get(ds, {})
            if rr.get(key) is None:
                cells += [PH, PH]; continue
            cells.append(f"{100 * rr[key]:.2f}")
            ci = rr.get("ci_ref_vs_ref_stop")
            cells.append("--" if key == "ref_stop" else ("Better" if ci[1] > 0 else ("Worse" if ci[2] < 0 else "n.s.")))
        body.append(name + " & " + " & ".join(cells) + r" \\")
    nn = " and ".join(f"{n[ds]:,} {nm}" for ds, nm in dss if n[ds]) or "5,000 Taiwan and 5,000 CDC"
    kd = {ds: R["ref"].get(ds, {}).get("n_direct_set") for ds, _ in dss}
    dnote = ""
    if all(kd[ds] for ds, _ in dss) and any(kd[ds] < (n[ds] or 0) for ds, _ in dss):
        dnote = (" Direct, which calls the LLM at every step, is replayed on the first " + " and ".join(f"{kd[ds]:,}" for ds, _ in dss)
                 + " records and compared with never acquiring on the same records.")
    head = (r"\multirow{2}{*}[-0.6ex]{Method} & \multicolumn{2}{c}{Taiwan} & \multicolumn{2}{c}{CDC} \\" "\n"
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" "\n"
            r" & Utility & B / W & Utility & B / W \\")
    tex = ("\\begin{table}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{l|cccc}\n\\toprule\n" + head +
           "\n\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n"
           "\\caption{\\textbf{Realized utility} on " + nn + " records ($\\times$100, net of costs) over "
           + str(len(KNOW_LLMS)) + " LLMs. B / W: LLMs significantly better / worse than never acquiring. Reference: LLM-free reference-model belief." + dnote + "}\n"
           "\\label{tab:realized_big}\n\\end{table}\n")
    open(f"{OUT}/realized_big.tex", "w").write(tex)


def para():
    p = f"{IS}/results/para_bl.json"
    R = json.load(open(p)) if os.path.exists(p) else {"summary": {}}
    rows = [("direct", "Direct", ""), ("cot_need", COT, ""), ("voi_prompt", "VoI-P", ""), ("pal", PAL, ""), ("diva", r"\diva{}", SH)]
    body = []
    for key, name, sh in rows:
        s = R["summary"].get(key) or {}
        cells = [PH if s.get(k) is None or (k == "spread" and s.get("spread_over") != 3) else f"{s[k]:.2f}" for k in ("orig", "p1", "p2", "spread")]  # spread only over all three prompts
        body.append(sh + name + " & " + " & ".join(cells) + r" \\")
    head = r"Method & Original & Para.~1 & Para.~2 & Spread \\"
    tex = ("\\begin{table}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{l|cccc}\n\\toprule\n" + head +
           "\n\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n"
           "\\caption{\\textbf{Prompt paraphrases of the baselines.} DP regret on the controlled environments, averaged over the four domains and "
           + str(len(KNOW_LLMS)) + " LLMs. Spread: Range over the three prompts per LLM, averaged over LLMs.}\n"
           "\\label{tab:para_bl}\n\\end{table}\n")
    open(f"{OUT}/para_bl.tex", "w").write(tex)


if __name__ == "__main__":
    realized(); para(); print("ok")
