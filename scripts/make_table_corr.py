"""paper/tables/corr.tex from results/corr.json (scripts/agg_corr.py): correlated-evidence controlled environments.
Cells average over the 20 LLMs and are gray TBD until every LLM has all 300 scenarios of the cell."""
import os, sys, json, math
sys.path.insert(0, os.path.dirname(__file__))
from complete import KNOW_LLMS, PH, TXT, MCITE, group_rows, mname

IS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOMS = ["finance", "credit", "clinical", "legal"]
DN = {"finance": "Finance", "credit": "Credit", "clinical": "Clinical", "legal": "Legal"}
VARS = ["dup_known", "cc_known", "dup_cand"]
N = 300
METHS = ["direct", "cot_need", "sc_cot", "react", "finrisk_ask", "voi_prompt", "voi_paper", "bed_llm", "pal", "never", "always_top",
         "diva", "diva_u", "diva_dp"]
EXTERNAL = ["direct", "cot_need", "sc_cot", "react", "finrisk_ask", "voi_prompt", "voi_paper", "bed_llm", "pal"]
OURS = {"diva", "diva_u", "diva_dp"}
NCOL = 10


def label(m):
    n = mname("cot_need" if m == "cot_need" else m)
    if n in MCITE:
        return "\\begin{tabular}[c]{@{}l@{}}" + n + "\\\\\\citep{" + MCITE[n] + "}\\end{tabular}", n
    return n, n


def cell_vals(R, m):
    """dict column -> mean over the 20 LLMs, or None when any LLM is incomplete"""
    out = {}
    ok_all = True
    per = {}
    for d in DOMS:
        vs = []
        for t in KNOW_LLMS:
            s = R.get(t, {}).get(d, {}).get(m)
            if not s or s.get("n", 0) < N:
                vs = None; break
            vs.append(s)
        per[d] = vs
        if vs is None:
            ok_all = False
        out[d] = None if vs is None else 100 * sum(s["dp_regret"] for s in vs) / len(vs)
    out["avg"] = sum(out[d] for d in DOMS) / 4 if ok_all else None
    for v in VARS:
        out[v] = (100 * sum(s["dp_" + v] for d in DOMS for s in per[d]) / (4 * len(KNOW_LLMS))) if ok_all else None
    out["buy"] = (100 * sum(s["buy_corr"] for d in DOMS for s in per[d]) / (4 * len(KNOW_LLMS))) if ok_all else None
    return out


def ref_vals(R, k):
    rr = R["_ref"]
    out = {d: 100 * rr[d][k]["dp_regret"] for d in DOMS}
    out["avg"] = sum(out[d] for d in DOMS) / 4
    for v in VARS:
        out[v] = 100 * sum(rr[d][k]["dp_" + v] for d in DOMS) / 4
    out["buy"] = 100 * sum(rr[d][k]["buy_corr"] for d in DOMS) / 4
    return out


def wins(R, doms=None):
    """LLMs where DIVA's regret (averaged over `doms`, default all four domains) is below that of every external baseline"""
    doms = doms or DOMS
    w = 0
    for t in KNOW_LLMS:
        def avg(m):
            ss = [R.get(t, {}).get(d, {}).get(m) for d in doms]
            if any(not s or s.get("n", 0) < N for s in ss):
                return None
            return sum(s["dp_regret"] for s in ss) / len(doms)
        dv = avg("diva"); bs = [avg(m) for m in EXTERNAL]
        if dv is None or any(b is None for b in bs):
            return None
        w += all(dv < b - 1e-12 for b in bs)
    return w


def fmt(x):
    return PH if x is None else f"{abs(x):.2f}" if abs(x) < 0.005 else f"{x:.2f}"


def fmtp(x):
    return PH if x is None else f"{x:.0f}"


def main():
    R = json.load(open(f"{IS}/results/corr.json"))
    vals = {m: cell_vals(R, m) for m in METHS}
    cols = DOMS + ["avg"] + VARS
    best = {}
    for c in cols:
        xs = sorted({round(vals[m][c], 2) for m in METHS if vals[m][c] is not None})
        best[c] = xs[:2] if all(vals[m][c] is not None for m in METHS) else []

    def num(m, c):
        x = vals[m][c]
        s = fmt(x)
        if x is not None and best[c]:
            if round(x, 2) == best[c][0]:
                s = "\\textbf{" + s + "}"
            elif len(best[c]) > 1 and round(x, 2) == best[c][1]:
                s = "\\underline{" + s + "}"
        return s

    names, rows, labs, cl = [], [], [], []
    for m in METHS:
        lab, n = label(m)
        names.append(n); labs.append((m, lab)); cl.append([num(m, c) for c in cols] + [fmtp(vals[m]["buy"])])
    from make_tables import merge_diva_rows
    cl = [list(r) for r in zip(*merge_diva_rows(list(names), [list(c) for c in zip(*cl)], range(len(cl[0]))))]
    for (m, lab), cells in zip(labs, cl):
        if m in OURS:
            r = "\\rowcolor{oursrow} " + lab + " & " + " & ".join(cells) + " \\\\"
        else:
            r = lab + " & " + " & ".join(cells) + " \\\\"
        rows.append(r)
    body = group_rows(names, rows, NCOL)
    body.append("\\midrule")
    body.append("\\multicolumn{" + str(NCOL) + "}{l}{\\textit{Exact models (no LLM)}} \\\\")
    for k, nm in [("factored_myopic", "Factored, myopic"), ("factored_dp", "Factored, DP"), ("joint_dp", "Joint, DP")]:
        v = ref_vals(R, k)
        body.append(nm + " & " + " & ".join(fmt(v[c]) for c in cols) + " & " + fmtp(v["buy"]) + " \\\\")
    w = wins(R)
    body.append("\\midrule")
    wd = [wins(R, [d]) for d in DOMS] + [w]
    body.append("Wins of \\diva{} & " + " & ".join(PH if x is None else f"{x}/{len(KNOW_LLMS)}" for x in wd) + " & & & & \\\\")
    head = ("\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{l|ccccc|ccc|c}\n\\toprule\n"
            "\\multirow{2}{*}[-2.05ex]{\\begin{tabular}[c]{@{}l@{}}Method\\\\(Avg.\\ across " + str(len(KNOW_LLMS)) + " LLMs)\\end{tabular}} & "
            "\\multicolumn{5}{c|}{Regret by domain} & \\multicolumn{3}{c|}{Regret by variant} & "
            "\\multirow{2}{*}[-2.05ex]{\\begin{tabular}[c]{@{}c@{}}Correlated\\\\buys (\\%)\\end{tabular}} \\\\\n"
            "\\cmidrule(r){2-6}\\cmidrule(lr){7-9}\n"
            " & Finance & Credit & Clinical & Legal & Avg. & \\begin{tabular}[c]{@{}c@{}}Same\\\\records\\end{tabular} & "
            "\\begin{tabular}[c]{@{}c@{}}Shared\\\\source\\end{tabular} & \\begin{tabular}[c]{@{}c@{}}Both\\\\candidates\\end{tabular} & \\\\\n"
            "\\midrule\n")
    cap = ("\\caption{\\textbf{Correlated evidence in controlled environments.} DP regret against the true joint model, over "
           "20 LLMs and 300 scenarios per domain. Correlated buys: Share of first decisions buying a check whose partner "
           "was run.}\n\\label{tab:corr}\n")
    tex = head + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n" + cap + "\\end{table*}\n"
    open(f"{IS}/paper/tables/corr.tex", "w").write(tex)
    # text macros (red [TBD] until every cell they use is complete)
    ext = [vals[m]["avg"] for m in EXTERNAL]
    allc = all(x is not None for x in ext + [vals["diva"]["avg"], vals["never"]["avg"], vals["diva_dp"]["avg"]])
    f2 = lambda x: TXT if x is None or not allc else f"{x:.2f}"
    f0 = lambda x: TXT if x is None or not allc else f"{x:.0f}"
    bm = min(EXTERNAL, key=lambda m: vals[m]["avg"]) if allc else None
    fm, jd = ref_vals(R, "factored_myopic"), ref_vals(R, "joint_dp")
    buys = [vals[m]["buy"] for m in EXTERNAL]
    mac = {"CorrDiva": f2(vals["diva"]["avg"]), "CorrDivaDP": f2(vals["diva_dp"]["avg"]), "CorrNever": f2(vals["never"]["avg"]),
           "CorrBestBl": f2(vals[bm]["avg"] if bm else None), "CorrBestBlName": (TXT if bm is None else label(bm)[1]),
           "CorrRatio": (TXT if not allc else f"{vals[bm]['avg'] / vals['diva']['avg']:.1f}"),
           "CorrFacMyo": f"{fm['avg']:.2f}", "CorrBuyDiva": f0(vals["diva"]["buy"]), "CorrBuyDivaDP": f0(vals["diva_dp"]["buy"]),
           "CorrBuyJoint": f"{jd['buy']:.0f}", "CorrBuyFacMyo": f"{fm['buy']:.0f}",
           "CorrBuyBlMax": (TXT if not allc else f"{max(buys):.0f}"),
           "CorrBuyBlMaxName": (TXT if not allc else label(EXTERNAL[buys.index(max(buys))])[1]),
           "CorrWins": (TXT if w is None else str(w))}
    open(f"{IS}/paper/tables/corr_num.tex", "w").write("% generated by scripts/make_table_corr.py\n" +
        "".join("\\providecommand{\\" + k + "}{}\\renewcommand{\\" + k + "}{" + v + "}\n" for k, v in mac.items()))
    print("wrote paper/tables/corr.tex; TBD cells:", tex.count(PH))


if __name__ == "__main__":
    main()
