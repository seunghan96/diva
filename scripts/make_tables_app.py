"""Appendix tables: ms_diag, decomp, qual -> paper/tables/."""
import json, os, re, glob, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_tables import (IS, OUT, NAMES, fmt_best, hcell, is_ours, OURS_CELL, reflist, mlabel)
from complete import group_rows
from complete import (mname, msort, lsort, ENV_ORDER, DOM_ORDER, ENVN as _C_ENVN, family_rows, FAMILY_NOTE, FAM_HEAD)
from complete import (PH, KNOW_LLMS, K4_LLMS, REASON_KNOW, CTRL_LLMS, CTRL_BL_LLMS, EP_CTRL_LLMS, K4SPEC_LLMS, PARA_LLMS,
                      know_ok, know_n, know_size, ctrl_ok)


def _phrow(label, ncells):
    """a row whose values are pending (intended LLM set or scenario slice incomplete)"""
    return label + " & " + " & ".join([PH] * ncells) + r" \\"

BS = "\\"


def table_ms_diag():
    from complete import MS_DEV_LLMS
    got = {}
    for f in sorted(glob.glob(f"{IS}/logs/diagdev_ms_*.log")):
        tag = f.split("diagdev_ms_")[1][:-4]
        vals = {"elicited": [], "true_util": [], "true_beliefs": [], "true_all": []}
        for line in open(f):
            for k in vals:
                m = re.search(r"\b" + k + r":([0-9.]+)/", line)
                if m:
                    vals[k].append(float(m.group(1)))
        if all(len(v) == 4 for v in vals.values()):
            got[tag] = [np.mean(vals[k]) for k in ["elicited", "true_util", "true_beliefs", "true_all"]]
    rows = family_rows(MS_DEV_LLMS, lambda t: [f"{100*x:.2f}" for x in got[t]] if t in got else [PH] * 4, 6)
    head = FAM_HEAD + r" & Elicited & \begin{tabular}[c]{@{}c@{}}Stated\\payoffs\end{tabular} & \begin{tabular}[c]{@{}c@{}}True\\beliefs\end{tabular} & \begin{tabular}[c]{@{}c@{}}True\\all\end{tabular} \\"
    tex = _fam_tab(rows, r"\textbf{Error sources in the multi-state environments (development split).} Regret ($\times$100) of \diva{} with elicited quantities, with the stated payoff table, with the true beliefs, and with both.",
                   "tab:ms_diag", 4, head, star=False)
    open(f"{OUT}/ms_diag.tex", "w").write(tex)
    return len(got)


def table_decomp():
    D = json.load(open(f"{IS}/results/analysis_decomp.json"))
    keys = ["elicited", "true_prior", "true_lik", "true_all"]
    B = json.load(open(f"{IS}/results/bin_main.json"))

    def cells(tag):
        if tag in D and ctrl_ok(B, tag) and all(k in D[tag] for k in keys):
            return [f"{100*D[tag][k]['dp_regret']:.2f}" for k in keys] + [f"{D[tag]['_err']['norm_regret']:.3f}"]
        return [PH] * 5  # missing or partial run: pending
    rows = family_rows(CTRL_LLMS, cells, 7)
    head = FAM_HEAD + r" & Elicited & \begin{tabular}[c]{@{}c@{}}True\\prior\end{tabular} & \begin{tabular}[c]{@{}c@{}}True\\lik.\end{tabular} & \begin{tabular}[c]{@{}c@{}}True\\all\end{tabular} & MAE \\"
    tex = _fam_tab(rows, r"\textbf{Error decomposition on controlled environments} (all test scenarios, pooled over the four domains). Regret ($\times$100) of \diva{} with parts of the specification replaced by the true values, and the mean absolute error (MAE) of the elicited numbers.",
                   "tab:decomp", 5, head, star=False)
    open(f"{OUT}/decomp.tex", "w").write(tex)
    return len(rows)


def table_qual(tag="q8"):
    Q = json.load(open(f"{IS}/results/qual_{tag}.json"))[0]
    txt = Q["text"].replace("%", r"\%").replace("$", r"\$")
    body = (r"\\" + "\n").join(l for l in txt.split("\n") if l.strip())
    d = Q["direct"]
    cap = (r"\textbf{A controlled finance scenario.} The direct policy of " + NAMES.get(tag, tag) + f" requests ({d}) {Q['direct_name']}, "
           f"the most informative check (EIG {Q['eig'][d]:.2f} bits), but under every outcome of every check the probability of underperformance "
           r"stays above the threshold $b^\dagger=" + f"{Q['g']}/({Q['g']}+{Q['l']})" + r"$, so no check has decision value. "
           r"\diva{} stops and passes (DP-optimal).")
    tex = (r"\begin{figure*}[t]" "\n" r"\centering" "\n" r"\fbox{\parbox{0.96\textwidth}{\footnotesize " + body + "}}\n"
           r"\caption{" + cap + "}\n" r"\label{fig:qual}" "\n" r"\end{figure*}" "\n")
    open(f"{OUT}/qual.tex", "w").write(tex)
    return Q["id"]




def table_controls():
    X = json.load(open(f"{IS}/results/extra_controls.json"))
    K = "dp_regret"
    B = json.load(open(f"{IS}/results/bin_main.json"))

    def cells(tag):
        v = X["controlled"].get(tag)
        if v and ctrl_ok(B, tag):  # all 3,000 test scenarios per domain
            d, lo, hi = v["diff_ci"]
            return [f"{100*v['fac_eig_cost'][K]:.2f}", f"{OURS_CELL}{100*v['diva'][K]:.2f}", f"[{100*lo:.2f}, {100*hi:.2f}]"]
        return [PH, OURS_CELL + PH, PH]
    lines = family_rows(CTRL_LLMS, cells, 5)
    t1 = (r"\begin{tabular}[t]{@{}ll|ccc}" "\n" r"\toprule" "\n" + FAM_HEAD + r" & \begin{tabular}[c]{@{}c@{}}\diva{} w/\\EIG$^\ddagger$\end{tabular} & \cellcolor{oursrow}\diva{} & \begin{tabular}[c]{@{}c@{}}95\% CI\\of diff.\end{tabular} \\" "\n" r"\midrule" "\n" + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular}")
    names = msort([("diva", None), ("knowledge_free", None), ("diva_u", None), ("knowledge_free_u", None), ("heuristic", None), ("fac_eig_cost", None),
                   ("always_stop", None), ("emp_nb", None)])
    names = [(m, mname(m)) for m, _ in names]
    dss = ["taiwan", "german", "cdc", "heart"]
    # a dataset column is shown only when all 20 LLMs have their specification on every test scenario
    okd = {d: know_ok(d) and all(f"{t}:{d}" in X["knowledge"] for t in KNOW_LLMS) for d in dss}
    colv = {d: [float(np.mean([X['knowledge'][f'{t}:{d}'][m][K] for t in KNOW_LLMS])) if okd[d] else None for m, _ in names] for d in dss}
    colf = {d: fmt_best(colv[d], scale=100) for d in dss}
    l2 = [(r"\rowcolor{oursrow} " if is_ours(n) else "") + n + " & " + " & ".join(colf[d][i] for d in dss) + r" \\" for i, (m, n) in enumerate(names)]
    l2 = group_rows([n for _, n in names], l2, 1 + len(dss))
    t2 = (r"\begin{tabular}[t]{l|" + "c" * len(dss) + "@{}}\n" r"\toprule" "\n" "Method & " + " & ".join(ENVN[d] for d in dss) + r" \\" "\n" r"\midrule" "\n" + "\n".join(l2) + "\n" r"\bottomrule" "\n" r"\end{tabular}")
    tex = (r"\begin{table*}[t]" "\n" r"\centering" "\n" r"\small" "\n" r"\setlength{\tabcolsep}{3pt}" "\n" + t1 + "\n" r"\hspace{12pt}" "\n" + t2 + "\n"
           r"\caption{\textbf{Additional controls.} Left: \diva{} against tuned EIG on controlled environments. Right: Knowledge-required records.}" "\n"
           r"\label{tab:controls}" "\n" r"\end{table*}" "\n")
    open(f"{OUT}/controls.tex", "w").write(tex)
    return len(lines)


def _tab(cols, rows, caption, label, star=False, size=r"\small", nlab=1, colsep="3pt", head=None):
    """generic table: \\small (user rule), canonical orders are applied by the callers; head overrides the header row(s);
    long method headers are set on two lines (make_tables.hcell)."""
    env = "table*" if star else "table"
    if head is None:
        head = " & ".join(cols) + r" \\"
    return ("\\begin{" + env + "}[t]\n\\centering\n" + size + "\n\\setlength{\\tabcolsep}{" + colsep + "}\n\\begin{tabular}{" + "l" * nlab + "c" * (len(cols) - nlab) + "}\n\\toprule\n"
            + head + "\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\caption{" + caption + "}\n\\label{" + label + "}\n\\end{" + env + "}\n")


ENVN = dict(_C_ENVN)


def _chunks(rows, n):
    import math as _m
    if not rows:
        return [[]]
    k = _m.ceil(len(rows) / n); size = _m.ceil(len(rows) / k)
    return [rows[i:i + size] for i in range(0, len(rows), size)]


def table_full():
    K = "dp_regret"
    KM = json.load(open(f"{IS}/results/know_main.json"))
    out = []
    ms = [(m, mname(m)) for m, _ in msort([("direct", 0), ("never", 0), ("diva", 0), ("diva_u", 0)])]
    names = [n for _, n in ms]
    head = "Env. & " + FAM_HEAD + " & $n$ & " + " & ".join(_h(names)) + r" \\"
    for grp, envs, lab in [("tabular records (credit and clinical)", ["tab:taiwan", "tab:german", "tab:cdc", "tab:heart"], "tab:full_know"),
                           ("SEC filings", ["sec:ret_1", "sec:ret_5", "sec:vol_5", "sec:vol_20"], "tab:full_know_sec")]:
        blocks = []
        for e in envs:
            def cells(tag, e=e):
                rec = KM.get(e, {}).get(tag, {})
                if not ("full:diva" in rec and know_n(e, tag) >= know_size(e)):  # missing or partial runs are pending
                    return [PH] + _shade(names, [PH] * len(ms))
                dps = [rec[f"full:{m}"][K] if f"full:{m}" in rec else None for m, _ in ms]
                mys = [rec[f"full:{m}"]["norm_regret"] if f"full:{m}" in rec else None for m, _ in ms]
                return [str(rec["n_full"])] + _shade(names, [("--" if d is None else c) for d, c in zip(dps, _pair(dps, mys))])
            blocks.append(family_rows(KNOW_LLMS, cells, 4 + len(ms), lead=["\\multirow{" + str(len(KNOW_LLMS)) + "}{*}{" + ENVN[e.split(':')[1]] + "}"]))
        for k, ch in enumerate(_env_blocks(blocks, 2)):
            out.append(_fam_tab(ch, r"\textbf{Knowledge-required environments on all available test scenarios ($n$ per row), " + grp + (" (continued)" if k else "") + r"}. Regret ($\times$100) against the DP / myopic optimum, on identical scenarios within each row.",
                                lab + (f"_{k}" if k else ""), 1 + len(ms), head, lead=1))
    B = json.load(open(f"{IS}/results/bin_main.json"))
    blocks = []
    for d in DOM_ORDER:
        def cells(tag, d=d):
            r = B.get(tag, {}).get(d, {})
            if not ("diva" in r and r["diva"]["n"] >= 3000):  # missing or partial runs are pending
                return [PH, PH, OURS_CELL + PH, PH, OURS_CELL + PH, OURS_CELL + PH]
            dp = fmt_best([r["direct"][K] if "direct" in r else None, r["diva"][K]], scale=100)
            my = fmt_best([r["direct"]["norm_regret"] if "direct" in r else None, r["diva"]["norm_regret"]], scale=100)
            return [str(r["diva"]["n"]), dp[0], OURS_CELL + dp[1], my[0], OURS_CELL + my[1], OURS_CELL + f"{r['diva'].get('over_acq', float('nan')):.2f}"]
        blocks.append(family_rows(CTRL_LLMS, cells, 9, lead=["\\multirow{" + str(len(CTRL_LLMS)) + "}{*}{" + ENVN[d] + "}"]))
    head = (r"\multirow{2}{*}[-0.6ex]{Domain} & \multirow{2}{*}[-0.6ex]{Family} & \multirow{2}{*}[-0.6ex]{Size} & \multirow{2}{*}[-0.6ex]{$n$} & \multicolumn{2}{c}{DP optimum} & \multicolumn{2}{c}{Myopic optimum} & \cellcolor{oursrow} \\"
            "\n" r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}" "\n" r" & & & & Direct & \cellcolor{oursrow}\diva{} & Direct & \cellcolor{oursrow}\diva{} & \cellcolor{oursrow}\multirow{-2}{*}[0.55ex]{\begin{tabular}[c]{@{}c@{}}\diva{}\\over-acq.\end{tabular}} \\")  # F6: lower row, so the colour of this row does not cover it
    for k, ch in enumerate(_env_blocks(blocks, 2)):
        out.append(_fam_tab(ch, r"\textbf{Controlled environments on all available test scenarios per domain ($n$ per row)" + (" (continued)" if k else "") + r"}. Regret ($\times$100) against the DP optimum and the myopic optimum.",
                            "tab:full_bin" + (f"_{k}" if k else ""), 6, head, lead=1))
    open(f"{OUT}/full.tex", "w").write("\n".join(out))
    labs = re.findall(r"\\label\{(tab:full_[^}]+)\}", "\n".join(out))
    open(f"{OUT}/full_refs.tex", "w").write("The per-cell results are in " + reflist(labs) + ".\n")
    return len(labs)


def table_episodes():
    """tab:episodes (main text): rows = methods, columns = domains | Avg., episode regret averaged over the fixed 20 LLMs
    (TBD unless all are complete). Per-LLM version: tab:full_episodes_llm (Full Results)."""
    from make_tables import method_rows_table, llm_rows_table
    E = json.load(open(f"{IS}/results/episodes.json"))

    def val(m, d):
        v = [_ep_val(E, t, m, d) for t in EP_CTRL_LLMS]
        return float(np.mean(v)) if all(x is not None for x in v) else None
    tex, vals, _ = method_rows_table(EP_METHS, [(d, ENVN[d]) for d in DOM_ORDER], val, "tab:episodes", len(EP_CTRL_LLMS), group=[(4, "Controlled environments")], star=True)
    # main-text tab:episodes is the merged knowledge-required | controlled table (episodes_all_table.build_merged)

    def v_avg(t, m):
        v = [_ep_val(E, t, m, d) for d in DOM_ORDER]
        return float(np.mean(v)) if all(x is not None for x in v) else None
    t = llm_rows_table(EP_CTRL_LLMS, [x for x in EP_METHS if x[0] != "diva_u"], v_avg, "tab:full_episodes_llm",
                       r"\textbf{Controlled episode regret per LLM}, averaged over domains (Table~\ref{tab:episodes}). Each LLM takes the final action, so Never varies by LLM.")
    open(f"{OUT}/full_episodes_llm.tex", "w").write(t)
    return sum(v is None for v in vals.values())


def table_realized():
    R = json.load(open(f"{IS}/results/realized_tab.json"))
    nm = ["Never", r"\diva{}", "", r"\diva{}-U", ""]
    blocks = []
    for ds in ["taiwan", "german", "cdc", "heart"]:
        def cells(tag, ds=ds):
            v = R.get(f"{tag}:{ds}")
            if not v or v["n"] < know_size(ds):  # missing or partial run: pending
                return [str(v["n"]) if v else PH] + _shade(nm, [PH] * 5)
            cu = v.get("ci_u", [None, float("nan"), float("nan")])
            return [str(v["n"])] + _shade(nm, [f"{100*v['stop']:.2f}", f"{100*v['diva']:.2f}", f"[{100*v['ci'][1]:.2f}, {100*v['ci'][2]:.2f}]", f"{100*v.get('diva_u', float('nan')):.2f}", f"[{100*cu[1]:.2f}, {100*cu[2]:.2f}]"])
        blocks.append(family_rows(KNOW_LLMS, cells, 9, lead=["\\multirow{" + str(len(KNOW_LLMS)) + "}{*}{" + ENVN[ds] + "}"]))
    head = "Env. & " + FAM_HEAD + r" & $n$ & Never & \cellcolor{oursrow}\diva{} & 95\% CI & \cellcolor{oursrow}\diva{}-U & 95\% CI \\"
    out = []
    for k, ch in enumerate(_env_blocks(blocks, 2)):
        out.append(_fam_tab(ch, r"\textbf{Realized utility on real records" + (" (continued)" if k else "") + r"} ($\times$100, normalized by $l$, net of costs). Checks reveal the record's actual values and the final action is taken with the policy's own belief and scored with the record's actual outcome. CI: Clustered bootstrap CI of the difference to never acquiring.",
                            "tab:realized" + (f"_{k}" if k else ""), 6, head, lead=1))
    open(f"{OUT}/realized.tex", "w").write("\n".join(out))
    labs = re.findall(r"\\label\{(tab:realized[^}]*)\}", "\n".join(out))
    open(f"{OUT}/realized_refs.tex", "w").write("Per-cell utilities are in " + reflist(labs) + ".\n")
    return len(labs)


def table_ms():
    from complete import MS_LLMS
    K = "dp_regret"
    A = json.load(open(f"{IS}/results/ms_test_ms2.json")) if os.path.exists(f"{IS}/results/ms_test_ms2.json") else {}
    Q = json.load(open(f"{IS}/results/ms_test_ms2q.json")) if os.path.exists(f"{IS}/results/ms_test_ms2q.json") else {}
    TU = json.load(open(f"{IS}/results/ms_tuned.json")) if os.path.exists(f"{IS}/results/ms_tuned.json") else {}

    def av(src, tag, m):
        v = [src[tag][d][m][K] for d in DOM_ORDER if tag in src and d in src[tag] and m in src[tag][d] and src[tag][d][m]["n"] >= 100]
        return None if len(v) < 4 else float(np.mean(v))
    # canonical method order within each group (stated beliefs | qualitative beliefs)
    cols = [(A, "direct"), (A, "always_stop"), (TU, "always_top"), (A, "fac_eig"), (TU, "fac_eig_tuned"), (A, "fac_es"), (TU, "fac_es_tuned"),
            (A, "diva_elicited_payoff"), (A, "diva"), (Q, "direct"), (Q, "diva")]
    names = [mname(m) for _, m in cols]

    def cells(tag):  # every LLM of the fixed set, TBD where missing
        vals = [av(src, tag, m) for src, m in cols]
        return _shade(names, fmt_best(vals[:9], scale=100) + fmt_best(vals[9:], scale=100))
    body = family_rows(MS_LLMS, cells, 13)
    O = OURS_CELL
    hdr = _h(names, "stack")  # F6: wide table, citations stacked
    hdr[7] = O + r"\begin{tabular}[c]{@{}c@{}}\diva{}\\(elic.)\end{tabular}"
    t = (r"\begin{table*}[t]" "\n" r"\centering" "\n" r"\small" "\n" r"\setlength{\tabcolsep}{3pt}" "\n" r"\begin{tabular}{@{}ll|ccccccccc|cc@{}}" "\n" r"\toprule" "\n"
         r"\multirow{2}{*}[-2ex]{Family} & \multirow{2}{*}[-2ex]{Size} & \multicolumn{1}{c}{Prompting} & \multicolumn{2}{c}{Reference} & \multicolumn{4}{c}{Ablation} & \multicolumn{2}{c|}{" + O + r"Ours} & \multicolumn{2}{c}{Verbal Beliefs} \\" "\n"
         r"\cmidrule(lr){3-3}\cmidrule(lr){4-5}\cmidrule(lr){6-9}\cmidrule(lr){10-11}\cmidrule(lr){12-13}" "\n"
         r" & & " + " & ".join(hdr[:9]) + r" & Direct & " + O + r"\diva{} \\" "\n"
         r"\midrule" "\n" + "\n".join(body) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n"
         r"\caption{\textbf{Multi-state environments.} Regret on held-out scenarios, normalized by half the payoff range. " + FAMILY_NOTE + "}\n" r"\label{tab:ms}" "\n" r"\end{table*}" "\n")
    open(f"{OUT}/ms.tex", "w").write(t)
    return len(body)


def table_confirm():
    KM = json.load(open(f"{IS}/results/know_main.json")); K = "dp_regret"
    rows = []
    ms = [m for m, _ in msort([(x, 0) for x in ["direct", "never", "always_top", "diva_kappa", "diva_beta", "diva", "diva_u"]])]
    names = [mname(m) for m in ms]
    for e in ["sec:ret_1", "sec:ret_5", "sec:vol_5", "sec:vol_20"]:
        en = ENVN[e.split(':')[1]]
        if not (know_ok(e) and all(all(f"f1000:{m}" in KM[e].get(t, {}) for m in ms) for t in KNOW_LLMS)):
            rows.append(en + " & " + " & ".join(_shade(names + [r"\diva{}-U", ""], [PH] * (len(ms) + 2))) + r" \\"); continue
        recs = [KM[e][t] for t in KNOW_LLMS]
        av = lambda m, k=K: np.mean([r[f"f1000:{m}"][k] for r in recs])
        cells = _pair([av(m) for m in ms], [av(m, "norm_regret") for m in ms])
        rows.append(f"{en} & " + " & ".join(_shade(names, cells)) +
                    f" & {OURS_CELL}{av('diva_u','stopped'):.2f} & {100*av('diva_u','placebo'):.1f} / {100*av('direct','placebo'):.1f} \\\\")
    t = _tab(["Env."] + _h(names) + [OURS_CELL + r"\begin{tabular}[c]{@{}c@{}}Stop\\\diva{}-U\end{tabular}", r"\begin{tabular}[c]{@{}c@{}}Placebo (\%)\\\diva{}-U / Direct\end{tabular}"], rows,
             r"\textbf{Held-out SEC filings.}", "tab:confirm", star=True, size=r"\small")
    open(f"{OUT}/confirm.tex", "w").write(t)
    return len(rows)


def table_know_detail():
    KM = json.load(open(f"{IS}/results/know_main.json"))
    ms = [m for m, _ in msort([(x, 0) for x in ["never", "always_top", "diva_kappa", "diva_beta", "diva", "diva_u"]])]
    names = [mname(m) for m in ms]
    rows = []
    for e in ["tab:taiwan", "tab:german", "tab:cdc", "tab:heart", "sec:ret_1", "sec:ret_5", "sec:vol_5", "sec:vol_20"]:
        nm = ENVN[e.split(":")[1]]
        if not (know_ok(e) and all(all(f"full:{m}" in KM[e].get(t, {}) for m in ms) for t in KNOW_LLMS)):
            rows.append(nm + " & " + " & ".join(_shade(names + ["", r"\diva{}", r"\diva{}-U"], [PH] * (len(ms) + 3))) + r" \\"); continue
        recs = [KM[e][t] for t in KNOW_LLMS]
        def av(m, k):
            return np.mean([r[f"full:{m}"][k] for r in recs])
        cells = _pair([av(m, "dp_regret") for m in ms], [av(m, "norm_regret") for m in ms])
        opt_stop = av("never", "dp_stop") if "dp_stop" in recs[0]["full:never"] else float("nan")
        rows.append(f"{nm} & " + " & ".join(_shade(names, cells)) + f" & {opt_stop:.2f} & {OURS_CELL}{av('diva','stopped'):.2f} & {OURS_CELL}{av('diva_u','stopped'):.2f}" + r" \\")
    t = _tab(["Env."] + _h(names) + [r"\begin{tabular}[c]{@{}c@{}}Stop\\opt.\end{tabular}", OURS_CELL + r"\begin{tabular}[c]{@{}c@{}}Stop\\\diva{}\end{tabular}", OURS_CELL + r"\begin{tabular}[c]{@{}c@{}}Stop\\\diva{}-U\end{tabular}"], rows,
             r"\textbf{Stopping behavior on knowledge-required environments.}", "tab:know_detail", star=True, size=r"\small")
    open(f"{OUT}/know_detail.tex", "w").write(t)
    return len(rows)


def table_altgold():
    p = f"{IS}/results/alt_gold.json"
    if not os.path.exists(p):
        return 0
    A = json.load(open(p)); ms = [(m, mname(m)) for m, _ in msort([("never", 0), ("direct", 0), ("always_top", 0), ("diva", 0), ("diva_u", 0)])]
    names = [n for _, n in ms]
    rows = []
    for ds in ["taiwan", "german", "cdc", "heart"]:
        need = min(1000, know_size(ds))
        if not all(t in A.get(ds, {}) and all(m in A[ds][t] and A[ds][t][m].get("n", 0) >= need for m, _ in ms) for t in KNOW_LLMS):
            rows.append(ENVN[ds] + " & " + " & ".join(_shade(names, [PH] * len(ms))) + r" \\"); continue
        cells = [A[ds][t] for t in KNOW_LLMS]
        # user 10-12: one-column layout, one row per reference model (main / naive Bayes) instead of "a / b" cells
        for k, (key, lab) in enumerate([("ref", "Main"), ("nb", "NB")]):
            vals = fmt_best([np.mean([c[m][key] for c in cells]) for m, _ in ms], scale=100)
            env = (r"\multirow{2}{*}{" + ENVN[ds] + "}") if k == 0 else ""
            rows.append(f"{env} & {lab} & " + " & ".join(_shade(names, vals)) + r" \\")
        rows.append(r"\cmidrule(lr){1-" + str(2 + len(ms)) + "}")
    rows = rows[:-1] if rows and rows[-1].startswith(r"\cmidrule") else rows
    t = _tab(["Env.", "Ref."] + _h(names), rows, r"\textbf{Robustness to the reference model.}", "tab:altgold", star=False, colsep="2pt", nlab=2)
    open(f"{OUT}/altgold.tex", "w").write(t)
    return len(rows)


def table_extra_bl():
    R = json.load(open(f"{IS}/results/bin_main.json")); K = "dp_regret"
    cols = [(c, mname(c)) for c, _ in msort([("bl_salience", 0), ("bl_uncertainty", 0), ("bl_eig", 0), ("bl_random", 0), ("bl_voi_prompt", 0), ("bl_voi_paper", 0), ("diva", 0)])]

    def cells(tag):
        vals = []
        for c, _ in cols:
            v = [R.get(tag, {}).get(d, {}).get(c + "@sub", {}) for d in DOM_ORDER]
            vals.append(float(np.mean([x[K] for x in v])) if all(x.get("n", 0) >= 300 for x in v) else None)
        f = fmt_best(vals, scale=100) if all(x is not None for x in vals) else [PH if x is None else f"{100*x:.2f}" for x in vals]
        return _shade([n for _, n in cols], f)
    rows = family_rows(CTRL_BL_LLMS, cells, 2 + len(cols))
    t = _fam_tab(rows, r"\textbf{Additional baselines on controlled environments.}", "tab:extra_bl", len(cols), FAM_HEAD + " & " + " & ".join(_h([n for _, n in cols])) + r" \\")
    open(f"{OUT}/extra_bl.tex", "w").write(t)
    return len(rows)


# episode display names (user rule 2026-10-11): the exact first-decision baselines keep their Table-3 names; Direct and
# Always ask take their first decision as in Table tab:know and then follow DIVA (stated in the captions)
EPN = {}  # hybrids keep their honest names (complete.MNAME: Direct -> DIVA, Always ask -> DIVA)
from complete import mrank as _mrank
K4MS = sorted([(m, EPN.get(m, mname(m))) for m in ["never", "direct1", "ask1", "beta", "faceig", "gen_dp", "diva", "diva_u", "diva_dp"]],
              key=lambda x: _mrank(x[1]))
K4MAIN = [x for x in K4MS if x[0] in ("direct1", "never", "ask1", "diva", "diva_u", "diva_dp")]  # the Table-3 methods


def _k4_ci(cells, a, b, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    common = None
    for c in cells:
        ids = set(c[a]["ids"]) & set(c[b]["ids"])
        common = ids if common is None else common & ids
    common = sorted(common or [])
    if not common:
        return None
    mat = []
    for c in cells:
        da = dict(zip(c[a]["ids"], c[a]["vals_ref"])); db = dict(zip(c[b]["ids"], c[b]["vals_ref"]))
        mat.append([da[i] - db[i] for i in common])
    mat = np.array(mat); est = mat.mean()
    bs = [mat[:, rng.integers(0, mat.shape[1], mat.shape[1])].mean() for _ in range(B)]
    return est, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def _k4_rows(key, llms=None):
    """rows = environments, columns = every exact policy (K4MS), test scenarios 300--999, averaged over llms (all 20 LLMs;
    the pre-registered confirmatory statistics keep K4_LLMS, see table_ci_main)"""
    llms = KNOW_LLMS if llms is None else llms
    E = json.load(open(f"{IS}/results/episodes_know_k4.json")); rows = []
    names = [n for _, n in K4MS]
    for e in ENV_ORDER:
        if not (know_ok(e, llms, 1000) and all(t in E.get(e, {}) and all(m in E[e][t] for m, _ in K4MS) for t in llms)):
            rows.append(ENVN[e] + " & " + " & ".join(_shade(names, [PH] * len(K4MS))) + r" \\"); continue
        cells = [E[e][t] for t in llms]
        col = [float(np.mean([c[m][key] for c in cells])) for m, _ in K4MS]
        rows.append(f"{ENVN[e]} & " + " & ".join(_shade(names, fmt_best(col, scale=100))) + r" \\")  # DP-Never CIs: appendix Table tab:ci_main
    return rows


def table_episodes_k4():
    p = f"{IS}/results/episodes_know_k4.json"
    if not os.path.exists(p):
        return 0
    from make_tables import method_rows_table
    E = json.load(open(p))

    def val(m, e):  # appendix Table tab:episodes_know_prereg: rows 300--999, the methods of the confirmatory test, all 20 LLMs
        if not (know_ok(e, KNOW_LLMS, 1000) and all(t in E.get(e, {}) and m in E[e][t] for t in KNOW_LLMS)):
            return None
        return float(np.mean([E[e][t][m]["regret_ref"] for t in KNOW_LLMS]))
    tex, _, _ = method_rows_table(K4MAIN, [(e, ENVN[e]) for e in ENV_ORDER], val, "tab:episodes_know_prereg", len(KNOW_LLMS),
                                  group=[(2, "Credit"), (2, "Clinical"), (4, "SEC filings (2025--2026)")], star=True, colsep="4.5pt")
    open(f"{OUT}/episodes_know_prereg.tex", "w").write(tex)  # main-text tab:episodes_know: episodes_all_table.py (rows 0-149)
    rows = _k4_rows("regret_ref")
    t = _tab(["Env."] + _h([n for _, n in K4MS], "stack"), rows, r"\textbf{Full episodes of every exact policy.}", "tab:episodes_know_exact", star=True, size=r"\small")
    open(f"{OUT}/episodes_know_exact.tex", "w").write(t)
    rows2 = _k4_rows("regret")
    t2 = _tab(["Env."] + _h([n for _, n in K4MS], "stack"), rows2, r"\textbf{Full episodes with the policy's own final action.}", "tab:episodes_know_own", star=True, size=r"\small")
    open(f"{OUT}/episodes_know_own.tex", "w").write(t2)
    return len(rows)


def table_episodes_know_first300():
    p = f"{IS}/results/episodes_know.json"
    if not os.path.exists(p):
        return 0
    E = json.load(open(p))
    ms = sorted([(m, EPN.get(m, mname(m))) for m in ["never", "always", "direct1", "ask1", "diva", "diva_u"]], key=lambda x: _mrank(x[1]))
    names = [n for _, n in ms]
    rows = []
    for e in ENV_ORDER:
        if not (know_ok(e, KNOW_LLMS, 300) and all(t in E.get(e, {}) and all(m in E[e][t] for m, _ in ms) for t in KNOW_LLMS)):
            rows.append(ENVN[e] + " & " + " & ".join(_shade(names + [r"\diva{}"], [PH] * (len(ms) + 1))) + r" \\"); continue
        cells = [E[e][t] for t in KNOW_LLMS]
        av = lambda m, k: np.mean([c[m][k] for c in cells])
        cellstr = _pair([av(m, "regret_ref") for m, _ in ms], [av(m, "regret") for m, _ in ms])
        rows.append(f"{ENVN[e]} & " + " & ".join(_shade(names, cellstr)) + f" & {OURS_CELL}{av('diva','n_acq'):.2f} / {av('diva_u','n_acq'):.2f}" + r" \\")
    t = _tab(["Env."] + _h(names) + [OURS_CELL + r"\begin{tabular}[c]{@{}c@{}}Acq.\\\diva{} / U\end{tabular}"], rows,
             r"\textbf{Exploratory full episodes} on the first 300 test scenarios.", "tab:episodes_know300", star=True, size=r"\small")
    open(f"{OUT}/episodes_know300.tex", "w").write(t)
    return len(rows)


def table_large():
    """two 70B-class LLMs: rows = methods (canonical order), columns = LLMs (canonical order)"""
    B = json.load(open(f"{IS}/results/bin_main.json")); K = "dp_regret"
    cols = [(c, mname(c)) for c, _ in msort([("bl_direct", 0), ("bl_cot_need", 0), ("bl_pal", 0), ("bl_voi_paper", 0), ("always_stop", 0), ("diva", 0)])]
    tags = lsort(["l70", "q72"])
    val = {}
    for tag in tags:
        for c, _ in cols:
            v = [B.get(tag, {}).get(d, {}).get(c + "@sub", {}) for d in DOM_ORDER]
            val[(tag, c)] = float(np.mean([x[K] for x in v])) if all(x.get("n", 0) >= 300 for x in v) else None
    colf = {t: fmt_best([val[(t, c)] for c, _ in cols], scale=100) for t in tags}
    rows = [(r"\rowcolor{oursrow} " if is_ours(n) else "") + mlabel(n) + " & " + " & ".join(colf[t][i] for t in tags) + r" \\" for i, (c, n) in enumerate(cols)]
    rows = group_rows([n for _, n in cols], rows, 1 + len(tags))
    # user 10-12: one-line LLM headers (Qwen2.5-72B, Llama-3.3-70B fit)
    t = _tab(["Method"] + [NAMES[x] for x in tags], rows, r"\textbf{Larger LLMs on controlled environments.}", "tab:large", size=r"\small", colsep="2pt")
    open(f"{OUT}/large.tex", "w").write(t)
    return len(rows)


def table_const():
    """tab:const: generic specification (every check the same strength delta; ONE delta per variant selected on the average
    development DP regret over the eight environments, the protocol of DIVA; scripts/const_fair.py) against the elicited
    specification, all on the first 300 test scenarios of Table tab:know (LLM columns averaged over the 20 LLMs)."""
    p = f"{IS}/results/const_fair.json"
    if not os.path.exists(p):
        return 0
    C = json.load(open(p)); sel = C["_selected"]; K = json.load(open(f"{IS}/results/know_main.json")); rows = []
    names = ["Never", r"Generic \diva{}", r"Generic \diva{}-U", r"\diva{}", r"\diva{}-U"]
    for e in ENV_ORDER:
        en = e.split(":")[1] if ":" in e else e
        v = C.get(en); recs = K.get(("tab:" if en in ("taiwan", "german", "cdc", "heart") else "sec:") + en, {})
        okl = all(t in recs and "s300:diva" in recs[t] and "s300:diva_u" in recs[t] for t in KNOW_LLMS)
        if not v:
            rows.append(ENVN[e] + " & " + " & ".join(_shade(names, [PH] * 5)) + r" \\")
            continue
        llm = lambda m, k: float(np.mean([recs[t][f"s300:{m}"][k] for t in KNOW_LLMS])) if okl else None
        cells = [v["s300:never"], v[f"s300:diva:{sel['diva']}"], v[f"s300:diva_u:{sel['diva_u']}"]]
        dps = [c["dp_regret"] for c in cells] + [llm("diva", "dp_regret"), llm("diva_u", "dp_regret")]
        mys = [c["norm_regret"] for c in cells] + [llm("diva", "norm_regret"), llm("diva_u", "norm_regret")]
        out = _pair(dps, mys) if okl else [PH if d is None else f"{100*d:.2f} / {100*m:.2f}" for d, m in zip(dps, mys)]
        rows.append(f"{ENVN[e]} & " + " & ".join(_shade(names, out)) + r" \\")
    hdr = ["Never", r"\begin{tabular}[c]{@{}c@{}}Generic\\\diva{}\end{tabular}", r"\begin{tabular}[c]{@{}c@{}}Generic\\\diva{}-U\end{tabular}", OURS_CELL + r"\diva{}", OURS_CELL + r"\diva{}-U"]
    t = _tab(["Env."] + hdr, rows, r"\textbf{What the LLM's knowledge contributes.}", "tab:const", star=True, size=r"\small")
    open(f"{OUT}/const.tex", "w").write(t)
    return len(rows)


def table_ci_know():
    p = f"{IS}/results/ci_tables.json"
    if not os.path.exists(p):
        return 0
    C = json.load(open(p)).get("knowledge", {}); rows = []
    keys = [("diva-never:dp", r"\diva{}$-$Never", "DP"), ("diva-never:myo", r"\diva{}$-$Never", "myopic"), ("diva_u-never:dp", r"\diva{}-U$-$Never", "DP"),
            ("diva_u-never:myo", r"\diva{}-U$-$Never", "myopic"), ("diva_u-direct:dp", r"\diva{}-U$-$Direct", "DP")]
    two = lambda a, b: r"\begin{tabular}[c]{@{}c@{}}" + a + r"\\\relax " + b + r"\end{tabular}"  # \relax: a following "[" is not a row-skip argument
    for e in ENV_ORDER:
        v = C.get(e)
        if not v or not (v["n_llm"] == len(KNOW_LLMS) and know_ok(e, KNOW_LLMS, 300)):
            rows.append(_phrow(ENVN[e], len(keys))); continue
        rows.append(f"{ENVN[e]} & " + " & ".join(two(f"{100*v[k][0]:.2f}", f"[{100*v[k][1]:.2f}, {100*v[k][2]:.2f}]") for k, _, _ in keys) + r" \\")
    t = _tab(["Env."] + [two(a, "(" + b + ")") for _, a, b in keys], rows,
             r"\textbf{Paired differences with 95\% CIs on knowledge-required environments.}", "tab:ci_know", star=True, size=r"\small")
    open(f"{OUT}/ci_know.tex", "w").write(t)
    return len(rows)


def table_transfer():
    p = f"{IS}/results/transfer.json"
    if not os.path.exists(p):
        return 0
    T = json.load(open(p)); rows = []
    nm = dict(ENVN, sec="SEC filings")
    two = lambda a, b: r"\begin{tabular}[c]{@{}c@{}}" + a + r"\\\relax " + b + r"\end{tabular}"  # \relax: a following "[" is not a row-skip argument
    for d in [x for x in ENV_ORDER[:4] + ["sec"] if x in T]:
        v = T[d]
        envs = ["ret_1", "ret_5", "vol_5", "vol_20"] if d == "sec" else [d]
        ok = all(know_ok(e, KNOW_LLMS, 400) for e in envs)  # first 400 test scenarios of every LLM
        vals = [v["test_point"], v["test_global_u"], v["test_selected"]]
        cells = [f"{x:.2f}" for x in vals] if ok else [PH] * 3
        rows.append(f"{nm.get(d, d)} & ({v['selected'][0]}, {v['selected'][1]}) & " + " & ".join(_shade([r"\diva{}", r"\diva{}-U", r"\diva{}-U"], cells)) + r" \\")
    t = _tab([two("Held-out", "domain"), two("Selected", r"($\lambda$, $\sigma$)"), OURS_CELL + two(r"\diva{}", "(point)"), OURS_CELL + two(r"\diva{}-U", r"($\lambda{=}0.75$)"), OURS_CELL + two(r"\diva{}-U", "(selected)")], rows,
             r"\textbf{Leave-one-domain-out transfer.} Regret relative to never asking on the held-out domain.", "tab:transfer")
    open(f"{OUT}/transfer.tex", "w").write(t)
    return len(rows)


def table_dev_diag():
    p = f"{IS}/results/episodes_know_dev.json"
    if not os.path.exists(p):
        return 0
    E = json.load(open(p)); rows = []
    ms = [(m, mname(m)) for m, _ in msort([("never", 0), ("direct1", 0), ("diva", 0), ("diva_u", 0), ("diva_dp", 0)])]
    names = [n for _, n in ms]
    for e in ENV_ORDER:
        cells = [c for c in E.get(e, {}).values() if all(m in c for m, _ in ms)]
        if not cells:
            continue
        col = [float(np.mean([c[m]['regret_ref'] for c in cells])) for m, _ in ms]
        rows.append(f"{ENVN[e]} & {len(cells)} & " + " & ".join(_shade(names, fmt_best(col, scale=100))) + r" \\")
    t = _tab(["Env.", "LLMs"] + _h(names), rows,
             r"\textbf{Development episodes} behind the pre-registration of \diva{}-DP (reference-posterior final action).", "tab:dev_diag", star=True)
    open(f"{OUT}/dev_diag.tex", "w").write(t)
    return len(rows)


def table_gold():
    p = f"{IS}/results/gold_disagree.json"
    if not os.path.exists(p):
        return 0
    G = json.load(open(p)); rows = []
    two = lambda a, b: r"\begin{tabular}[c]{@{}c@{}}" + a + r"\\\relax " + b + r"\end{tabular}"  # \relax: a following "[" is not a row-skip argument
    for e in ENV_ORDER:
        if e in G:
            v = G[e]
            rows.append(f"{ENVN[e]} & {v['n']} & {v['myopic_stop']:.2f} & {v['dp_stop']:.2f} & {v['myo_stop_dp_acq']:.2f} & {v['same_check_when_both_acquire']:.2f} \\\\")
    t = _tab(["Env.", "$n$", two("Stop", "(myopic)"), two("Stop", "(DP)"), two("Myopic stop,", "DP acquire"), two("Same", "check")], rows,
             r"\textbf{Agreement of the two first-decision optima} on the knowledge-required test scenarios.", "tab:gold")
    open(f"{OUT}/gold.tex", "w").write(t)
    return len(rows)


def table_think_know():
    """Native reasoning (direct policy with thinking) on knowledge-required environments, first 300 test scenarios."""
    from evalx import read as _read, score as _score, summarize as _sum, choose_max as _cm
    from eval_tab import diva_scores as _ds
    from eval_real import fio_real as _fr
    rows = []; res = {}
    ms = ["direct", "think", "never", "diva"]  # canonical order
    names = [mname(m) for m in ms]
    for kind, envs in [("tab", ["taiwan", "german", "cdc", "heart"]), ("sec", ["ret_1", "ret_5", "vol_5", "vol_20"])]:
        for e in envs:
            ep = f"{IS}/envs_tab/data/{e}_test.jsonl.gz" if kind == "tab" else f"{IS}/envs_real/data/{e}_test.jsonl.gz"
            env = {x["id"]: x for x in _read(ep)[:300]}
            cells = []; ctags = []
            for f in glob.glob(f"{IS}/outputs/think_know/dir_*_{e}.jsonl.gz"):
                tag = os.path.basename(f)[4:-9][: -(len(e) + 1)]
                if tag not in REASON_KNOW:
                    continue
                th = {r["id"]: max(r["dist"], key=r["dist"].get) for r in _read(f) if r.get("dist")}
                dp = f"{IS}/outputs/test_tab/dir_{tag}_{e}.jsonl.gz" if kind == "tab" else f"{IS}/outputs/test_real/dir_{tag}_{e}.jsonl.gz"
                fp = f"{IS}/outputs/test_tab/fk_{tag}_{e}.jsonl.gz" if kind == "tab" else f"{IS}/outputs/test_real/fr_{tag}_{e}.jsonl.gz"
                if not (os.path.exists(dp) and os.path.exists(fp)):
                    continue
                dr = {r["id"]: max(r["dist"], key=r["dist"].get) for r in _read(dp) if r.get("dist")}
                fr = {r["id"]: r for r in _read(fp)}
                ids = [i for i in env if i in th and i in dr and i in fr]
                if len(ids) < len(env):  # all 300 scenarios required
                    continue
                R = {"never": [], "direct": [], "think": [], "diva": []}
                for i in ids:
                    sc = env[i]
                    f_ = _ds(sc, fr[i], "cond", "greedy", "elicit", 0.75) if kind == "tab" else _fr(sc, fr[i], lam=0.75)
                    for m, c in (("never", "S"), ("direct", dr[i]), ("think", th[i]), ("diva", _cm(f_["net"]))):
                        R[m].append(_score(sc, c))
                cells.append({m: _sum(v) for m, v in R.items()}); ctags.append(tag)
            if sorted(ctags) != sorted(REASON_KNOW):  # every reasoning-capable LLM of the fixed set
                rows.append(ENVN[e] + " & " + " & ".join(_shade(names, [PH] * len(ms))) + r" \\"); continue
            res[e] = len(cells)
            av = lambda m, k: float(np.mean([c[m][k] for c in cells]))
            rows.append(f"{ENVN[e]} & " + " & ".join(_shade(names, _pair([av(m, 'dp_regret') for m in ms], [av(m, 'norm_regret') for m in ms]))) + r" \\")
    if not rows:
        return 0
    t = _tab(["Env."] + _h(names), rows,
             r"\textbf{Native reasoning on knowledge-required environments.}", "tab:think_know", star=True, size=r"\small")
    open(f"{OUT}/think_know.tex", "w").write(t)
    return len(rows)


def table_realized_ref():
    p = f"{IS}/envs_tab/data/realized_utility_test.json"
    if not os.path.exists(p):
        return 0
    D = json.load(open(p)); rows = []
    for ds in ["taiwan", "german", "cdc", "heart"]:
        if ds in D:
            v = D[ds]
            rows.append(f"{ENVN[ds]} & {v['stop_now'][0]:.3f} & {v['dp_policy'][0]:.3f} & {v['dp_minus_stop'][0]:.3f} $\\pm$ {v['dp_minus_stop'][1]:.3f} \\\\")
    t = _tab(["Env.", "Never", r"\begin{tabular}[c]{@{}c@{}}Reference\\DP\end{tabular}", r"\begin{tabular}[c]{@{}c@{}}Difference\\($\pm$ SE)\end{tabular}"], rows,
             r"\textbf{Realized utility of the reference model's own DP policy}, net of costs (higher is better).", "tab:realized_ref")
    open(f"{OUT}/realized_ref.tex", "w").write(t)
    return len(rows)


def table_know1000():
    """Knowledge-required first decision on the first 1,000 test scenarios (identical rows per LLM) with all LLM baselines."""
    from evalx import read as _read, score as _score, summarize as _sum, choose_max as _cm
    from eval_tab import diva_scores as _ds, diva_u as _du
    from eval_real import fio_real as _fr, fio_real_u as _fu
    BLM = ["cot_need", "bed_llm", "voi_prompt", "pal", "voi_paper"]
    ms = [(m, mname(m)) for m, _ in msort([("direct", 0), ("cot_need", 0), ("bed_llm", 0), ("voi_prompt", 0), ("pal", 0), ("voi_paper", 0), ("never", 0), ("diva", 0), ("diva_u", 0)])]
    names = [n for _, n in ms]
    rows = []
    for kind, envs in [("tab", ["taiwan", "german", "cdc", "heart"]), ("sec", ["ret_1", "ret_5", "vol_5", "vol_20"])]:
        for e in envs:
            ep = f"{IS}/envs_tab/data/{e}_test.jsonl.gz" if kind == "tab" else f"{IS}/envs_real/data/{e}_test.jsonl.gz"
            envl = _read(ep)[:1000]; env = {x["id"]: x for x in envl}
            cells = []
            for f in glob.glob(f"{IS}/outputs/test_tab/fk_*_{e}.jsonl.gz" if kind == "tab" else f"{IS}/outputs/test_real/fr_*_{e}.jsonl.gz"):
                tag = os.path.basename(f)[3:-9][: -(len(e) + 1)]
                bl = {}
                for m in BLM:
                    pth = f"{IS}/baselines/outputs/{tag}/{m}/{e}_test.jsonl.gz"
                    if os.path.exists(pth):
                        bl[m] = {r["id"]: r["choice"] for r in _read(pth) if r.get("choice") is not None}
                if len(bl) < len(BLM):
                    continue
                dp = f"{IS}/outputs/test_tab/dir_{tag}_{e}.jsonl.gz" if kind == "tab" else f"{IS}/outputs/test_real/dir_{tag}_{e}.jsonl.gz"
                dr = {r["id"]: max(r["dist"], key=r["dist"].get) for r in _read(dp) if r.get("dist")} if os.path.exists(dp) else {}
                fr = {r["id"]: r for r in _read(f)}
                ids = [i for i in env if i in fr and i in dr and all(i in bl[m] for m in BLM)]
                if len(ids) < len(envl) or tag not in KNOW_LLMS:  # all 1,000 scenarios required
                    continue
                R = {m: [] for m, _ in ms}
                for i in ids:
                    sc = env[i]
                    f_ = _ds(sc, fr[i], "cond", "greedy", "elicit", 0.75) if kind == "tab" else _fr(sc, fr[i], lam=0.75)
                    fu = _du(sc, fr[i], 1.0) if kind == "tab" else _fu(sc, fr[i], 1.0)
                    ch = {"direct": dr[i], "never": "S", "diva": _cm(f_["net"]), "diva_u": _cm(fu["net"])}
                    ch.update({m: bl[m][i] for m in BLM})
                    for m, _ in ms:
                        R[m].append(_score(sc, ch[m]))
                cells.append({m: _sum(v) for m, v in R.items()})
            if len(cells) < len(KNOW_LLMS):  # all 20 LLMs required
                rows.append([PH] * len(ms)); continue
            col = [float(np.mean([c[m]["dp_regret"] for c in cells])) for m, _ in ms]
            rows.append(fmt_best(col, scale=100))
    if not rows:
        return 0
    # F6 (user 10-11): the layout of Table tab:know (methods as rows, environments as columns), so no header needs wrapping
    from make_tables import _two_line, OURS_ROW
    envs8 = ["taiwan", "german", "cdc", "heart", "ret_1", "ret_5", "vol_5", "vol_20"]
    body = [(OURS_ROW if is_ours(n) else "") + _two_line(n) + " & " + " & ".join(rows[j][i] for j in range(len(envs8))) + r" \\"
            for i, n in enumerate(names)]
    body = group_rows(names, body, 1 + len(envs8))
    t = ("\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{l|cccccccc}\n\\toprule\n"
         "\\multirow{2}{*}[-0.7ex]{\\begin{tabular}[c]{@{}l@{}}Method\\\\(Avg.\\ across " + str(len(KNOW_LLMS)) + " LLMs)\\end{tabular}} & "
         "\\multicolumn{2}{c}{Credit} & \\multicolumn{2}{c}{Clinical} & \\multicolumn{4}{c}{SEC filings (2025--2026)} \\\\\n"
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-9}\n & " + " & ".join(ENVN[e] for e in envs8) + " \\\\\n\\midrule\n"
         + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\caption{CAPTION}\n\\label{tab:know1000}\n\\end{table*}\n")
    open(f"{OUT}/know1000.tex", "w").write(t)
    return len(rows)


def table_para():
    import sys as _s
    _s.path.insert(0, f"{IS}/scripts")
    from evalx import read, fac_scores, choose_max, score, summarize
    from eval_tab import diva_scores, diva_u

    def cells(tag):
        vals = {"orig": [], "p1": [], "p2": []}
        for d in DOM_ORDER:
            env = {s_["id"]: s_ for s_ in read(f"{IS}/envs_synth/data/binary/{d}_test.jsonl.gz")[:300]}
            src = {"orig": f"{IS}/outputs/test_bin/facnum_{tag}_{d}.jsonl.gz", "p1": f"{IS}/outputs/para/fac_p1_{tag}_{d}.jsonl.gz", "p2": f"{IS}/outputs/para/fac_p2_{tag}_{d}.jsonl.gz"}
            R = {k: ({r["id"]: r for r in read(v)} if os.path.exists(v) else {}) for k, v in src.items()}
            ids = [i for i in env if all(i in R[k] for k in R)]
            if len(ids) < len(env):  # all 300 scenarios required
                return [PH] * 5
            for k in R:
                vals[k].append(summarize([score(env[i], choose_max(fac_scores(env[i], R[k][i])["net"])) for i in ids])["dp_regret"])
        rec = []
        for ds in ["taiwan", "german", "cdc", "heart"]:
            env = {s_["id"]: s_ for s_ in read(f"{IS}/envs_tab/data/{ds}_test.jsonl.gz")}
            pa, pb = f"{IS}/outputs/test_tab/fk_{tag}_{ds}.jsonl.gz", f"{IS}/outputs/para/fk_p1_{tag}_{ds}.jsonl.gz"
            a = {r["id"]: r for r in read(pa)} if os.path.exists(pa) else {}; b = {r["id"]: r for r in read(pb)} if os.path.exists(pb) else {}
            ids = [i for i in env if i in a and i in b]
            nb = len(b)
            if ids and len(ids) == min(len(env), nb) and nb > 0:
                rec.append((summarize([score(env[i], choose_max(diva_u(env[i], a[i], 1.0)["net"])) for i in ids])["dp_regret"],
                            summarize([score(env[i], choose_max(diva_u(env[i], b[i], 1.0)["net"])) for i in ids])["dp_regret"]))
        rt = [f"{100*np.mean([x for x, _ in rec]):.2f}", f"{100*np.mean([y for _, y in rec]):.2f}"] if len(rec) == 4 else [PH, PH]
        return [f"{100*np.mean(vals['orig']):.2f}", f"{100*np.mean(vals['p1']):.2f}", f"{100*np.mean(vals['p2']):.2f}"] + rt
    rows = family_rows(PARA_LLMS, cells, 7)
    head = (r"\multirow{2}{*}[-0.6ex]{Family} & \multirow{2}{*}[-0.6ex]{Size} & \multicolumn{3}{c}{Controlled (\diva{})} & \multicolumn{2}{c}{Records (\diva{}-U)} \\" "\n"
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}" "\n" r" & & Original & Para.~1 & Para.~2 & Original & Para.~1 \\")
    t = _fam_tab(rows, r"\textbf{Robustness to paraphrased elicitation questions.} Regret ($\times$100) of \diva{} on 300 controlled test scenarios per domain and of \diva{}-U on the tabular records, with the original and paraphrased questions.",
                 "tab:para", 5, head, star=True)
    open(f"{OUT}/para.tex", "w").write(t)
    return len(rows)


def table_ci_main():
    """tab:ci_main (appendix): the 95% CIs moved out of the main-text tables. Top: controlled Delta CI of DIVA minus the best
    LLM baseline per LLM (per-LLM Table tab:full_ctrl_llm). Bottom: knowledge-required episode CI of
    DIVA-DP minus never acquiring (Table tab:episodes_know). Same completeness rules as the source tables."""
    B = json.load(open(f"{IS}/results/bin_main.json"))
    CI = json.load(open(f"{IS}/results/ci_tables.json"))["controlled"] if os.path.exists(f"{IS}/results/ci_tables.json") else {}
    from complete import REASON_CTRL
    bl = ["bl_direct", "bl_cot_need", "bl_sc_cot", "bl_react", "bl_voi_prompt", "bl_voi_paper", "bl_finrisk_ask", "bl_bed_llm", "bl_pal", "think", "diva"]
    def c1(tag):
        ok = all(B.get(tag, {}).get(d, {}).get(m + "@sub", {}).get("n", 0) >= 300 for d in DOM_ORDER
                 for m in bl if m != "think" or tag in REASON_CTRL)
        ci = CI.get(tag)
        return [f"{100*ci['ci'][0]:.2f} [{100*ci['ci'][1]:.2f}, {100*ci['ci'][2]:.2f}]" if (ok and ci) else PH]
    # user 10-12: one full-width table, LLMs in two side-by-side halves next to the environments (no tall one-column list)
    from complete import LLM_FAMILIES, SIZE_OF
    neg = lambda x: x.replace("-", "$-$")
    llm = [(f"{f} {SIZE_OF[t]}", neg(c1(t)[0])) for f, ts in LLM_FAMILIES for t in ts if t in CTRL_BL_LLMS]
    p = f"{IS}/results/episodes_know_k4.json"
    E = json.load(open(p)) if os.path.exists(p) else {}
    env = []
    for e in ENV_ORDER:
        if not (know_ok(e, K4_LLMS, 1000) and all(t in E.get(e, {}) and all(m in E[e][t] for m, _ in K4MS) for t in K4_LLMS)):
            env.append((ENVN[e], PH)); continue
        ci = _k4_ci([E[e][t] for t in K4_LLMS], "diva_dp", "never")
        env.append((ENVN[e], neg(f"{100*ci[0]:.2f} [{100*ci[1]:.2f}, {100*ci[2]:.2f}]") if ci else PH))
    h = (len(llm) + 1) // 2
    nr = max(h, len(env))
    cell = lambda L, i: L[i] if i < len(L) else ("", "")
    rows = [" & ".join([*cell(llm, i), *cell(llm, h + i), *cell(env, i)]) + r" \\" for i in range(nr)]
    l1, l2 = llm, env
    tex = (r"\begin{table*}[t]" "\n" r"\centering" "\n" r"\small" "\n" r"\setlength{\tabcolsep}{3pt}" "\n"
           r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lc|lc|lc@{}}" "\n" r"\toprule" "\n"
           r"\multicolumn{4}{c|}{\diva{} $-$ best LLM baseline (controlled)} & \multicolumn{2}{c}{\diva{}-DP $-$ Never (episodes)} \\" "\n"
           r"\cmidrule(lr){1-4}\cmidrule(lr){5-6}" "\n"
           r"LLM & $\Delta$ [95\% CI] & LLM & $\Delta$ [95\% CI] & Env. & $\Delta$ [95\% CI] \\" "\n" r"\midrule" "\n"
           + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular*}" "\n"
           r"\caption{\textbf{Paired differences with 95\% CIs.} Left: \diva{} minus each LLM's best baseline (no Reason.). Right: \diva{}-DP minus Never on pre-registered episodes (scenarios 300--999, 18 LLMs). Negative values favor \diva{}.}" "\n"
           r"\label{tab:ci_main}" "\n" r"\end{table*}" "\n")
    open(f"{OUT}/ci_main.tex", "w").write(tex)
    return len(l1) + len(l2)


def table_k4():
    """tab:k4 from results/k4.json (written by k4_table.py): all LLMs of the sampled-specification set on every test scenario."""
    p = f"{IS}/results/k4.json"
    if not os.path.exists(p):
        return 0
    out = json.load(open(p)); rows = []
    for e in ENV_ORDER:
        c = out.get(e, {})
        need = min(1000, know_size(e))  # every environment: test rows 0-999
        if not all(t in c and c[t]["n"] >= need for t in K4SPEC_LLMS):
            rows.append(ENVN[e] + f" & {PH} & {PH} & {OURS_CELL}{PH} & {OURS_CELL}{PH} & {PH}" + r" \\"); continue
        c = [c[t] for t in K4SPEC_LLMS]
        av = lambda k: float(np.mean([x[k] for x in c]))
        f = fmt_best([av("never"), av("greedy"), av("mean")], scale=100)
        rows.append(f"{ENVN[e]} & {int(np.mean([x['n'] for x in c]))} & {f[0]} & {OURS_CELL}{f[1]} & {OURS_CELL}{f[2]} & {av('agree'):.2f} \\\\")
    t = _tab(["Env.", "$n$", "Never", OURS_CELL + r"\begin{tabular}[c]{@{}c@{}}\diva{}\\(greedy)\end{tabular}", OURS_CELL + r"\begin{tabular}[c]{@{}c@{}}\diva{}\\(mean of 5)\end{tabular}", "Agreement"], rows,
             r"\textbf{Sampled specifications.}", "tab:k4", star=True, size=r"\small")
    open(f"{OUT}/k4.tex", "w").write(t)
    return len(rows)


def table_voiask():
    """tab:voiask_know from results/voiask_know.json (written by voiask_know.py): all 20 LLMs on the first 300 scenarios."""
    p = f"{IS}/results/voiask_know.json"
    if not os.path.exists(p):
        return 0
    out = json.load(open(p)); K = "dp_regret"; rows = []
    ms = [(m, mname(m)) for m, _ in msort([("direct", 0), ("voi_ask", 0), ("never", 0), ("diva", 0), ("diva_u", 0)])]
    names = [n for _, n in ms]
    for e in ENV_ORDER:
        c = out.get(e, {})
        if not all(t in c and all(c[t][m]["n"] >= min(300, know_size(e)) for m, _ in ms) for t in KNOW_LLMS):
            rows.append(ENVN[e] + " & " + " & ".join(_shade(names, [PH] * len(ms))) + f" & {PH}" + r" \\"); continue
        cells = [c[t] for t in KNOW_LLMS]
        av = lambda m, k: float(np.mean([x[m][k] for x in cells]))
        rows.append(f"{ENVN[e]} & " + " & ".join(_shade(names, _pair([av(m, K) for m, _ in ms], [av(m, 'norm_regret') for m, _ in ms]))) + f" & {np.mean([x['voi_ask']['stopped'] for x in cells]):.2f}" + r" \\")
    t = _tab(["Env."] + _h(names) + [r"\begin{tabular}[c]{@{}c@{}}Stop\\VoI-Ask\end{tabular}"], rows, r"\textbf{VoI-Ask on knowledge-required environments.}", "tab:voiask_know", star=True, size=r"\small")
    open(f"{OUT}/voiask_know.tex", "w").write(t)
    return len(rows)


def _h(names, mode=False):
    """header cells of method columns, our methods shaded. mode False: name / (citation) at natural width;
    "stack": name / authors / (year) for tables that would otherwise not fit (F6, user 10-11: never split a method name)"""
    return [(OURS_CELL if is_ours(n) else "") + hcell(n, True, narrow=mode) for n in names]



def _pair(dps, mys):
    """'a / b' cells with best / second best marked separately for the DP and the myopic optimum"""
    a = fmt_best(dps, scale=100); b = fmt_best(mys, scale=100)
    return [f"{x} / {y}" for x, y in zip(a, b)]



def _shade(names, cells):
    return [(OURS_CELL if is_ours(n) else "") + c for n, c in zip(names, cells)]



# controlled episodes: the methods of Table tab:know (canonical names/order); DIVA-U equals DIVA (stated reliabilities)
EP_METHS = [(m, mname(m)) for m, _ in msort([("bl_direct", 0), ("bl_cot_need", 0), ("bl_sc_cot", 0), ("bl_react", 0), ("bl_finrisk_ask", 0),
                                              ("bl_voi_prompt", 0), ("bl_voi_paper", 0), ("bl_bed_llm", 0), ("bl_pal", 0), ("bl_never", 0),
                                              ("always_top", 0), ("fiovoi@bl", 0), ("diva_u", 0), ("fiovoi_dp@bl", 0)])]



def _ep_val(E, tag, m, d):
    m = "fiovoi@bl" if m == "diva_u" else m  # DIVA-U is DIVA when reliabilities are stated
    m = "bl_always_all" if m == "always_top" else m  # controlled Always ask episodes: baselines policy always_all (acquire every check, then decide)
    r = E.get(tag, {}).get(d, {})
    # the first 100 test scenarios x two common random worlds
    ok = (r.get(m + ":n", 0) >= 200) if "@bl" in m else (len(r.get(m + ":ids", [])) >= 100)
    return r.get(m) if ok else None



def _fam_tab(rows, caption, label, ncols, head, star=True, colsep="3pt", lead=0):
    """table with family-grouped LLM rows (complete.family_rows): lead label columns, Family | Size, then ncols value columns"""
    env = "table*" if star else "table"
    spec = "@{}" + "l" * lead + "ll|" + "c" * ncols + "@{}"
    return ("\\begin{" + env + "}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{" + colsep + "}\n\\begin{tabular}{" + spec + "}\n\\toprule\n"
            + head + "\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\caption{" + caption + " " + FAMILY_NOTE + "}\n\\label{" + label + "}\n\\end{" + env + "}\n")



def _env_blocks(blocks, per):
    """chunks of whole environment blocks (each a list of rows), separated by \\midrule"""
    out = []
    for i in range(0, len(blocks), per):
        rows = []
        for j, b in enumerate(blocks[i:i + per]):
            if j:
                rows.append("\\midrule")
            rows += b
        out.append(rows)
    return out


if __name__ == "__main__":
    for f in sys.argv[1:] or ["ms_diag", "decomp", "qual", "controls", "full", "episodes", "realized", "ms", "confirm", "know_detail", "para", "altgold", "extra_bl", "episodes_k4", "episodes_know_first300", "large", "const", "ci_know", "transfer", "dev_diag", "gold", "think_know", "realized_ref", "know1000", "k4", "voiask", "ci_main"]:
        print(f, globals()["table_" + f]())
