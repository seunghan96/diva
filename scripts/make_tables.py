"""LaTeX tables from results/*.json -> paper/tables/*.tex (numbers x100, DP-regret; best bold, second underlined)."""
import json, os, sys
import numpy as np
IS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{IS}/paper/tables"; os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from complete import group_rows
from complete import NAMES, LLM_ORDER, METHOD_ORDER, MCITE, mname, msort, lsort, fmt_best, ENVN, DOM_ORDER, ENV_ORDER
DOMS = ["finance", "credit", "clinical", "legal"]
KEY = "dp_regret"
PH = "\\textcolor{gray}{TBD}"


def fmt_row(vals, lower=True):
    xs = [v for v in vals if v is not None]
    order = sorted(set(xs)) if lower else sorted(set(xs), reverse=True)
    out = []
    for v in vals:
        if v is None:
            out.append("--"); continue
        s = f"{100*v:.2f}"
        if order and v == order[0]:
            s = f"\\textbf{{{s}}}"
        elif len(order) > 1 and v == order[1]:
            s = f"\\underline{{{s}}}"
        out.append(s)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Shared builders: \small, no adjustbox, vertical rule after the label column (and before Avg.),
# ours shaded (oursrow), multirow-centered single header cells, canonical method / LLM / environment orders, TBD cells.
# ---------------------------------------------------------------------------------------------------------------------
OURS_ROW = "\\rowcolor{oursrow} "
OURS_CELL = "\\cellcolor{oursrow}"
# two-line headers for long method names in the per-LLM (method-column) tables
HDR2 = {"Always ask": "Always\\\\ask", "BED-LLM": "BED-\\\\LLM", "\\diva{} w/ EIG": "\\diva{} w/\\\\EIG", "\\diva{} w/ EIG$^\\ddagger$": "\\diva{} w/\\\\EIG$^\\ddagger$",
        "\\diva{} w/ ES": "\\diva{} w/\\\\ES", "\\diva{} w/ ES$^\\ddagger$": "\\diva{} w/\\\\ES$^\\ddagger$", "VoI-Ask": "VoI-\\\\Ask",
        "\\diva{}-U": "\\diva{}-\\\\U", "\\diva{}-DP": "\\diva{}-\\\\DP"}


def reflist(labels):
    """'Tables~\\ref{a}, \\ref{b}, and~\\ref{c}' (every label referenced individually)"""
    r = ["\\ref{" + l + "}" for l in labels]
    if len(r) == 1:
        return "Table~" + r[0]
    return "Tables~" + ", ".join(r[:-1]) + (", and~" if len(r) > 2 else " and~") + r[-1]


def is_ours(name):
    return name.startswith("\\diva{}") and "kappa" not in name and "beta" not in name and " w/ " not in name


CITE_W = {"BED-LLM": "4.8em"}  # header width of a cited method column (the citation wraps inside it)


def hcell(name, two=False, narrow=False):
    """header cell of a method column. Default the name is never split; a citation goes on a second line at
    natural width. narrow=True (tables with many method columns): the citation wraps inside a box as wide as needed for the
    name, and the name itself is still kept on one line."""
    if two and name in MCITE:
        if narrow is True:  # many method columns (per-LLM tables): name only, the citation is given in Table tab:know (F6)
            return "\\begin{tabular}[c]{@{}c@{}}" + name + "\\end{tabular}"
        if narrow == "stack":  # name / authors / (year): the name stays whole, the citation takes two short lines
            return ("\\begin{tabular}[c]{@{}c@{}}" + name + "\\\\\\citeauthor{" + MCITE[name] + "}\\\\(\\citeyear{" + MCITE[name] + "})\\end{tabular}")
        return "\\begin{tabular}[c]{@{}c@{}}" + name + "\\\\\\citep{" + MCITE[name] + "}\\end{tabular}"
    if two and name in HDR2 and narrow == "stack" and "-\\\\" not in HDR2[name]:  # split at a space only, never inside a name
        return "\\begin{tabular}[c]{@{}c@{}}" + HDR2[name] + "\\end{tabular}"
    return ("\\begin{tabular}[c]{@{}c@{}}" + name + "\\end{tabular}") if (two and narrow is True) else name  # [c] box: centred like two-line cells


CITE_W_NARROW = {"BED-LLM": "4.1em", "VoI-Ask": "3.6em"}


def mlabel(name, cite=True):
    """row label of a method in a main table: display name and citation"""
    return name + (" \\citep{" + MCITE[name] + "}" if cite and name in MCITE else "")


def method_rows_table(meths, cols, val, label, n_llm, group=None, star=True, colsep=None):
    """rows = methods (canonical order), columns = environments/domains + Avg. Each value is averaged over the fixed LLM
    set (val returns None unless every LLM is complete). Avg. = mean over the columns, printed only if all are present."""
    vals = {(m, c): val(m, c) for m, _ in meths for c, _ in cols}
    avg = {m: (float(np.mean([vals[(m, c)] for c, _ in cols])) if all(vals[(m, c)] is not None for c, _ in cols) else None) for m, _ in meths}
    colf = [fmt_best([vals[(m, c)] for m, _ in meths], scale=100) for c, _ in cols] + [fmt_best([avg[m] for m, _ in meths], scale=100)]
    lines = []
    for i, (m, name) in enumerate(meths):
        row = mlabel(name) + " & " + " & ".join(cf[i] for cf in colf) + " \\\\"
        lines.append((OURS_ROW if is_ours(name) else "") + row)
    lines = group_rows([n for _, n in meths], lines, len(cols) + 2)  # italic category rows (layout A)
    nc = len(cols)
    hdr = "\\multirow{2}{*}[-0.6ex]{\\begin{tabular}[c]{@{}l@{}}Method\\\\(Avg.\\ across " + str(n_llm) + " LLMs)\\end{tabular}}"
    if group is None:
        group = [(nc, "")]
    top, cm, k = [], [], 2
    for j, (w, g) in enumerate(group):
        last = j == len(group) - 1
        top.append("\\multicolumn{" + str(w) + "}{c" + ("|" if last else "") + "}{" + g + "}")
        cm.append("\\cmidrule(lr){" + f"{k}-{k + w - 1}" + "}"); k += w
    env = "table*" if star else "table"
    return ("\\begin{" + env + "}[t]\n\\centering\n\\small\n" + (f"\\setlength{{\\tabcolsep}}{{{colsep}}}\n" if colsep else "")
            + "\\begin{tabular}{l|" + "c" * nc + "|c}\n\\toprule\n"
            + hdr + " & " + " & ".join(top) + " & \\multirow{2}{*}[-0.6ex]{Avg.} \\\\\n" + "".join(cm) + "\n"
            + " & " + " & ".join(n for _, n in cols) + " & \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{CAPTION}\n\\label{" + label + "}\n\\end{" + env + "}\n"), vals, avg


def llm_rows_table(tags, meths, val, label, caption, na=lambda t, m: False, size="\\small", colsep="1.8pt", const=None):  # F6: 1.8pt (was 1.5pt; wider overflows)
    """rows = LLMs of the fixed set grouped by family (Family | Size columns, complete.family_rows; all rows, TBD where
    missing), columns = methods (canonical order). Best / second best per row (over the applicable methods), only when the
    row is complete."""
    from complete import family_rows, FAMILY_NOTE, FAM_HEAD

    def cells(t):
        vs = [None if na(t, m) else val(t, m) for m, _ in meths]
        appl = [v for (m, _), v in zip(meths, vs) if not na(t, m)]
        f = fmt_best(vs, scale=100) if all(v is not None for v in appl) else [PH if v is None else f"{100*v:.2f}" for v in vs]
        return [(OURS_CELL if is_ours(name) else "") + ("--" if na(t, m) else s) for (m, name), s in zip(meths, f)]
    lines = family_rows(tags, cells, len(meths) + 2)
    if const is not None:  # an LLM-independent method (Never) as one row instead of a constant column
        name, v = const
        lines.append("\\midrule\n\\multicolumn{2}{@{}l|}{" + name + "} & \\multicolumn{" + str(len(meths)) + "}{c@{}}{"
                     + (PH if v is None else f"{100*v:.2f}") + " (identical for every LLM)} \\\\")
    head = FAM_HEAD + " & " + " & ".join((OURS_CELL if is_ours(n) else "") + hcell(n, True, narrow=True) for _, n in meths) + " \\\\"
    return ("\\begin{table*}[t]\n\\centering\n" + size + "\n\\setlength{\\tabcolsep}{" + colsep + "}\n\\begin{tabular}{@{}ll|" + "c" * len(meths) + "@{}}\n\\toprule\n"
            + head + "\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{" + caption + " Baselines are cited in Table~\\ref{tab:know}. " + FAMILY_NOTE + "}\n\\label{" + label + "}\n\\end{table*}\n")


# controlled environments: methods of Table tab:main (canonical order); Reason. only in the per-LLM appendix tables
CTRL_METHS = [("bl_direct", "Direct"), ("bl_cot_need", "CoT"), ("bl_sc_cot", "SC"), ("bl_react", "ReAct"), ("bl_finrisk_ask", "Ask-P"),
              ("bl_voi_prompt", "VoI-P"), ("bl_voi_paper", "VoI-Ask"), ("bl_bed_llm", "BED-LLM"), ("bl_pal", "PAL"), ("think", "Reason."),
              ("always_stop", "Never"), ("always_top", "Always ask"), ("diva", "\\diva{}"),
              ("diva_u", "\\diva{}-U"), ("diva_dp", "\\diva{}-DP")]  # same rows as Table tab:know; DIVA-U = DIVA when reliabilities are stated
CTRL_METHS = [(k, mname(k)) for k, _ in msort(CTRL_METHS)]


def _ctrl_val(R, tag, c, d):
    c = "diva" if c == "diva_u" else c  # sigma = 0 when reliabilities are stated: DIVA-U is DIVA
    x = R.get(tag, {}).get(d, {}).get(c + "@sub", {})
    return x[KEY] if x.get("n", 0) >= 300 and x.get(KEY) is not None else None


def table_bin():
    """Table tab:main (main text): rows = methods, columns = the four controlled domains | Avg., each cell the DP regret
    on the first 300 test scenarios averaged over the fixed 20 LLMs (complete.CTRL_BL_LLMS), TBD unless all 20 are complete.
    Reason. (reasoning models only) is reported in the per-LLM appendix tables (tab:full_ctrl_llm*), written here too."""
    from complete import CTRL_BL_LLMS, REASON_CTRL
    R = json.load(open(f"{IS}/results/bin_main.json"))
    meths = [(k, n) for k, n in CTRL_METHS if k != "think"]

    def val(m, d):
        v = [_ctrl_val(R, t, m, d) for t in CTRL_BL_LLMS]
        return float(np.mean(v)) if all(x is not None for x in v) else None
    cols = [(d, ENVN[d]) for d in DOM_ORDER]
    tex, vals, avg = method_rows_table(meths, cols, val, "tab:main", len(CTRL_BL_LLMS), group=[(4, "Controlled environments")], star=True)
    # tab:main is merged into tab:know (main_know.tex, _merged_table); main_bin.tex is no longer written
    # ---- appendix: per-LLM tables (domain average, and one per domain), all methods incl. Reason.
    na = lambda t, m: m == "think" and t not in REASON_CTRL

    def v_avg(t, m):
        v = [_ctrl_val(R, t, m, d) for d in DOM_ORDER]
        return float(np.mean(v)) if all(x is not None for x in v) else None
    # per-LLM tables: DIVA-U is omitted (identical to DIVA when reliabilities are stated); 16 method columns need \footnotesize
    PM = [(k, n) for k, n in CTRL_METHS if k not in ("diva_u", "always_stop")]
    FS = "\\small"

    def never(d=None):  # never acquiring is LLM-independent
        ds = DOM_ORDER if d is None else [d]
        v = [_ctrl_val(R, t, "always_stop", x) for x in ds for t in CTRL_BL_LLMS]
        return float(np.mean([_ctrl_val(R, CTRL_BL_LLMS[0], "always_stop", x) for x in ds])) if all(x is not None for x in v) else None
    out = [llm_rows_table(CTRL_BL_LLMS, PM, v_avg, "tab:full_ctrl_llm",
                          "\\textbf{DP regret per LLM} on controlled environments, averaged over the four domains, first 300 test scenarios per domain. Reason.: Native reasoning (reasoning models only).", na, size=FS, const=("Never", never()))]
    for d in DOM_ORDER:
        out.append(llm_rows_table(CTRL_BL_LLMS, PM, lambda t, m, d=d: _ctrl_val(R, t, m, d), "tab:full_ctrl_llm_" + d,
                                  "\\textbf{DP regret per LLM: " + ENVN[d] + ".} Controlled, first 300 test scenarios.", na, size=FS, const=("Never", never(d))))
    open(f"{OUT}/full_ctrl_llm.tex", "w").write("\n".join(out))
    open(f"{OUT}/full_ctrl_llm_refs.tex", "w").write(
        reflist(["tab:full_ctrl_llm"]) + " reports every method of Table~\\ref{tab:know} (controlled environments) and native reasoning (Reason.) for each of the "
        + str(len(CTRL_BL_LLMS)) + " LLMs, averaged over the four domains, and " + reflist(["tab:full_ctrl_llm_" + d for d in DOM_ORDER])
        + " report each domain. \\diva{}-U equals \\diva{} when reliabilities are stated and is not repeated, and never acquiring, which does not depend on the LLM, is reported in one row.\n")
    return {"pending_main": sum(v is None for v in vals.values())}


def table_know():
    T = json.load(open(f"{IS}/results/tab_main.json")); S = json.load(open(f"{IS}/results/real_main.json"))
    envs = [("tab", "taiwan", "Taiwan"), ("tab", "german", "German"), ("tab", "cdc", "CDC"), ("tab", "heart", "Heart"),
            ("real", "ret_1", "Ret-1d"), ("real", "ret_5", "Ret-5d"), ("real", "vol_5", "Vol-5d"), ("real", "vol_20", "Vol-20d")]
    meths = [("direct", "Direct"), ("fac_eig", "Fac-EIG"), ("always_stop", "Never"), ("diva_noshrink", "\\diva{} ($\\lambda{=}1$)"),
             ("diva_robust", "\\diva{}-Robust"), ("diva", "\\diva{}")]
    rows = {m: [] for m, _ in meths}; wins = {m: 0 for m, _ in meths}; cells = 0
    for kind, e, _ in envs:
        src = T if kind == "tab" else S
        tags = [t for t in src if e in src[t] and "diva" in src[t][e] and src[t][e]["diva"]["n"] >= 200]
        for m, _ in meths:
            v = [src[t][e][m][KEY] for t in tags if m in src[t][e]]
            rows[m].append(float(np.mean(v)) if v else None)
        for t in tags:
            cells += 1
            for m, _ in meths:
                if m != "diva" and m in src[t][e] and src[t][e]["diva"][KEY] < src[t][e][m][KEY]:
                    wins[m] += 1
    lines = []
    cols_vals = list(zip(*[rows[m] for m, _ in meths]))
    best = []
    for j in range(len(envs)):
        best.append(fmt_row([rows[m][j] for m, _ in meths]))
    for i, (m, name) in enumerate(meths):
        lines.append(name + " & " + " & ".join(best[j][i] for j in range(len(envs))) + " \\\\")
    head = "Method & " + " & ".join(n for _, _, n in envs) + " \\\\"
    tex = ("\\begin{table*}[t]\n\\centering\n\\small\n\\begin{tabular}{l" + "c" * len(envs) + "}\n\\toprule\n"
           " & \\multicolumn{2}{c}{Credit} & \\multicolumn{2}{c}{Clinical} & \\multicolumn{4}{c}{SEC filings (2025--2026)} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-9}\n"
           + head + "\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
           "\\caption{\\textbf{Results on knowledge-required environments.} Regret ($\\times 100$) against the DP-optimal policy of a reference model fitted on real records, averaged over LLMs. Reliabilities are never stated, so they must come from the LLM's knowledge. The best results are in bold and the second best are underlined.}\n\\label{tab:know}\n\\end{table*}\n")
    open(f"{OUT}/main_know.tex", "w").write(tex)
    return rows, wins, cells




def _tl(a, b):
    """two-line label cell, vertically centered in its row"""
    return "\\begin{tabular}[c]{@{}l@{}}" + a + "\\\\" + b + "\\end{tabular}"


def table_ablation():
    """Main Table tab:ablation: a cumulative build-up from the direct choice to DIVA-DP on the
    controlled probe subset, every row adding exactly one module to the row above, then DIVA with its decision value replaced
    by EIG and the true-specification reference. Same scenarios (data/probe_subset), same metric (KEY), same 20 LLMs
    (complete.KNOW_LLMS); a row is TBD unless every LLM covers every scenario.
      direct    the LLM's direct choice (probe record)
      raw       decision value computed from the LLM's raw imagined beliefs (11-option scale, probe record)
      bins      decision value on the elicited specification, 11-option answers (outputs/abl/facbin)
      diva      the same with numeric answers (= DIVA, outputs/test_bin/facnum)
      diva_dp   DP lookahead on the same specification (= DIVA-DP)
      fac_eig   EIG instead of the decision value on the DIVA specification (fixed 0.05-bit threshold)
      true_spec decision value with the true likelihoods (myopic)"""
    sys.path.insert(0, f"{IS}/scripts")
    from evalx import read, fac_scores, choose_max, score, summarize
    from metrics import derive
    from dcore import dp_value
    from complete import KNOW_LLMS
    sub = {s["id"]: s for s in read(f"{IS}/data/probe_subset.jsonl.gz")}
    keys = ["direct", "raw", "bins", "diva", "diva_dp", "fac_eig", "true_spec"]
    per = {k: [] for k in keys}
    complete_all = True
    for tag in KNOW_LLMS:
        fac = {}
        for d in DOMS:
            p = f"{IS}/outputs/test_bin/facnum_{tag}_{d}.jsonl.gz"
            if os.path.exists(p):
                fac.update({r["id"]: r for r in read(p) if r["id"] in sub})
        fb = {r["id"]: r for r in read(f"{IS}/outputs/abl/facbin_{tag}.jsonl.gz")} if os.path.exists(f"{IS}/outputs/abl/facbin_{tag}.jsonl.gz") else {}
        pr = {r["id"]: r for r in read(f"{IS}/outputs/probe/probe_{tag}.jsonl.gz")} if os.path.exists(f"{IS}/outputs/probe/probe_{tag}.jsonl.gz") else {}
        ids = [i for i in sub if i in fac and i in fb and i in pr]
        if len(ids) < len(sub):
            complete_all = False
            continue
        R = {k: [] for k in keys}
        for i in ids:
            sc = sub[i]; fs = fac_scores(sc, fac[i])
            dd = derive(sc, pr[i])
            R["direct"].append(score(sc, dd["direct"][0] if "direct" in dd else "S"))
            R["raw"].append(score(sc, dd["iovoi"][0] if "iovoi" in dd else "S"))
            R["bins"].append(score(sc, choose_max(fac_scores(sc, fb[i])["net"])))
            R["diva"].append(score(sc, choose_max(fs["net"])))
            kk = list(fs["par"]); v, a = dp_value(fs["b"], [fs["par"][k] for k in kk], float(sc["g"]), float(sc["l"]))
            R["diva_dp"].append(score(sc, "S" if a == "S" else kk[a]))
            R["fac_eig"].append(score(sc, choose_max(fs["eig"], 0.05)))
            tf = {"prior": sc["prior"], "obs": [[o["s1"], o["s0"]] for o in sc.get("observed", [])], "chk": {c["label"]: [c["s1"], c["s0"]] for c in sc["checks"]}}
            R["true_spec"].append(score(sc, choose_max(fac_scores(sc, tf)["net"])))
        for k in keys:
            per[k].append(summarize(R[k])[KEY])
    val = {k: (100 * float(np.mean(per[k])) if complete_all else None) for k in keys}
    sgn = lambda d: ("+" if d >= 0 else "$-$") + f"{abs(round(d + 0.0, 2)):.2f}"
    val = {k: (round(v, 2) if v is not None else None) for k, v in val.items()}  # Delta from the displayed (rounded) values (fact-check 10-12)
    build = [("direct", "Direct"), ("raw", "+ Decision value on imagined beliefs"), ("bins", "+ Elicited specification (11-option)"),
             ("diva", "+ Numeric answers (= \\diva{})"), ("diva_dp", "+ DP lookahead (= \\diva{}-DP)")]
    lines, prev = [], None
    fb = dict(zip([k for k, _ in build] + ["fac_eig"], fmt_best([val[k] / 100 if val[k] is not None else None for k, _ in build] + [val["fac_eig"] / 100 if val["fac_eig"] is not None else None], scale=100)))
    for k, n in build:
        v = val[k]
        cell = PH if v is None else fb[k]
        dl = "--" if prev is None or v is None or val[prev] is None else sgn(v - val[prev])
        lines.append((OURS_ROW if k in ("diva", "diva_dp") else "") + n + " & " + cell + " & " + dl + " \\\\")
        prev = k
    lines.append("\\midrule")
    for k, n in [("fac_eig", _tl("\\diva{} $-$ decision value", "(EIG on the same specification)"))]:
        v = val[k]
        lines.append(n + " & " + (PH if v is None else fb[k]) + " & " + ("--" if v is None or val["diva"] is None else sgn(v - val["diva"])) + " \\\\")
    lines.append("\\midrule")
    v = val["true_spec"]
    lines.append("Reference: True specification & " + (PH if v is None else f"{v:.2f}") + " & " + ("--" if v is None or val["diva"] is None else sgn(v - val["diva"])) + " \\\\")
    tex = ("\\begin{table}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{l|cc}\n\\toprule\n"
           "Variant & \\begin{tabular}[c]{@{}c@{}}Regret\\\\($\\times$100)\\end{tabular} & $\\Delta$ \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
           "\\caption{CAPTION}\n\\label{tab:ablation}\n\\end{table}\n")
    open(f"{OUT}/ablation.tex", "w").write(tex)
    return {k: val[k] for k in keys}





SHOW_GENERIC = False
# Table 3 / tab:know_myo: a cell is shown only when the method has been run for EVERY LLM of the fixed set on that env;
# otherwise a placeholder is printed.
from complete import KNOW_LLMS
PH = "\\textcolor{gray}{TBD}"
KNOW_ENVS = [("tab:taiwan", "Taiwan"), ("tab:german", "German"), ("tab:cdc", "CDC"), ("tab:heart", "Heart"),
             ("sec:ret_1", "Ret-1d"), ("sec:ret_5", "Ret-5d"), ("sec:vol_5", "Vol-5d"), ("sec:vol_20", "Vol-20d")]
KNOW_METHS = [("direct", "Direct"), ("cot_need", "CoT \\citep{wei2022chain}"), ("sc", "SC \\citep{wang2023selfconsistency}"),
              ("react", "ReAct \\citep{yao2023react}"), ("ask_p", "Ask-P"), ("voi_prompt", "VoI-P"),
              ("voi_ask", "VoI-Ask \\citep{dong2026value}"), ("bed_llm", "BED-LLM \\citep{choudhury2026bedllm}"),
              ("pal", "PAL \\citep{gao2023pal}"), ("never", "Never"),
              ("always_top", "Always ask"), ("diva", "\\diva{}"), ("diva_u", "\\diva{}-U"), ("diva_dp", "\\diva{}-DP")]
KNOW_METHS = [(k, mlabel(mname(k))) for k, _ in msort(KNOW_METHS)]  # canonical order and names, citation after the name


def _know_table(R, field, label, fname):
    ne = len(KNOW_ENVS)
    vals = {}
    for m, _ in KNOW_METHS:
        for e, _ in KNOW_ENVS:
            recs = R.get(e, {})
            ok = all(t in recs and f"s300:{m}" in recs[t] for t in KNOW_LLMS)
            vals[(m, e)] = float(np.mean([recs[t][f"s300:{m}"][field] for t in KNOW_LLMS])) if ok else None
    avg = {m: (float(np.mean([vals[(m, e)] for e, _ in KNOW_ENVS])) if all(vals[(m, e)] is not None for e, _ in KNOW_ENVS) else None)
           for m, _ in KNOW_METHS}
    cols = [fmt_row([vals[(m, e)] for m, _ in KNOW_METHS]) for e, _ in KNOW_ENVS] + [fmt_row([avg[m] for m, _ in KNOW_METHS])]
    lines = []
    for i, (m, name) in enumerate(KNOW_METHS):
        cells = [c[i] if c[i] != "--" else PH for c in cols]
        row = name + " & " + " & ".join(cells[:ne]) + " & " + cells[ne] + " \\\\"
        lines.append(("\\rowcolor{oursrow} " if m.startswith("diva") else "") + row)
    lines = group_rows([mname(m) for m, _ in KNOW_METHS], lines, ne + 2)  # italic category rows (layout A)
    if field == KEY:
        wins = []
        others = [m for m, _ in KNOW_METHS if not m.startswith("diva")]
        for e, _ in KNOW_ENVS:
            recs = R.get(e, {})
            if all(vals[(m, e)] is not None for m in others + ["diva_u"]):
                w = sum(recs[t]["s300:diva_u"][field] < min(recs[t][f"s300:{m}"][field] for m in others) for t in KNOW_LLMS)
                wins.append(f"{w}/{len(KNOW_LLMS)}")
            else:
                wins.append(PH)
        lines.append("\\midrule\nWins of \\diva{}-U & " + " & ".join(wins) + " & -- \\\\")
    tex = ("\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4.5pt}\n\\begin{tabular}{l|" + "c" * ne + "|c}\n\\toprule\n"
           "\\multirow{2}{*}[-0.6ex]{\\begin{tabular}[c]{@{}l@{}}Method\\\\(Avg.\\ across " + str(len(KNOW_LLMS)) + " LLMs)\\end{tabular}} & \\multicolumn{2}{c}{Credit} & \\multicolumn{2}{c}{Clinical} & \\multicolumn{4}{c|}{SEC filings (2025--2026)} & \\multirow{2}{*}[-0.6ex]{Avg.} \\\\\n"
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-9}\n"
           " & " + " & ".join(n for _, n in KNOW_ENVS) + " & \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
           "\\caption{CAPTION}\n\\label{" + label + "}\n\\end{table*}\n")
    open(f"{OUT}/{fname}", "w").write(tex)
    return sum(v is None for v in vals.values())


MERGED_AVG = False  # both Avg. columns or neither; both dropped as the table is otherwise too wide
MERGED_COLSEP = "2pt"
EXTERNAL_GROUPS = ("Prompting", "Decision-aware prompting", "Information gain", "Program-aided")


def _two_line(name):
    """method label with its citation on a second line (keeps the merged table within the text width)"""
    if name in MCITE:
        return "\\begin{tabular}[c]{@{}l@{}}" + name + "\\\\\\citep{" + MCITE[name] + "}\\end{tabular}"
    return name


def _merged_table(R):
    """Main-text Table tab:know: ONE table* with shared method rows; columns = the eight
    knowledge-required environments (| Avg.) | the four controlled domains (| Avg.). Cells: first-decision DP regret (x100)
    on the first 300 test scenarios averaged over the same 20 LLMs, TBD unless all 20 are complete. Wins of DIVA-U is
    counted against the external baselines (Prompting ... Program-aided) per LLM."""
    from complete import CTRL_BL_LLMS, METHOD_GROUP
    B = json.load(open(f"{IS}/results/bin_main.json"))
    ckey = {mname(k): k for k, _ in CTRL_METHS}
    GEN = json.load(open(f"{IS}/results/generic_main.json")) if os.path.exists(f"{IS}/results/generic_main.json") else {}
    GK = {"const_diva": "diva", "const_diva_u": "diva_u"}
    meths = [m for m, _ in msort([(k, None) for k, _ in KNOW_METHS] + [(k, None) for k in GK])]
    names = [mname(m) for m in meths]
    kv = {}
    for m in meths:
        if m in GK:
            for e, _ in KNOW_ENVS:
                kv[(m, e)] = GEN.get("know_first", {}).get(GK[m], {}).get(e.split(":")[1])
            continue
        for e, _ in KNOW_ENVS:
            recs = R.get(e, {})
            ok = all(t in recs and f"s300:{m}" in recs[t] for t in KNOW_LLMS)
            kv[(m, e)] = float(np.mean([recs[t][f"s300:{m}"][KEY] for t in KNOW_LLMS])) if ok else None
    cv = {}
    for m, n in zip(meths, names):
        if m in GK:
            for d in DOM_ORDER:
                cv[(m, d)] = GEN.get("ctrl_first", {}).get(GK[m], {}).get(d)
            continue
        c = ckey.get(n)
        for d in DOM_ORDER:
            v = [_ctrl_val(B, t, c, d) for t in CTRL_BL_LLMS] if c else [None]
            cv[(m, d)] = float(np.mean(v)) if all(x is not None for x in v) else None
    kper = lambda t, m, e: R[e][t][f"s300:{m}"][KEY]
    cper = lambda t, m, d: _ctrl_val(B, t, "diva" if m == "diva_u" else ckey[mname(m)], d)
    # first-decision group means of the strongest external baseline and our variants for Table tab:variants (scripts/make_table_variants.py)
    gm = lambda xs: float(np.mean(xs)) if all(x is not None for x in xs) else None
    json.dump({m: {"know": gm([kv[(m, e)] for e, _ in KNOW_ENVS]), "ctrl": gm([cv[(m, d)] for d in DOM_ORDER])}
               for m in ("voi_ask", "diva", "diva_u", "diva_dp") if m in meths}, open(f"{IS}/results/variants_first.json", "w"), indent=1)
    # all per-environment means (incl. the LLM-free generic rows) for Table tab:generic (scripts/make_table_generic.py)
    from complete import METHOD_GROUP as _MG
    json.dump({"know": {m: {e.split(":")[1]: kv[(m, e)] for e, _ in KNOW_ENVS} for m in meths},
               "ctrl": {m: {d: cv[(m, d)] for d in DOM_ORDER} for m in meths},
               "group": {m: _MG.get(n) for m, n in zip(meths, names)}}, open(f"{IS}/results/know_table_means.json", "w"), indent=1)
    keep = [i for i, m in enumerate(meths) if m not in ("diva_dp", "const_diva", "const_diva_u")]
    meths = [meths[i] for i in keep]; names = [names[i] for i in keep]
    return merged_method_table(meths, names, kv, cv, kper, cper, KNOW_LLMS, CTRL_BL_LLMS, "tab:know", "main_know.tex")


def merge_diva_rows(names, colf, cidx, pair=("\\diva{}", "\\diva{}-U")):
    """where DIVA-U equals DIVA (controlled columns cidx), print the value once, vertically centered over the two
    adjacent rows; the multirow sits in the lower row with a negative span so the row colour of that row does not cover it"""
    try:
        i = names.index(pair[0]); j = names.index(pair[1])
    except ValueError:
        return colf
    if j != i + 1:
        return colf
    for c in cidx:
        if colf[c][i] == colf[c][j]:
            colf[c][j] = "\\multirow{-2}{*}{" + colf[c][i] + "}"; colf[c][i] = ""
    return colf


def merged_method_table(meths, names, kv, cv, kper, cper, kllms, cllms, label, fname):
    """render a merged knowledge-required | controlled method table (layout of Table tab:know; user rules 2026-10-11).
    kv[(m, e)] / cv[(m, d)]: values averaged over the LLM sets (None = TBD); kper(t, m, e) / cper(t, m, d): per-LLM values for
    the Wins of DIVA-U row, counted against the external baselines (Prompting ... Program-aided)."""
    from complete import METHOD_GROUP
    mean = lambda xs: float(np.mean(xs)) if all(x is not None for x in xs) else None
    cols = [[kv[(m, e)] for m in meths] for e, _ in KNOW_ENVS]
    if MERGED_AVG:
        cols.append([mean([kv[(m, e)] for e, _ in KNOW_ENVS]) for m in meths])
    cols += [[cv[(m, d)] for m in meths] for d in DOM_ORDER]
    if MERGED_AVG:
        cols.append([mean([cv[(m, d)] for d in DOM_ORDER]) for m in meths])
    colf = [fmt_best(c, scale=100) for c in cols]
    nk = len(KNOW_ENVS) + (1 if MERGED_AVG else 0)
    nc = len(DOM_ORDER) + (1 if MERGED_AVG else 0)
    colf = merge_diva_rows(list(names), colf, range(nk, nk + nc))
    colf = merge_diva_rows(list(names), colf, range(nk, nk + nc), ("Generic \\diva{}", "Generic \\diva{}-U"))
    lines = []
    for i, (m, n) in enumerate(zip(meths, names)):
        lines.append((OURS_ROW if is_ours(n) else "") + _two_line(n) + " & " + " & ".join(cf[i] for cf in colf) + " \\\\")
    lines = group_rows(names, lines, 1 + nk + nc)
    # group header rows: keep the vertical rule between the two column groups
    lines = [l.replace(" &" * (nk + nc) + " \\\\", " &" * (nk - 1) + " & \\multicolumn{1}{c|}{}" + " &" * nc + " \\\\") if l.startswith("\\multicolumn{1}{l|}{\\textit") else l
             for l in lines]
    ext = [m for m, n in zip(meths, names) if METHOD_GROUP.get(n) in EXTERNAL_GROUPS]
    wins = []
    for e, _ in KNOW_ENVS:
        if all(kv[(m, e)] is not None for m in ext + ["diva_u"]):
            w = sum(kper(t, "diva_u", e) < min(kper(t, m, e) for m in ext) for t in kllms)
            wins.append(f"{w}/{len(kllms)}")
        else:
            wins.append(PH)
    if MERGED_AVG:
        wins.append("--")
    for d in DOM_ORDER:
        if all(cv[(m, d)] is not None for m in ext + ["diva_u"]):
            w = sum(cper(t, "diva_u", d) < min(cper(t, m, d) for m in ext) for t in cllms)
            wins.append(f"{w}/{len(cllms)}")
        else:
            wins.append(PH)
    if MERGED_AVG:
        wins.append("--")
    lines.append("\\midrule\nWins of \\diva{}-U & " + " & ".join(wins) + " \\\\")
    sp = f"{IS}/results/sig_wins.json"
    if label == "tab:know" and os.path.exists(sp):  # paired clustered-bootstrap significance of the wins (scripts/sig_wins.py)
        cnt = json.load(open(sp)).get("counts", {})
        cell = lambda k, nl: (f"{cnt[k]['sig']}/{nl}" if cnt.get(k, {}).get("n_llm") == nl else PH)
        sig = [cell(e, len(kllms)) for e, _ in KNOW_ENVS] + (["--"] if MERGED_AVG else [])
        sig += [cell(d, len(cllms)) for d in DOM_ORDER] + (["--"] if MERGED_AVG else [])
        hdr = " & ".join([n for _, n in KNOW_ENVS] + [ENVN[d] for d in DOM_ORDER])
        nk_, nc_ = len(KNOW_ENVS), len(DOM_ORDER)
        st = ("\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{l|" + "c" * nk_ + "|" + "c" * nc_ + "}\n\\toprule\n"
              "\\multirow{2}{*}[-0.7ex]{Count} & \\multicolumn{" + str(nk_) + "}{c|}{Knowledge-required environments} & \\multicolumn{" + str(nc_) + "}{c}{Controlled environments} \\\\\n"
              "\\cmidrule(lr){2-" + str(1 + nk_) + "}\\cmidrule(lr){" + str(2 + nk_) + "-" + str(1 + nk_ + nc_) + "}\n"
              " & " + hdr + " \\\\\n\\midrule\n"
              "Wins & " + " & ".join(w for w in wins if w != "--") + " \\\\\n"
              "Significant wins & " + " & ".join(x for x in sig if x != "--") + " \\\\\n\\bottomrule\n\\end{tabular}\n"
              "\\caption{\\textbf{Significance of \\diva{}-U's wins} in Table~\\ref{tab:know}. Wins: LLMs where \\diva{}-U beats every external baseline. "
              "Significant wins: LLMs where the 95\\% clustered bootstrap CI of the gap to the best external baseline is below zero.}\n\\label{tab:sig_wins}\n\\end{table*}\n")
        open(f"{OUT}/sig_wins.tex", "w").write(st)
    ka = " & \\multirow{2}{*}[-0.6ex]{Avg.}" if MERGED_AVG else ""
    ca = " & \\multirow{2}{*}[-0.6ex]{Avg.}" if MERGED_AVG else ""
    # Avg. of the knowledge group carries the rule between the groups
    if MERGED_AVG:
        ka = " & \\multicolumn{1}{c|}{\\multirow{2}{*}[-0.6ex]{Avg.}}"
        sec = "\\multicolumn{4}{c}{SEC filings (2025--2026)}"
        kav = " & \\multicolumn{1}{c|}{}"
    else:
        sec = "\\multicolumn{4}{c|}{SEC filings (2025--2026)}"
        kav = ""
    ctl = " & ".join("\\multirow{2}{*}[-0.6ex]{" + ENVN[d] + "}" for d in DOM_ORDER)
    knames = [n for _, n in KNOW_ENVS]
    if not MERGED_AVG:
        knames[-1] = "\\multicolumn{1}{c|}{" + knames[-1] + "}"
    spec = "l|" + "c" * nk + "|" + "c" * nc
    tex = ("% merged " + label + ": knowledge-required | controlled environments\n\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{" + MERGED_COLSEP + "}\n\\begin{tabular}{" + spec + "}\n\\toprule\n"
           "\\multirow{3}{*}[-1.4ex]{\\begin{tabular}[c]{@{}l@{}}Method\\\\(Avg.\\ across " + str(len(kllms)) + " LLMs)\\end{tabular}} & \\multicolumn{" + str(nk) + "}{c|}{Knowledge-required environments} & \\multicolumn{" + str(nc) + "}{c}{Controlled environments} \\\\\n"
           "\\cmidrule(lr){2-" + str(1 + nk) + "}\\cmidrule(lr){" + str(2 + nk) + "-" + str(1 + nk + nc) + "}\n"
           " & \\multicolumn{2}{c}{Credit} & \\multicolumn{2}{c}{Clinical} & " + sec + ka + " & " + ctl + ca + " \\\\\n"
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-9}\n"
           " & " + " & ".join(knames) + kav + " & " + " &" * (nc - 1) + " \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
           "\\caption{CAPTION}\n\\label{" + label + "}\n\\end{table*}\n")
    open(f"{OUT}/{fname}", "w").write(tex)
    return sum(v is None for v in list(kv.values()) + list(cv.values()))


def table_know2():
    R = json.load(open(f"{IS}/results/know_main.json"))
    p1 = _merged_table(R)  # Table tab:know: knowledge-required | controlled environments, shared method rows
    p2 = _know_table(R, "norm_regret", "tab:know_myo", "main_know_myo.tex")
    # ---- appendix (Full Results): every Table-3 method for every LLM of the fixed set, one table per environment and optimum
    meths = [(m, mname(m)) for m, _ in KNOW_METHS if m != "never"]  # never acquiring is LLM-independent: one row

    def never_k(e, field):
        recs = [r["s300:never"] for r in R.get(e, {}).values() if "s300:never" in r]
        if not recs or not all(t in R.get(e, {}) and "s300:never" in R[e][t] for t in KNOW_LLMS):
            return None
        return max(recs, key=lambda x: x.get("n", 0))[field]
    out = []
    for field, tag, opt in [(KEY, "", "DP"), ("norm_regret", "_myo", "myopic")]:
        for e, en in KNOW_ENVS:
            def val(t, m, e=e, field=field):
                r = R.get(e, {}).get(t, {}).get(f"s300:{m}")
                return r[field] if r and r.get(field) is not None else None
            out.append(llm_rows_table(KNOW_LLMS, meths, val, "tab:full_know_llm" + tag + "_" + e.split(":")[1],
                                      "\\textbf{" + ("Myopic" if tag else "DP") + " regret per LLM: " + en + ".} Same first 300 test scenarios as Table~\\ref{tab:know" + ("_myo" if tag else "") + "}.", const=("Never", never_k(e, field))))
    n_dp = len(KNOW_ENVS)
    open(f"{OUT}/full_know_llm.tex", "w").write("\n".join(out[:n_dp]))
    os.makedirs(f"{IS}/paper/parked", exist_ok=True)
    open(f"{IS}/paper/parked/full_know_llm_myo.tex", "w").write("\n".join(out[n_dp:]))
    open(f"{OUT}/full_know_llm_refs.tex", "w").write(
        reflist(["tab:full_know_llm_" + e.split(":")[1] for e, _ in KNOW_ENVS]) + " report the DP regret of every method of Table~\\ref{tab:know} for each of the "
        + str(len(KNOW_LLMS)) + " LLMs, one table per environment. Never acquiring does not depend on the LLM and is reported in one row.\n")
    return {"pending_cells_dp": p1, "pending_cells_myo": p2}

def table_abl_eig():
    """Appendix ablation tab:abl_eig: DIVA with its decision-value criterion replaced by EIG on the same elicited specification,
    with a fixed threshold (no cost term) or with the cost trade-off and threshold tuned on labelled development scenarios (‡).
    Rows = variants, columns = the eight knowledge-required environments | Avg., DP regret (x100) on the full test sets averaged
    over the fixed 20 LLMs (TBD unless every LLM is complete)."""
    R = json.load(open(f"{IS}/results/know_main.json"))
    meths = [("fac_eig", "\\diva{} w/ EIG (Fixed threshold)"), ("fac_eig_cost", "\\diva{} w/ EIG$^\\ddagger$ (Tuned on labels)"),
             ("diva", "\\diva{}"), ("diva_u", "\\diva{}-U")]

    def val(m, e):
        recs = R.get(e, {})
        if not all(t in recs and f"full:{m}" in recs[t] for t in KNOW_LLMS):
            return None
        return float(np.mean([recs[t][f"full:{m}"][KEY] for t in KNOW_LLMS]))
    ne = len(KNOW_ENVS)
    vals = {(m, e): val(m, e) for m, _ in meths for e, _ in KNOW_ENVS}
    avg = {m: (float(np.mean([vals[(m, e)] for e, _ in KNOW_ENVS])) if all(vals[(m, e)] is not None for e, _ in KNOW_ENVS) else None)
           for m, _ in meths}
    cols = [fmt_row([vals[(m, e)] for m, _ in meths]) for e, _ in KNOW_ENVS] + [fmt_row([avg[m] for m, _ in meths])]
    lines = []
    for i, (m, name) in enumerate(meths):
        cells = [c[i] if c[i] != "--" else PH for c in cols]
        if i == 2:
            lines.append("\\midrule")
        lines.append((OURS_ROW if is_ours(name) else "") + name + " & " + " & ".join(cells[:ne]) + " & " + cells[ne] + " \\\\")
    tex = ("\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4.5pt}\n\\begin{tabular}{l|" + "c" * ne + "|c}\n\\toprule\n"
           "\\multirow{2}{*}[-0.6ex]{\\begin{tabular}[c]{@{}l@{}}Variant\\\\(Avg.\\ across " + str(len(KNOW_LLMS)) + " LLMs)\\end{tabular}} & \\multicolumn{2}{c}{Credit} & \\multicolumn{2}{c}{Clinical} & \\multicolumn{4}{c|}{SEC filings (2025--2026)} & \\multirow{2}{*}[-0.6ex]{Avg.} \\\\\n"
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-9}\n"
           " & " + " & ".join(n for _, n in KNOW_ENVS) + " & \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
           "\\caption{CAPTION}\n\\label{tab:abl_eig}\n\\end{table*}\n")
    open(f"{OUT}/abl_eig.tex", "w").write(tex)
    return {m: avg[m] for m, _ in meths}, {k: v for k, v in vals.items()}


if __name__ == "__main__":
    for f in sys.argv[1:] or ["bin", "know"]:
        print(f, globals()["table_" + f]())
