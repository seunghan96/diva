"""Closest prior work as baselines (reviewer gap M4): DeLLMa (Liu et al., 2025) and UoT (Hu et al., 2024), run by
baselines/run_prior.py (jobs/f3_prior.tsv), scored as Table tab:know (first-decision DP regret, first 300 test scenarios, the
same 20 LLMs), next to Direct, VoI-Ask, BED-LLM, Never, DIVA and DIVA-U on IDENTICAL rows per (LLM, environment).
Writes results/prior_baselines.json and paper/tables/prior_baselines.tex (gray TBD until all 20 LLMs are complete)."""
import os, sys, json, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read, score, summarize, fac_scores, choose_max
from complete import KNOW_LLMS, ENV_ORDER, DOM_ORDER, ENVN, PH

KEY = "dp_regret"
NEW = ["dellma", "uot", "voi_tool"]
# (key, display name, group); canonical order of Table tab:know with the two prior-work methods slotted into their groups
ROWS = [("direct", "Direct", "Prompting"),
        ("voi_ask", "VoI-Ask \\citep{dong2026value}", "Decision-aware prompting"),
        ("dellma", "DeLLMa \\citep{liu2025dellma}", "Elicited model"),
        ("bed_llm", "BED-LLM \\citep{choudhury2026bedllm}", "Information gain"),
        ("uot", "UoT \\citep{hu2024uncertainty}", "Information gain"),
        ("voi_tool", "VoI-Tool", "Program-aided"),  # the LLM fills in a provided VOI calculator (baselines/voi_tool.py)
        ("never", "Never", "Reference"),
        ("diva", "\\diva{}", "Ours"), ("diva_u", "\\diva{}-U", "Ours")]
NMIN = 285  # >= 95% of the first 300 rows


def new_choices(tag, env):
    out = {}
    for m in NEW:
        p = f"{IS}/baselines/outputs/{tag}/{m}/{env}_test.jsonl.gz"
        out[m] = {r["id"]: r["choice"] for r in read(p) if r.get("choice") is not None and "error" not in r}
    return out


def know_cell(tag, env, envrows):
    p = f"{IS}/results/know_choices/{env}_{tag}.json"
    if not os.path.exists(p):
        return None
    C = json.load(open(p))
    nc = new_choices(tag, env)
    rows = {}
    for i, ch in C.items():
        ch = dict(ch)
        for m in NEW:
            if i in nc[m]:
                ch[m] = nc[m][i]
        rows[i] = ch
    return rows


def ctrl_cell(tag, d, env, first):
    rows = collections.defaultdict(dict)
    for r in read(f"{IS}/outputs/test_bin/facnum_{tag}_{d}.jsonl.gz"):
        sc = env.get(r["id"])
        if sc is None or r["id"] not in first:
            continue
        fs = fac_scores(sc, r)
        rows[r["id"]]["diva"] = rows[r["id"]]["diva_u"] = choose_max(fs["net"])  # DIVA-U = DIVA when reliabilities are stated
    # Direct: the same baseline run as Table tab:know (baselines/outputs/<tag>/direct, stop option rotated), fact-check 10-12
    for m, k in (("direct", "direct"), ("bed_llm", "bed_llm"), ("voi_paper", "voi_ask")):
        for r in read(f"{IS}/baselines/outputs/{tag}/{m}/{d}_test.jsonl.gz"):
            if r.get("choice") is not None and r["id"] in first:
                rows[r["id"]][k] = r["choice"]
    for m, v in new_choices(tag, d).items():
        for i, c in v.items():
            if i in first:
                rows[i][m] = c
    for i in first:
        rows[i]["never"] = "S"
    return rows


OPTIONAL = {"voi_tool"}  # added later (reviewer request); its cell is TBD until complete, without blanking the other rows


def summarize_cell(rows, env):
    keys = [k for k, _, _ in ROWS if k not in OPTIONAL]
    ids = [i for i, ch in rows.items() if all(k in ch for k in keys)]
    if len(ids) < NMIN:
        return None
    out = {k: summarize([score(env[i], rows[i][k]) for i in ids]) for k in keys} | {"_n": len(ids)}
    for k in OPTIONAL:  # identical rows: only when the optional method covers every row used for the others
        if all(k in rows[i] for i in ids):
            out[k] = summarize([score(env[i], rows[i][k]) for i in ids])
    return out


def main():
    res = {}
    for e in ENV_ORDER:
        base = "envs_tab" if e in ("taiwan", "german", "cdc", "heart") else "envs_real"
        envl = read(f"{IS}/{base}/data/{e}_test.jsonl.gz"); env = {s["id"]: s for s in envl}
        first = [s["id"] for s in envl[:300]]
        for t in KNOW_LLMS:
            rows = know_cell(t, e, env)
            if rows is not None:
                rows = {i: rows[i] for i in first if i in rows}
                c = summarize_cell(rows, env)
                if c:
                    res.setdefault(e, {})[t] = c
    for d in DOM_ORDER:
        envl = read(f"{IS}/envs_synth/data/binary/{d}_test.jsonl.gz"); env = {s["id"]: s for s in envl}
        first = set(s["id"] for s in envl[:300])
        for t in KNOW_LLMS:
            c = summarize_cell(ctrl_cell(t, d, env, first), env)
            if c:
                res.setdefault(d, {})[t] = c
    json.dump(res, open(f"{IS}/results/prior_baselines.json", "w"), indent=1)
    voi_tool_stats(res)
    table(res)
    return res


def table(res):
    cols = ENV_ORDER + DOM_ORDER
    V = {}
    for k, _, _ in ROWS:
        for e in cols:
            recs = res.get(e, {})
            ok = all(t in recs and k in recs[t] for t in KNOW_LLMS)
            V[(k, e)] = float(np.mean([recs[t][k][KEY] for t in KNOW_LLMS])) if ok else None
    cell = {}
    for e in cols:
        vs = [(V[(k, e)], k) for k, _, _ in ROWS if V[(k, e)] is not None]
        complete = all(V[(k, e)] is not None for k, _, _ in ROWS if k not in OPTIONAL)
        order = sorted(set(round(100 * v, 2) for v, _ in vs))
        for k, _, _ in ROWS:
            v = V[(k, e)]
            if v is None:
                cell[(k, e)] = PH; continue
            s = f"{100 * v:.2f}"
            if complete and round(100 * v, 2) == order[0]:
                s = "\\textbf{" + s + "}"
            elif complete and len(order) > 1 and round(100 * v, 2) == order[1]:
                s = "\\underline{" + s + "}"
            cell[(k, e)] = s
    # user 10-11 rule: in the controlled environments DIVA-U equals DIVA, so the value is printed once, centered over both rows
    for d in DOM_ORDER:
        if ("diva", d) in cell and ("diva_u", d) in cell and cell[("diva", d)] == cell[("diva_u", d)]:
            cell[("diva_u", d)] = "\\multirow{-2}{*}{" + cell[("diva", d)] + "}"; cell[("diva", d)] = ""
    nc = len(cols) + 1
    lines, cur = [], None
    for k, name, g in ROWS:
        if g != cur:
            if cur is not None:
                lines.append("\\midrule")
            lines.append("\\multicolumn{" + str(nc) + "}{l}{\\textit{" + g + "}} \\\\")
            cur = g
        lab = name if "\\citep" not in name else "\\begin{tabular}[c]{@{}l@{}}" + name.replace(" \\citep", "\\\\\\citep") + "\\end{tabular}"
        lines.append(("\\rowcolor{oursrow} " if g == "Ours" else "") + lab + " & " + " & ".join(cell[(k, e)] for e in cols) + " \\\\")
    head = ("\\multirow{3}{*}[-1.4ex]{\\begin{tabular}[c]{@{}l@{}}Method\\\\(Avg.\\ across " + str(len(KNOW_LLMS)) + " LLMs)\\end{tabular}} & "
            "\\multicolumn{8}{c|}{Knowledge-required environments} & \\multicolumn{4}{c}{Controlled environments} \\\\\n"
            "\\cmidrule(lr){2-9}\\cmidrule(lr){10-13}\n"
            " & \\multicolumn{2}{c}{Credit} & \\multicolumn{2}{c}{Clinical} & \\multicolumn{4}{c|}{SEC filings (2025--2026)} & "
            + " & ".join("\\multirow{2}{*}[-0.6ex]{" + ENVN[d] + "}" for d in DOM_ORDER) + " \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-9}\n"
            " & " + " & ".join(ENVN[e] for e in ENV_ORDER[:7]) + " & \\multicolumn{1}{c|}{" + ENVN[ENV_ORDER[7]] + "} & & & & \\\\\n")
    tex = ("\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{2pt}\n\\begin{tabular}{l|cccccccc|cccc}\n\\toprule\n" + head
           + "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
           "\\caption{\\textbf{Closest prior work as baselines.} DP regret on the first 300 test scenarios over "
           + str(len(KNOW_LLMS)) + " LLMs. DeLLMa and UoT are adapted to acquire-or-stop, and \\diva{}-U equals \\diva{} in controlled environments.}\n"
           "\\label{tab:prior}\n\\end{table*}\n")
    open(f"{IS}/paper/tables/prior_baselines.tex", "w").write(tex)
    n_tbd = sum(v is None for v in V.values())
    print("prior_baselines.tex written; TBD cells:", n_tbd)
    for k, name, _ in ROWS:
        print(f"{k:9s}", " ".join("  TBD " if V[(k, e)] is None else f"{100 * V[(k, e)]:6.2f}" for e in cols))



def voi_tool_stats(res):
    """results/voi_tool.json: per environment, mean DP regret (x100) of VoI-Tool, DIVA and DIVA-U over the LLMs complete in
    that environment, the number of such LLMs, the share of first decisions that stop, and the parse-failure rate (failures
    fall back to stopping)."""
    out = {}
    for e in ENV_ORDER + DOM_ORDER:
        recs = res.get(e, {})
        tags = [t for t in KNOW_LLMS if t in recs and "voi_tool" in recs[t]]
        st, fails, n = 0, 0, 0
        for t in KNOW_LLMS:
            p = f"{IS}/baselines/outputs/{t}/voi_tool/{e}_test.jsonl.gz"
            if not os.path.exists(p):
                continue
            for r in read(p)[:300]:
                n += 1; st += r.get("choice") == "S"; fails += bool((r.get("info") or {}).get("fail"))
        mean = lambda k: float(np.mean([100 * recs[t][k][KEY] for t in tags])) if tags else None
        out[e] = {"n_llm": len(tags), "voi_tool": mean("voi_tool"), "diva": mean("diva"), "diva_u": mean("diva_u"),
                  "never": mean("never"), "stop_rate": st / n if n else None, "parse_fail": fails / n if n else None, "rows": n}
    json.dump(out, open(f"{IS}/results/voi_tool.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    main()
