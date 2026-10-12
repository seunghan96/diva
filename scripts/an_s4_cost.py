"""cost of the first acquisition decision for every method, on the controlled and the knowledge-required
environments, measured from the call logs of all 20 LLMs (first 300 test scenarios per environment; no LLM calls).
  LLM calls       number of LLM requests per scenario (a self-consistency request with n samples counts once)
  Gen. tokens     generated tokens per scenario (characters / 4 of every generated text; letter-probability reads generate none)
DIVA, DIVA-U and DIVA-DP share one elicited specification, so their calls are those of the specification: 1 + 2|C u K| short
answers on the controlled environments, 1 + 3|C u K| on the records (outcome-conditioned risks and the frequency of each
finding) and 1 + 3|C| + |K| on SEC filings (one risk per value bin), each answer at most six generated tokens. Wall-clock time is not
reported, as the calls ran concurrently on a shared serving fleet. Writes results/s4_cost.json and paper/tables/cost_all.tex."""
import sys, os, json, ast, collections
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from an_common import env_rows, dir_path, KNOW_LLMS, CDOMS, TAB, SEC, N300
from evalx import read
from complete import PH, mname, MCITE

BL = ["direct", "cot_need", "sc_cot", "react", "finrisk_ask", "voi_prompt", "voi_paper", "bed_llm", "pal"]


def calls_of(r):
    cs = r.get("calls", [])
    if isinstance(cs, str):
        try:
            cs = ast.literal_eval(cs)
        except Exception:
            return None, None
    n = len(cs)
    tok = sum(sum(len(x) for x in (c.get("raw") or []) if isinstance(x, str)) // 4 for c in cs if c.get("kind") == "gen")
    return n, tok


def one(args):
    m, e, tag = args
    ids = [s["id"] for s in env_rows(e)[:N300]]
    if m == "direct" and e not in CDOMS:  # knowledge-required direct: three letter-probability reads (stop option rotated)
        rr = {r["id"]: r for r in read(dir_path(e, tag)) if r.get("dist")}
        v = [(len(rr[i].get("mass", [1, 1, 1])), 0) for i in ids if i in rr]
    else:
        rr = {r["id"]: r for r in read(f"{IS}/baselines/outputs/{tag}/{m}/{e}_test.calls.jsonl.gz")}
        v = [calls_of(rr[i]) for i in ids if i in rr]
        v = [x for x in v if x[0] is not None]
    if len(v) < 0.95 * len(ids):
        return m, e, tag, None
    return m, e, tag, (float(np.mean([a for a, _ in v])), float(np.mean([b for _, b in v])))


def diva_calls(e):
    env = env_rows(e)[:N300]
    if e in CDOMS:
        return float(np.mean([1 + 2 * (len(s["checks"]) + len(s.get("observed", []))) for s in env]))
    if e in TAB:
        return float(np.mean([1 + 3 * (len(s["checks"]) + len(s.get("observed", []))) for s in env]))
    return float(np.mean([1 + 3 * len(s["checks"]) + len(s.get("observed", [])) for s in env]))


def main():
    envs = CDOMS + TAB + SEC
    jobs = [(m, e, t) for m in BL for e in envs for t in KNOW_LLMS]
    with Pool(int(os.environ.get("NPROC", "32"))) as p:
        R = p.map(one, jobs)
    per = collections.defaultdict(dict)
    for m, e, t, v in R:
        if v is not None:
            per[(m, e)][t] = v
    groups = {"knowledge": TAB + SEC, "controlled": CDOMS}  # knowledge-required first
    out = {}
    for m in BL:
        for g, es in groups.items():
            ok = all(len(per[(m, e)]) == len(KNOW_LLMS) for e in es)
            miss = [f"{t}:{e}" for e in es for t in KNOW_LLMS if t not in per[(m, e)]]
            out[f"{m}|{g}"] = {"calls": float(np.mean([per[(m, e)][t][0] for e in es for t in per[(m, e)]])) if ok else None,
                               "tokens": float(np.mean([per[(m, e)][t][1] for e in es for t in per[(m, e)]])) if ok else None, "missing": miss[:10], "n_missing": len(miss)}
    for g, es in groups.items():
        c = float(np.mean([diva_calls(e) for e in es]))
        out[f"diva|{g}"] = {"calls": c, "tokens_max": 6 * c}
    json.dump(out, open(f"{IS}/results/s4_cost.json", "w"), indent=1)
    lines = []
    for m in BL:
        name = mname(m); cite = MCITE.get(name)
        cell = ("\\begin{tabular}[c]{@{}l@{}}" + name + "\\\\\\citep{" + cite + "}\\end{tabular}") if cite else name
        vals = []
        for g in groups:
            o = out[f"{m}|{g}"]
            vals += [PH if o["calls"] is None else f"{o['calls']:.1f}", PH if o["tokens"] is None else f"{o['tokens']:.0f}"]
        lines.append(cell + " & " + " & ".join(vals) + " \\\\")
        print(m, out[f"{m}|controlled"], out[f"{m}|knowledge"]["calls"], out[f"{m}|knowledge"]["tokens"], out[f"{m}|knowledge"]["n_missing"])
    lines.append("\\midrule")
    vals = []
    for g in groups:
        o = out[f"diva|{g}"]
        vals += [f"{o['calls']:.1f}", f"$\\le${o['tokens_max']:.0f}"]
    lines.append("\\rowcolor{oursrow} \\begin{tabular}[c]{@{}l@{}}\\diva{}, \\diva{}-U,\\\\\\diva{}-DP\\end{tabular} & " + " & ".join(vals) + " \\\\")
    print("diva", out["diva|controlled"], out["diva|knowledge"])
    tex = ("\\begin{table}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{l|cc|cc}\n\\toprule\n"
           "\\multirow{2}{*}[-0.7ex]{Method} & \\multicolumn{2}{c|}{\\begin{tabular}[c]{@{}c@{}}Knowledge-\\\\required\\end{tabular}} & \\multicolumn{2}{c}{Controlled} \\\\\n"
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n & Calls & Tokens & Calls & Tokens \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
           "\\caption{\\textbf{Cost of the first acquisition decision} per scenario, averaged over the 20 LLMs and the first 300 test scenarios of every environment. "
           "Tokens: Generated tokens (characters / 4), at most six per answer of \\diva{}.}\n\\label{tab:cost_all}\n\\end{table}\n")
    open(f"{IS}/paper/tables/cost_all.tex", "w").write(tex)


if __name__ == "__main__":
    main()
