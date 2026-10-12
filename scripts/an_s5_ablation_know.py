"""the ablation of Table tab:ablation repeated on the knowledge-required environments, by re-scoring existing
outputs (no LLM calls). First 300 test scenarios per environment, all 20 LLMs, first-decision DP regret.
Cumulative rows (each adds one module):
  direct      the direct policy
  lik         decision value on the elicited specification, outcome likelihoods, no shrinkage (records; SEC filings elicit
              outcome-conditioned risks only, so this row equals the next one there)
  cond        outcome-conditioned risks instead of likelihoods, no shrinkage
  diva        + shrinkage lambda = 0.75 (= DIVA, the frozen configuration)
  diva_u      + uncertainty over the elicited strengths (= DIVA-U)
Further rows: diva_dp (DP lookahead on the DIVA specification), fac_eig (EIG instead of the decision value on the DIVA
specification, 0.05-bit threshold as in Table tab:ablation), sup (records only: the specification fitted on the training
records, emp_s1 / emp_s0, without the LLM; on SEC filings the analogous specification is the reference model itself).
Rows the knowledge-required environments cannot reproduce without new LLM calls: decision value on raw imagined beliefs
and the 11-option elicitation (both were run on the controlled probe only).
Writes results/s5_ablation_know.json and paper/tables/abl_know.tex."""
import sys, os, json, collections
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from an_common import load, spec_u, KNOW_LLMS, TAB, SEC, TOL
from evalx import score, choose_max, clip
from dcore import posterior, voi_exact
from complete import ENVN, PH, fmt_best

ROWS = ["direct", "lik", "cond", "diva", "diva_u", "diva_dp", "fac_eig", "sup"]


def one(args):
    e, tag = args
    rows = load(e, tag)
    if rows is None:
        return e, tag, None
    from episodes_know import spec_dp_first
    R = collections.defaultdict(list)
    for sc, fr, dist in rows:
        g, l = float(sc["g"]), float(sc["l"])
        ch = {"direct": max(dist, key=dist.get)}
        if e in TAB:
            from eval_tab import diva_scores
            ch["lik"] = choose_max(diva_scores(sc, fr, "lik", "greedy", "elicit", 1.0)["net"])
            ch["cond"] = choose_max(diva_scores(sc, fr, "cond", "greedy", "elicit", 1.0)["net"])
            fs = diva_scores(sc, fr, "cond", "greedy", "elicit", 0.75)
            par = {k: ([1 - a1, a1], [1 - a0, a0], c) for k, (a1, a0, c) in fs["par"].items()}; b = fs["b"]
            # supervised specification: naive Bayes fitted on the training records (no LLM)
            cl = lambda p: min(max(float(p), 0.005), 0.995)
            bs = posterior(float(sc["prior"]), [(cl(o["emp_s1"]), cl(o["emp_s0"]), int(o["value"])) for o in sc["observed"]])
            ch["sup"] = choose_max({c["label"]: voi_exact(bs, cl(c["emp_s1"]), cl(c["emp_s0"]), g, l) - float(c["cost"]) for c in sc["checks"]})
        else:
            from eval_real import fio_real
            f1 = fio_real(sc, fr, lam=1.0)
            ch["lik"] = ch["cond"] = choose_max(f1["net"])
            fs = fio_real(sc, fr, lam=0.75); par = fs["par"]; b = fs["b0"]
        ch["diva"] = choose_max(fs["net"])
        ch["diva_u"] = choose_max(spec_u(e, sc, fr))
        ch["diva_dp"] = spec_dp_first(b, par, g, l)
        ch["fac_eig"] = choose_max(fs["eig"], 0.05)
        for k, c in ch.items():
            R[k].append(score(sc, c)["dp_regret"])
    return e, tag, {k: float(np.mean(v)) for k, v in R.items()}


def main():
    envs = TAB + SEC
    with Pool(int(os.environ.get("NPROC", "24"))) as p:
        res = p.map(one, [(e, t) for e in envs for t in KNOW_LLMS])
    per = collections.defaultdict(dict)
    for e, t, r in res:
        if r is not None:
            per[e][t] = r
    val = {}
    for e in envs:
        ok = len(per[e]) == len(KNOW_LLMS)
        for k in ROWS:
            vs = [per[e][t].get(k) for t in KNOW_LLMS if t in per[e]]
            val[(k, e)] = float(np.mean(vs)) if ok and all(v is not None for v in vs) and vs else None
    json.dump({f"{k}|{e}": v for (k, e), v in val.items()}, open(f"{IS}/results/s5_ablation_know.json", "w"), indent=1)

    def avg(k, es):
        v = [val[(k, e)] for e in es]
        return float(np.mean(v)) if all(x is not None for x in v) else None
    names = [("direct", "Direct"), ("lik", "+ Decision value on elicited likelihoods"), ("cond", "+ Outcome-conditioned risks"),
             ("diva", "+ Shrinkage $\\lambda{=}0.75$ (= \\diva{})"), ("diva_u", "+ Uncertainty (= \\diva{}-U)")]
    cols = envs
    lines = []
    for k, n in names:
        vs = [val[(k, e)] for e in cols] + [avg(k, cols)]
        cells = [PH if v is None else f"{100*v:.2f}" for v in vs]
        lines.append(("\\rowcolor{oursrow} " if k in ("diva", "diva_u") else "") + n + " & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    for k, n in [("diva_dp", "\\diva{} + DP lookahead (= \\diva{}-DP)"), ("fac_eig", "\\begin{tabular}[c]{@{}l@{}}\\diva{} $-$ decision value\\\\(EIG on the same specification)\\end{tabular}")]:
        vs = [val[(k, e)] for e in cols] + [avg(k, cols)]
        lines.append(n + " & " + " & ".join(PH if v is None else f"{100*v:.2f}" for v in vs) + " \\\\")
    lines.append("\\midrule")
    vs = [val[("sup", e)] if e in TAB else None for e in cols]
    lines.append("Reference: Supervised specification & " + " & ".join(("--" if e not in TAB else (PH if v is None else f"{100*v:.2f}")) for e, v in zip(cols, vs)) + " & -- \\\\")
    head = "Variant & " + " & ".join(ENVN[e] for e in cols) + " & Avg. \\\\"
    tex = ("\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{l|" + "c" * (len(cols) + 1) + "}\n\\toprule\n"
           + head + "\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
           "\\caption{\\textbf{Ablation on the knowledge-required environments.} DP regret ($\\times$100) on the first 300 test scenarios, averaged over the same 20 LLMs. "
           "SEC filings elicit outcome-conditioned risks only, so the first two specification rows coincide there. Supervised specification: Fitted on the training records without the LLM.}\n"
           "\\label{tab:abl_know}\n\\end{table*}\n")
    open(f"{IS}/paper/tables/abl_know.tex", "w").write(tex)
    for k in ROWS:
        print(f"{k:8s}", " ".join(f"{ENVN[e]}:{100*val[(k, e)]:.2f}" if val[(k, e)] is not None else f"{ENVN[e]}:--" for e in cols), "avg", avg(k, cols))


if __name__ == "__main__":
    main()
