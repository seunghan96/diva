"""Reviewer M1: which information to acquire (selection) vs when to stop (stopping), on first decisions.
A composite policy stops if its STOP rule says stop and otherwise acquires the check ranked first by its SELECTION rule.
  selection: llm (the direct policy's most likely check, ignoring its stop option), eig (largest EIG on the elicited
             specification), diva (largest net decision value on the same specification), diva_u, oracle (largest DP Q-value)
  stopping:  llm (the direct policy stops), diva (no positive net value), diva_u, oracle (the DP optimum stops), never
First 300 test scenarios per environment, all 20 LLMs, the frozen specification of Table tab:know. DIVA-U equals DIVA on the
controlled environments. Writes results/m1_decomp.json: {group: {sel: {stop: {"dp": mean, "my": mean, "per_llm": {tag: dp}}}}}
plus diagnostics (stop accuracy, selection accuracy)."""
import sys, os, json, collections
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from an_common import load, spec_scores, spec_u, dp_stop_ok, oracle_check, KNOW_LLMS, CDOMS, TAB, SEC, TOL
from evalx import score

SELS = ["llm", "eig", "diva", "diva_u", "oracle"]
STOPS = ["llm", "diva", "diva_u", "oracle", "never"]


def one(args):
    e, tag = args
    rows = load(e, tag)
    if rows is None:
        return e, tag, None
    acc = collections.defaultdict(list)
    for sc, fr, dist in rows:
        fs = spec_scores(e, sc, fr)
        nu = spec_u(e, sc, fr) or fs["net"]
        labs = [c["label"] for c in sc["checks"]]
        sel = {"llm": max(labs, key=lambda k: dist.get(k, 0.0)), "eig": max(labs, key=lambda k: fs["eig"][k]),
               "diva": max(labs, key=lambda k: fs["net"][k]), "diva_u": max(labs, key=lambda k: nu[k]), "oracle": oracle_check(sc)}
        stop = {"llm": max(dist, key=dist.get) == "S", "diva": max(fs["net"].values()) <= TOL, "diva_u": max(nu.values()) <= TOL,
                "oracle": dp_stop_ok(sc), "never": False}
        for a in SELS:
            for b in STOPS:
                s = score(sc, "S" if stop[b] else sel[a])
                acc[(a, b, "dp")].append(s["dp_regret"]); acc[(a, b, "my")].append(s["norm_regret"])
        # diagnostics: stop decision agreement with the DP optimum; selected check is DP-optimal among checks
        best_q = max(float(v) for v in sc["exact"]["q"].values())
        for b in ["llm", "diva", "diva_u"]:
            acc[("stopacc", b, "")].append(float(stop[b] == dp_stop_ok(sc)))
        for a in SELS:
            acc[("selacc", a, "")].append(float(float(sc["exact"]["q"][sel[a]]) >= best_q - 1e-9))
    return e, tag, {"|".join(k): float(np.mean(v)) for k, v in acc.items()}


def main():
    envs = CDOMS + TAB + SEC
    jobs = [(e, t) for e in envs for t in KNOW_LLMS]
    with Pool(int(os.environ.get("NPROC", "24"))) as p:
        R = p.map(one, jobs)
    per = collections.defaultdict(dict)
    missing = []
    for e, t, r in R:
        if r is None:
            missing.append(f"{t}:{e}")
        else:
            per[e][t] = r
    groups = {"controlled": CDOMS, "records": TAB, "sec": SEC, "knowledge": TAB + SEC}
    out = {"missing": missing, "groups": {}}
    for g, es in groups.items():
        complete = all(t in per[e] for e in es for t in KNOW_LLMS)
        G = {}
        keys = set(k for e in es for t in per[e] for k in per[e][t])
        for k in keys:
            vals = {t: float(np.mean([per[e][t][k] for e in es])) for t in KNOW_LLMS if all(t in per[e] for e in es)}
            G[k] = {"mean": float(np.mean(list(vals.values()))) if complete else None, "per_llm": vals}
        out["groups"][g] = G
    out["per_env"] = {e: {k: float(np.mean([per[e][t][k] for t in per[e]])) for k in next(iter(per[e].values()))} for e in per if len(per[e]) == len(KNOW_LLMS)}
    json.dump(out, open(f"{IS}/results/m1_decomp.json", "w"), indent=1)
    print("missing", missing)
    for g in groups:
        print("==", g)
        for a in SELS:
            print(f"  sel={a:7s}", " ".join(f"{b}:{100*out['groups'][g][f'{a}|{b}|dp']['mean']:.2f}" if out['groups'][g][f'{a}|{b}|dp']['mean'] is not None else f"{b}:TBD" for b in STOPS))
        print("  stopacc", {b: round(out['groups'][g][f'stopacc|{b}|']['mean'] or -1, 3) for b in ["llm", "diva", "diva_u"]})
        print("  selacc", {a: round(out['groups'][g][f'selacc|{a}|']['mean'] or -1, 3) for a in SELS})


def table():
    """paper/tables/m1_decomp.tex: rows = selection rule, columns = stopping rule x {controlled, knowledge-required}"""
    from complete import PH
    out = json.load(open(f"{IS}/results/m1_decomp.json"))
    G = out["groups"]
    rows = [("llm", "LLM (direct policy)"), ("eig", "EIG"), ("diva_u", "\\diva{}-U"), ("oracle", "DP optimum")]
    cols = [("knowledge", "llm"), ("knowledge", "diva_u"), ("knowledge", "oracle"), ("controlled", "llm"), ("controlled", "diva_u"), ("controlled", "oracle")]  # knowledge-required first (user 10-12)
    lines = []
    for a, name in rows:
        cells = []
        for g, b in cols:
            v = G[g][f"{a}|{b}|dp"]["mean"]
            cells.append(PH if v is None else f"{100*abs(v) if abs(v) < 5e-5 else 100*v:.2f}")
        lines.append(("\\rowcolor{oursrow} " if a == "diva_u" else "") + name + " & " + " & ".join(cells) + " \\\\")
    tex = ("\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{6pt}\n\\begin{tabular}{l|ccc|ccc}\n\\toprule\n"
           "\\multirow{2}{*}[-0.6ex]{Selection rule} & \\multicolumn{3}{c|}{Knowledge-required: Stopping rule} & \\multicolumn{3}{c}{Controlled: Stopping rule} \\\\\n"
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n & LLM & \\diva{}-U & DP & LLM & \\diva{}-U & DP \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
           "\\caption{\\textbf{Which check vs.\\ when to stop.} DP regret ($\\times$100) of a policy that stops by the column's rule and otherwise acquires the check ranked first by the row's rule, "
           "averaged over the 20 LLMs and the first 300 test scenarios of every environment. DP: The DP optimum. \\diva{}-U equals \\diva{} on the controlled environments.}\n"
           "\\label{tab:m1_decomp}\n\\end{table*}\n")
    open(f"{IS}/paper/tables/m1_decomp.tex", "w").write(tex)


if __name__ == "__main__":
    if sys.argv[1:] != ["table"]:
        main()
    table()
