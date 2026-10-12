"""Clean evaluation on FRESH tabular scenarios (envs_tab/data/*_fresh, never used in any design decision) for every
method, on the first 300 fresh scenarios and the fixed LLM set (complete.KNOW_LLMS). SEC: the held-out slice is
rows >= 1000 (know_main f1000). A cell is a number only when every LLM has the method on all 300 scenarios.
Writes results/fresh.json and paper/tables/fresh.tex."""
import os, sys, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read, score, summarize, choose_max
from eval_tab import diva_scores, diva_u
from complete import PH, KNOW_LLMS
from make_tables_app import _tab, ENVN, _pair
from complete import msort, mname, NAMES, group_rows
from make_tables import is_ours

BL = {"cot_need": "CoT", "sc_cot": "SC", "react": "ReAct", "finrisk_ask": "Ask-P", "voi_prompt": "VoI-P", "voi_paper": "VoI-Ask",
      "bed_llm": "BED-LLM", "pal": "PAL"}
MS = [(m, mname(m)) for m, _ in msort([("direct", "Direct")] + list(BL.items()) + [("never", "Never"), ("always_top", "Always ask"), ("diva", r"\diva{}"), ("diva_u", r"\diva{}-U")])]
N = 300


def main():
    res = {}
    for e in ["taiwan", "german", "cdc", "heart"]:
        envl = read(f"{IS}/envs_tab/data/{e}_fresh.jsonl.gz")[:N]; env = {s["id"]: s for s in envl}
        res[e] = {}
        for t in KNOW_LLMS:
            fp, dp = f"{IS}/outputs/fresh_tab/fk_{t}_{e}.jsonl.gz", f"{IS}/outputs/fresh_tab/dir_{t}_{e}.jsonl.gz"
            if not (os.path.exists(fp) and os.path.exists(dp)):
                continue
            fr = {r["id"]: r for r in read(fp)}
            dr = {r["id"]: max(r["dist"], key=r["dist"].get) for r in read(dp) if r.get("dist")}
            bl = {}
            for m in BL:
                p = f"{IS}/baselines/outputs/{t}/{m}/{e}_fresh.jsonl.gz"
                if os.path.exists(p):
                    bl[m] = {r["id"]: r["choice"] for r in read(p) if r.get("choice") is not None}
            R = {}
            for m, _ in MS:
                rows = []
                for i, sc in env.items():
                    if i not in fr:
                        continue
                    if m == "direct":
                        c = dr.get(i)
                    elif m in BL:
                        c = bl.get(m, {}).get(i)
                    elif m == "never":
                        c = "S"
                    else:
                        f = diva_scores(sc, fr[i], "cond", "greedy", "elicit", 0.75)
                        c = {"diva": choose_max(f["net"]), "always_top": max(f["net"], key=f["net"].get)}.get(m) or choose_max(diva_u(sc, fr[i], 1.0)["net"])
                    if c is not None:
                        rows.append(score(sc, c))
                if len(rows) >= N:
                    s = summarize(rows); R[m] = {"dp": s["dp_regret"], "myo": s["norm_regret"]}
            res[e][t] = R
    json.dump(res, open(f"{IS}/results/fresh.json", "w"), indent=1)
    write_table(res)


def write_table(res):
    """rows = methods (canonical order), columns = datasets; cells 'DP / myopic' regret averaged over the fixed LLM set"""
    envs = ["taiwan", "german", "cdc", "heart"]
    cols = {}
    for e in envs:
        ok = [all(m in res.get(e, {}).get(t, {}) for t in KNOW_LLMS) for m, _ in MS]
        dp = [float(np.mean([res[e][t][m]["dp"] for t in KNOW_LLMS])) if o else None for (m, _), o in zip(MS, ok)]
        my = [float(np.mean([res[e][t][m]["myo"] for t in KNOW_LLMS])) if o else None for (m, _), o in zip(MS, ok)]
        cols[e] = [PH if d is None else c for d, c in zip(dp, _pair(dp, my))]
    rows = [("\\rowcolor{oursrow} " if is_ours(n) else "") + n + " & " + " & ".join(cols[e][i] for e in envs) + r" \\" for i, (m, n) in enumerate(MS)]
    rows = group_rows([n for _, n in MS], rows, 1 + len(envs))
    t = _tab(["Method"] + [ENVN[e] for e in envs], rows, r"\textbf{Fresh tabular scenarios} not used in any design decision, averaged over the same 20 LLMs.", "tab:fresh", star=True, size=r"\small")
    open(f"{IS}/paper/tables/fresh.tex", "w").write(t)
    print("ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "table":  # re-write the table from results/fresh.json only
        write_table(json.load(open(f"{IS}/results/fresh.json")))
    else:
        main()
