"""k=4 sampled specifications (greedy answer + 4 samples per question): DIVA with the greedy specification vs the
sample-mean specification (log-odds average of 5 answers), and the spread of DIVA's choices across the five single-sample
specifications (choice agreement). Rows with k=4 outputs only. Writes results/k4.json and paper/tables/k4.tex."""
import sys, os, glob, json, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read, score, summarize, choose_max
from eval_tab import diva_scores
from eval_real import fio_real, spec_j
from make_tables_app import _tab, ENVN


def tab_spec_j(fr, j):
    pick = lambda v: [v[j]] if isinstance(v, list) and j < len(v) and v[j] is not None else (v[:1] if isinstance(v, list) else v)
    f2 = {"prior": pick(fr["prior"]), "obs": [{k: pick(x) for k, x in o.items()} for o in fr.get("obs", [])],
          "chk": {c: {k: pick(x) for k, x in r.items()} for c, r in fr["chk"].items()}}
    return f2


out = collections.defaultdict(dict)
for kind, envs in [("tab", ["taiwan", "german", "cdc", "heart"]), ("sec", ["ret_1", "ret_5", "vol_5", "vol_20"])]:
    for e in envs:
        ep = f"{IS}/envs_tab/data/{e}_test.jsonl.gz" if kind == "tab" else f"{IS}/envs_real/data/{e}_test.jsonl.gz"
        env = {s["id"]: s for s in read(ep)[:1000]}  # every environment: test rows 0-999
        for f in glob.glob(f"{IS}/outputs/test_tab_k4/fk_*_{e}.jsonl.gz" if kind == "tab" else f"{IS}/outputs/test_real_k4/fr_*_{e}.jsonl.gz"):
            tag = os.path.basename(f)[3:-9][: -(len(e) + 1)]
            fr = {r["id"]: r for r in read(f) if r.get("k")}
            ids = [i for i in env if i in fr]
            if len(ids) < 100:
                continue
            R = collections.defaultdict(list); agree = []
            for i in ids:
                sc = env[i]
                if kind == "tab":
                    g = diva_scores(sc, fr[i], "cond", "greedy", "elicit", 0.75); m = diva_scores(sc, fr[i], "cond", "mean", "elicit", 0.75)
                    chs = [choose_max(diva_scores(sc, tab_spec_j(fr[i], j), "cond", "greedy", "elicit", 0.75)["net"]) for j in range(5)]
                else:
                    g = fio_real(sc, fr[i], lam=0.75); m = fio_real(sc, fr[i], mode="mean", lam=0.75)
                    chs = [choose_max(fio_real(sc, spec_j(fr[i], j), lam=0.75)["net"]) for j in range(5)]
                R["greedy"].append(score(sc, choose_max(g["net"]))); R["mean"].append(score(sc, choose_max(m["net"]))); R["never"].append(score(sc, "S"))
                agree.append(max(chs.count(c) for c in set(chs)) / 5)
            out[e][tag] = {k: summarize(v)["dp_regret"] for k, v in R.items()} | {"agree": float(np.mean(agree)), "n": len(ids)}
json.dump(out, open(f"{IS}/results/k4.json", "w"), indent=1)
# the table is written with the completeness rules of complete.py (all intended LLMs, every scenario)
from make_tables_app import table_k4
print(table_k4())
