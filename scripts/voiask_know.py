"""VoI-Ask (Dong et al. 2026 pipeline) on knowledge-required environments: identical first-300 rows with DIVA, DIVA-U,
direct and never for every LLM with a VoI-Ask output. Writes results/voiask_know.json and paper/tables/voiask_know.tex."""
import sys, os, glob, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read, score, summarize, choose_max
from eval_tab import diva_scores, diva_u
from eval_real import fio_real, fio_real_u
from make_tables_app import _tab, ENVN
K = "dp_regret"
out = {}
for kind, envs in [("tab", ["taiwan", "german", "cdc", "heart"]), ("sec", ["ret_1", "ret_5", "vol_5", "vol_20"])]:
    for e in envs:
        ep = f"{IS}/envs_tab/data/{e}_test.jsonl.gz" if kind == "tab" else f"{IS}/envs_real/data/{e}_test.jsonl.gz"
        env = {s["id"]: s for s in read(ep)[:300]}
        for p in glob.glob(f"{IS}/baselines/outputs/*/voi_paper/{e}_test.jsonl.gz"):
            tag = p.split("/")[-3]
            va = {r["id"]: r["choice"] for r in read(p) if r.get("choice") is not None}
            fp = f"{IS}/outputs/test_tab/fk_{tag}_{e}.jsonl.gz" if kind == "tab" else f"{IS}/outputs/test_real/fr_{tag}_{e}.jsonl.gz"
            dp = f"{IS}/outputs/test_tab/dir_{tag}_{e}.jsonl.gz" if kind == "tab" else f"{IS}/outputs/test_real/dir_{tag}_{e}.jsonl.gz"
            if not os.path.exists(fp):
                continue
            fr = {r["id"]: r for r in read(fp)}
            dr = {r["id"]: max(r["dist"], key=r["dist"].get) for r in read(dp) if r.get("dist")} if os.path.exists(dp) else {}
            ids = [i for i in env if i in va and i in fr and i in dr]
            if len(ids) < 250:
                continue
            R = {"voi_ask": [], "direct": [], "never": [], "diva": [], "diva_u": []}
            for i in ids:
                sc = env[i]
                fs = diva_scores(sc, fr[i], "cond", "greedy", "elicit", 0.75) if kind == "tab" else fio_real(sc, fr[i], lam=0.75)
                fu = diva_u(sc, fr[i], 1.0) if kind == "tab" else fio_real_u(sc, fr[i], 1.0)
                for m, c in [("voi_ask", va[i]), ("direct", dr[i]), ("never", "S"), ("diva", choose_max(fs["net"])), ("diva_u", choose_max(fu["net"]))]:
                    R[m].append(score(sc, c))
            out.setdefault(e, {})[tag] = {m: summarize(v) for m, v in R.items()}
json.dump(out, open(f"{IS}/results/voiask_know.json", "w"), indent=1)
# the table is written with the completeness rules of complete.py (all intended LLMs, every scenario)
from make_tables_app import table_voiask
print(table_voiask())
