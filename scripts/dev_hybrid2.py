"""DEV-ONLY: hybrid of DIVA's stopping decision with the LLM's direct ranking of checks (no new LLM calls).
Variants: {diva, diva_u} x check choice {own argmax, direct argmax over checks}. Writes results/dev_hybrid2.json."""
import sys, os, glob, json, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read, score, summarize, choose_max
from eval_tab import diva_scores, diva_u
from eval_real import fio_real, fio_real_u
out = collections.defaultdict(dict)
cells = []
for ds in ["taiwan", "german", "cdc", "heart"]:
    for f in glob.glob(f"{IS}/outputs/tab_dev/fk_*_{ds}.jsonl.gz"):
        tag = os.path.basename(f)[3:-9][: -(len(ds) + 1)]
        cells.append(("tab", ds, f"{IS}/envs_tab/data/{ds}_dev.jsonl.gz", f, f"{IS}/outputs/tab_dev/dir_{tag}_{ds}.jsonl.gz", tag))
for tg in ["ret_1", "ret_5", "vol_5", "vol_20"]:
    dn = "ret1" if tg == "ret_1" else tg
    for f in glob.glob(f"{IS}/outputs/real_dev/fr_*_{dn}.jsonl.gz"):
        tag = os.path.basename(f)[3:-9][: -(len(dn) + 1)]
        if tag.split("_")[-1] in ("joint", "ratio", "contrast"):
            continue
        cells.append(("sec", tg, f"{IS}/envs_real/data/{tg}_dev.jsonl.gz", f, f"{IS}/outputs/real_dev/dir_{tag}_{dn}.jsonl.gz", tag))
for kind, e, ep, fp, dp, tag in cells:
    if not os.path.exists(dp):
        continue
    env = {s["id"]: s for s in read(ep)}; fr = {r["id"]: r for r in read(fp)}
    dr = {r["id"]: r["dist"] for r in read(dp) if r.get("dist")}
    ids = [i for i in env if i in fr and i in dr]
    if len(ids) < 100:
        continue
    V = collections.defaultdict(list)
    for i in ids:
        sc = env[i]
        f = diva_scores(sc, fr[i], "cond", "greedy", "elicit", 0.75) if kind == "tab" else fio_real(sc, fr[i], lam=0.75)
        fu = diva_u(sc, fr[i], 1.0) if kind == "tab" else fio_real_u(sc, fr[i], 1.0)
        labs = [c["label"] for c in sc["checks"]]
        dd = {k: v for k, v in dr[i].items() if k in labs}
        dtop = max(dd, key=dd.get) if dd else None
        dfull = max(dr[i], key=dr[i].get)
        for nm, net in (("diva", f["net"]), ("diva_u", fu["net"])):
            c = choose_max(net)
            V[nm].append(score(sc, c))
            V[nm + "+dirck"].append(score(sc, "S" if c == "S" or dtop is None else dtop))
        V["direct"].append(score(sc, dfull)); V["never"].append(score(sc, "S"))
    out[e][tag] = {m: {"dp": summarize(v)["dp_regret"], "myo": summarize(v)["norm_regret"]} for m, v in V.items()}
json.dump(out, open(f"{IS}/results/dev_hybrid2.json", "w"), indent=1)
for e, c in out.items():
    ms = list(next(iter(c.values())).keys())
    print(e, len(c), {m: (round(100 * np.mean([v[m]["dp"] for v in c.values()]), 2), round(100 * np.mean([v[m]["myo"] for v in c.values()]), 2)) for m in ms})
