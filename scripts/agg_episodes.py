"""Aggregate episode outputs (DIVA family from outputs/episodes, LLM baselines from baselines/outputs/*/*/*.episode.jsonl.gz).
Utility normalized by l. Budget curves: decision utility WITHOUT costs (u_expected + cost) after k forced acquisitions.
Writes results/episodes.json."""
import sys, os, json, glob, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read

DOMS = ["finance", "credit", "clinical", "legal"]
L = {}
for d in DOMS:
    for s in read(f"{IS}/envs_synth/data/binary/{d}_test.jsonl.gz"):
        L[s["id"]] = float(s["l"])
out = collections.defaultdict(lambda: collections.defaultdict(dict))
for f in glob.glob(f"{IS}/outputs/episodes/ep_*.jsonl.gz"):
    b = os.path.basename(f)[3:-9]; tag, dom = b.rsplit("_", 1)
    rows = [r for r in read(f) if r["id"] in L]
    ids = sorted({r["id"] for r in rows})
    by = collections.defaultdict(list); curve = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r.get("cau_expected") is None:
            continue
        R_ = r.get("util_range") or 1.0
        if "k" in r:
            curve[r["method"]][r["k"]].append((r["cau_expected"] + r["cost"]) / R_)
        else:
            by[r["method"]].append(r["policy_regret_norm"]); by[r["method"] + ":cau"].append(r["cau_expected"] / R_)
            by[r["method"] + ":n_acq"].append(r.get("n_acq", 0))
    for m, v in by.items():
        out[tag][dom][m] = float(np.mean(v))
    for m, c in curve.items():
        out[tag][dom]["curve:" + m] = {int(k): float(np.mean(v)) for k, v in sorted(c.items())}
    out[tag][dom]["ids"] = ids
# LLM baseline episodes
for f in glob.glob(f"{IS}/baselines/outputs/*/*/*_test.episode.jsonl.gz") + glob.glob(f"{IS}/baselines/outputs_palfix/*/*/*_test.episode.jsonl.gz"):
    tag = f.split("/")[-3]; meth = f.split("/")[-2]; dom = os.path.basename(f).split("_")[0]
    if meth == "pal" and "/outputs/" in f:
        continue  # pre-fix PAL episodes are invalid
    rows = [r for r in read(f) if r["id"] in L and r.get("eval")]
    if rows:
        out[tag][dom]["bl_" + meth] = float(np.mean([r["eval"]["policy_regret_norm"] for r in rows]))
        out[tag][dom]["bl_" + meth + ":cau"] = float(np.mean([r["eval"]["cau_expected"] / r["eval"]["util_range"] for r in rows]))
        out[tag][dom]["bl_" + meth + ":n_acq"] = float(np.mean([r.get("n_acq", 0) for r in rows]))
        out[tag][dom]["bl_" + meth + ":ids"] = sorted({r["id"] for r in rows})
# matched episodes (R6-SF3): DIVA variants re-scored on exactly the (scenario, world) pairs of the LLM baselines
for f in glob.glob(f"{IS}/outputs/episodes/ep_*.jsonl.gz"):
    b = os.path.basename(f)[3:-9]; tag, dom = b.rsplit("_", 1)
    pairs = None
    for g in glob.glob(f"{IS}/baselines/outputs/{tag}/*/{dom}_test.episode.jsonl.gz") + glob.glob(f"{IS}/baselines/outputs_palfix/{tag}/*/{dom}_test.episode.jsonl.gz"):
        if g.split("/")[-2] == "pal" and "/outputs/" in g:
            continue
        pr = {(r["id"], r.get("rep", 0)) for r in read(g) if r.get("eval")}
        pairs = pr if pairs is None else (pairs & pr)
    if not pairs:
        continue
    by = collections.defaultdict(list)
    for r in read(f):
        if "k" in r or r.get("cau_expected") is None:
            continue
        if (r["id"], r.get("rep", 0)) in pairs:
            by[r["method"]].append(r["policy_regret_norm"])
    for m, v in by.items():
        out[tag][dom][m + "@bl"] = float(np.mean(v)); out[tag][dom][m + "@bl:n"] = len(v)
json.dump(out, open(f"{IS}/results/episodes.json", "w"), indent=1)
for tag in sorted(out):
    for dom in DOMS:
        o = out[tag].get(dom, {})
        if "fiovoi" in o:
            print(tag, dom, " ".join(f"{m}:{o[m]:.3f}" for m in ["fiovoi", "fiovoi_dp", "fac_eig", "oracle_dp"] if m in o),
                  "stop:%.3f" % o["curve:fiovoi_budget"][0] if "curve:fiovoi_budget" in o else "")
