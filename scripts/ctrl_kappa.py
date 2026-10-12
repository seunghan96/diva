"""DIVA-kappa / DIVA-beta on the CONTROLLED environments: kappa (cost multiplier) and beta (additive bonus) tuned per LLM
on the controlled development split (outputs/dev_bin/facnum_*), same grids and criterion (DP regret) as kappa_tune.py.
Writes results/kappa_ctrl.json {tag: {"kappa", "beta", "dev": {...}, "dev_beta": {...}, "n_dev"}}."""
import os, sys, glob, json
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read, fac_scores
from kappa_tune import tune, tune_beta

DOMS = ["finance", "credit", "clinical", "legal"]
out = {}
dev = {}
for d in DOMS:
    for s in read(f"{IS}/envs_synth/data/binary/{d}_dev.jsonl.gz"):
        dev[s["id"]] = s
tags = sorted({os.path.basename(f)[7:-9].rsplit("_", 1)[0] for f in glob.glob(f"{IS}/outputs/dev_bin/facnum_*.jsonl.gz")})
for tag in tags:
    pairs = []
    for d in DOMS:
        for r in read(f"{IS}/outputs/dev_bin/facnum_{tag}_{d}.jsonl.gz"):
            if r["id"] in dev:
                pairs.append((dev[r["id"]], fac_scores(dev[r["id"]], r)))
    if len(pairs) < 400:
        continue
    k, rk = tune(pairs); b, rb = tune_beta(pairs)
    out[tag] = {"kappa": k, "beta": b, "dev": rk, "dev_beta": rb, "n_dev": len(pairs)}
    print(tag, k, b, len(pairs), flush=True)
json.dump(out, open(f"{IS}/results/kappa_ctrl.json", "w"), indent=1)
