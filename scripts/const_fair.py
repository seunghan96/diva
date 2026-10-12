"""Diagnosis of the generic-specification control (scripts/const_strength.py). For every environment and generic strength
delta in GRID, the generic DIVA / DIVA-U on the development split (first 1,000) and on the first 300 / 1,000 test scenarios,
with stop and over/under-acquisition rates. Protocol of DIVA: ONE delta per variant selected on the average development
DP regret over the eight environments (const_strength.py selects delta per environment and per variant).
Writes results/const_fair.json."""
import sys, os, json
from multiprocessing import Pool
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read, score, summarize
from const_strength import GRID, choices
KEYS = ["dp_regret", "norm_regret", "stopped", "dp_stop", "over_acq", "under_acq"]
ENVS = [("tab", e) for e in ["taiwan", "german", "cdc", "heart"]] + [("sec", e) for e in ["ret_1", "ret_5", "vol_5", "vol_20"]]


def run(arg):
    kind, e = arg
    base = f"{IS}/envs_tab/data/{e}" if kind == "tab" else f"{IS}/envs_real/data/{e}"
    out = {}
    for split, rows in [("dev", read(f"{base}_dev.jsonl.gz")[:1000]), ("test", read(f"{base}_test.jsonl.gz")[:1000])]:
        for d in GRID:
            a, b = [], []
            for sc in rows:
                c1, c2 = choices(kind, sc, d)
                a.append(score(sc, c1)); b.append(score(sc, c2))
            for name, v in [("diva", a), ("diva_u", b)]:
                out[f"{split}:{name}:{d}"] = {k: summarize(v)[k] for k in KEYS}
                if split == "test":
                    out[f"s300:{name}:{d}"] = {k: summarize(v[:300])[k] for k in KEYS}
    nv = [score(sc, "S") for sc in read(f"{base}_test.jsonl.gz")[:300]]
    out["s300:never"] = {k: summarize(nv)[k] for k in KEYS}
    return e, out


if __name__ == "__main__":
    with Pool(8) as p:
        R = dict(p.map(run, ENVS))
    sel = {m: min(GRID, key=lambda d: np.mean([R[e][f"dev:{m}:{d}"]["dp_regret"] for _, e in ENVS])) for m in ["diva", "diva_u"]}
    R["_selected"] = sel
    json.dump(R, open(f"{IS}/results/const_fair.json", "w"), indent=1)
    print("selected", sel)
