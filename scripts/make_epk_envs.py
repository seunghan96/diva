"""Episode copies of the knowledge-required test environments (rows 0-999) with K sampled worlds per scenario from the
REFERENCE model (tabular: the stored joint over the six checks and Y; SEC: the stored naive-Bayes model), so that
multi-step prompting policies can be run as full episodes (baselines.run --mode episode). Writes
envs_tab/data/{ds}_epk.jsonl.gz and envs_real/data/{tg}_epk.jsonl.gz."""
import os, sys, json, gzip, ast, zlib
import numpy as np
IS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = 2


def tab_worlds(sc, rng):
    J = np.asarray(sc["joint"], dtype=float).reshape(64, 2); p = J.flatten() / J.sum()
    labs = sc.get("label_order") or [c["label"] for c in sc["checks"]]
    out = []
    for _ in range(K):
        k = rng.choice(128, p=p); cfg, y = divmod(k, 2)
        out.append({"state": int(y), "y": int(y), "outcomes": {L: int((cfg >> i) & 1) for i, L in enumerate(labs)}, "source": "reference_joint"})
    return out


def sec_worlds(sc, rng):
    m = ast.literal_eval(sc["model"]) if isinstance(sc["model"], str) else sc["model"]
    pri = np.asarray(m["prior"], dtype=float); pri = pri / pri.sum()
    out = []
    for _ in range(K):
        y = int(rng.choice(len(pri), p=pri)); oc = {}
        for it in m["items"]:
            lk = np.asarray(it["lik"][y], dtype=float); oc[it["label"]] = int(rng.choice(len(lk), p=lk / lk.sum()))
        out.append({"state": y, "y": y, "outcomes": oc, "source": "reference_nb"})
    return out


for kind, envs in [("tab", ["taiwan", "german", "cdc", "heart"]), ("sec", ["ret_1", "ret_5", "vol_5", "vol_20"])]:
    for e in envs:
        src = f"{IS}/envs_tab/data/{e}_test.jsonl.gz" if kind == "tab" else f"{IS}/envs_real/data/{e}_test.jsonl.gz"
        dst = src.replace("_test.jsonl.gz", "_epk.jsonl.gz")
        rng = np.random.default_rng(zlib.crc32(f"{e}-epk".encode()))
        rows = []
        for k, l in enumerate(gzip.open(src, "rt")):
            if k >= 1000:
                break
            sc = json.loads(l)
            sc["episodes"] = tab_worlds(sc, rng) if kind == "tab" else sec_worlds(sc, rng)
            rows.append(sc)
        with gzip.open(dst, "wt") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(e, len(rows), flush=True)
