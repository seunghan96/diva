"""R6-SF2: multi-state baselines with thresholds tuned on the DEVELOPMENT split (per LLM and domain) on DIVA's own
specification: Fac-EIG‡ and Fac-ES‡ (acquire the argmax if its score exceeds tau), plus Always ask (DIVA's top check).
Test = held-out scenarios 500-999 (outputs/test_ms2). Writes results/ms_tuned.json {tag: {dom: {method: summary}}}."""
import sys, os, glob, json, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read, score, summarize, choose_max
from eval_ms import build, entropy_search, LAST_SPEC
TAUS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
K = "dp_regret"


def scores(sc, fr):
    M, _ = build(sc, fr); esv = entropy_search(sc, LAST_SPEC[0])
    labs = [c["label"] for c in sc["checks"]]; net, eig = {}, {}
    for j, lab in enumerate(labs):
        st = M.item_stats([], j); net[lab] = st[0] - M.cost[j]; eig[lab] = st[1]
    return net, eig, esv


def main():
    out = collections.defaultdict(dict)
    for f in glob.glob(f"{IS}/outputs/test_ms2/ms_*.jsonl.gz"):
        tag, dom = os.path.basename(f)[3:-9].rsplit("_", 1)
        dvf = f"{IS}/outputs/ms_dev/ms_{tag}_{dom}.jsonl.gz"
        if not os.path.exists(dvf):
            continue
        dev = {s["id"]: s for s in read(f"{IS}/envs_synth/data/ms/{dom}_dev.jsonl.gz")}
        dfr = {r["id"]: r for r in read(dvf)}
        D = [(dev[i], *scores(dev[i], dfr[i])) for i in dev if i in dfr]
        if len(D) < 100:
            continue
        te = {s["id"]: s for s in read(f"{IS}/envs_synth/data/ms/{dom}_test.jsonl.gz")}
        tfr = {r["id"]: r for r in read(f)}
        T = [(te[i], *scores(te[i], tfr[i])) for i in te if i in tfr]
        res = {}
        for nm, idx in (("fac_eig_tuned", 2), ("fac_es_tuned", 3)):
            tau = min(TAUS, key=lambda t: summarize([score(x[0], choose_max(x[idx], t)) for x in D])[K])
            res[nm] = summarize([score(x[0], choose_max(x[idx], tau)) for x in T]) | {"tau": tau}
        res["always_top"] = summarize([score(x[0], max(x[1], key=x[1].get)) for x in T])
        res["diva"] = summarize([score(x[0], choose_max(x[1])) for x in T])
        out[tag][dom] = res
        print(tag, dom, {m: (round(100 * v[K], 2), round(v["stopped"], 2)) for m, v in res.items()}, flush=True)
    json.dump(out, open(f"{IS}/results/ms_tuned.json", "w"), indent=1)


if __name__ == "__main__":
    main()
