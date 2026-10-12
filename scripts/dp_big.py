"""DIVA-DP first action for controlled scenarios with more than 8 checks (exact DP under the elicited specification),
computed in parallel and cached per LLM: results/dp_cache/{tag}.json {id: action}. agg_bin.py reads the cache."""
import os, sys, glob, json
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read, fac_scores
from dcore import dp_value

DOMS = ["finance", "credit", "clinical", "legal"]


def one(args):
    sc, r = args
    fs = fac_scores(sc, r)
    keys = list(fs["par"])
    v, a = dp_value(fs["b"], [fs["par"][k] for k in keys], float(sc["g"]), float(sc["l"]))
    return r["id"], ("S" if a == "S" else keys[a])


def main(tags):
    os.makedirs(f"{IS}/results/dp_cache", exist_ok=True)
    env = {}
    for d in DOMS:
        for s in read(f"{IS}/envs_synth/data/binary/{d}_test.jsonl.gz"):
            if len(s["checks"]) > 8:
                env[s["id"]] = s
    with Pool(int(os.environ.get("DP_PROC", "48"))) as pool:
        for tag in tags:
            p = f"{IS}/results/dp_cache/{tag}.json"
            cache = json.load(open(p)) if os.path.exists(p) else {}
            work = []
            for d in DOMS:
                for r in read(f"{IS}/outputs/test_bin/facnum_{tag}_{d}.jsonl.gz"):
                    if r["id"] in env and r["id"] not in cache:
                        work.append((env[r["id"]], r))
            for i, a in pool.imap_unordered(one, work, chunksize=4):
                cache[i] = a
            json.dump(cache, open(p, "w"))
            print(tag, len(cache), flush=True)


if __name__ == "__main__":
    main(sys.argv[1].split(","))
