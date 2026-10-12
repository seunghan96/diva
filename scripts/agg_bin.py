"""Aggregate first-step results on the controlled binary test track into results/bin_main.json.
Methods: DIVA (facnum), DIVA-DP, Fac-EIG, always-stop, oracle, direct (run_direct), and every baseline found under
baselines/outputs/{tag}/{method}/{domain}_test*.jsonl.gz. Evaluates each method on the scenarios it covers and, for the
main table, on the common subset (first N rows per domain covered by all methods)."""
import sys, os, json, glob, collections, gzip
import numpy as np
DPC = {}
KC = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'kappa_ctrl.json'))) if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'kappa_ctrl.json')) else {}
sys.path.insert(0, os.path.dirname(__file__))
from evalx import read, fac_scores, choose_max, score, summarize, spearman
from dcore import dp_value

IS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOMS = ["finance", "credit", "clinical", "legal"]


def load_env(d):
    return {s["id"]: s for s in read(f"{IS}/envs_synth/data/binary/{d}_test.jsonl.gz")}


def bl_rows(tag, dom):
    out = {}
    for p in glob.glob(f"{IS}/baselines/outputs/{tag}/*/{dom}_test*.jsonl.gz"):
        if p.endswith(".calls.jsonl.gz") or "episode" in p:
            continue
        meth = p.split("/")[-2]
        for r in read(p):
            out.setdefault(meth, {})[r["id"]] = r
    return out


def main(tags, nsub=300):
    res = {}
    X = json.load(open(f"{IS}/results/extra_controls.json")).get("controlled", {}) if os.path.exists(f"{IS}/results/extra_controls.json") else {}
    from extra_controls import eig_cost_choice
    for d in DOMS:
        env = load_env(d)
        ids_order = list(env)
        env = {i: env[i] for i in ids_order}
        for tag in tags:
            R = collections.defaultdict(dict)
            for r in read(f"{IS}/outputs/test_bin/facnum_{tag}_{d}.jsonl.gz"):
                sc = env.get(r["id"])
                if sc is None:
                    continue
                fs = fac_scores(sc, r)
                R["diva"][r["id"]] = choose_max(fs["net"])
                R["always_top"][r["id"]] = max(fs["net"], key=fs["net"].get)  # Always ask: DIVA's top-ranked check, never stops
                R["fac_eig"][r["id"]] = choose_max(fs["eig"], 0.05)
                if tag in KC:
                    from kappa_tune import kappa_choice, beta_choice
                    R["diva_kappa"][r["id"]] = kappa_choice(sc, fs, KC[tag]["kappa"]); R["diva_beta"][r["id"]] = beta_choice(sc, fs, KC[tag]["beta"])
                if tag in X:
                    R["fac_eig_cost"][r["id"]] = eig_cost_choice(sc, fs["eig"], X[tag]["lam"], X[tag]["tau"])
                if len(sc["checks"]) <= 8:
                    keys = list(fs["par"]); v, a = dp_value(fs["b"], [fs["par"][k] for k in keys], float(sc["g"]), float(sc["l"]))
                    R["diva_dp"][r["id"]] = "S" if a == "S" else keys[a]
                elif r["id"] in DPC.setdefault(tag, (json.load(open(f"{IS}/results/dp_cache/{tag}.json")) if os.path.exists(f"{IS}/results/dp_cache/{tag}.json") else {})):
                    R["diva_dp"][r["id"]] = DPC[tag][r["id"]]  # exact DP for >8 checks (scripts/dp_big.py)
            for r in read(f"{IS}/outputs/think/dir_{tag}_{d}.jsonl.gz"):
                if r.get("dist"):
                    R["think"][r["id"]] = max(r["dist"], key=r["dist"].get)
            for r in read(f"{IS}/outputs/test_dir/dir_{tag}_{d}.jsonl.gz"):
                if r.get("dist"):
                    R["direct"][r["id"]] = max(r["dist"], key=r["dist"].get)
            for meth, rows in bl_rows(tag, d).items():
                for i, r in rows.items():
                    if r.get("choice") is not None:
                        R["bl_" + meth][i] = r["choice"]
            for i in ids_order:
                R["always_stop"][i] = "S"
                R["oracle"][i] = env[i]["exact"]["gold_choice"]
            for m, ch in R.items():
                ids = [i for i in ids_order if i in ch]
                res.setdefault(tag, {}).setdefault(d, {})[m] = summarize([score(env[i], ch[i]) for i in ids])
                # common subset: first nsub ids
                sub = [i for i in ids_order[:nsub] if i in ch]
                if len(sub) >= 0.8 * nsub:
                    res[tag][d][m + "@sub"] = summarize([score(env[i], ch[i]) for i in sub])
    _p = f"{IS}/results/bin_main.json"; _old = json.load(open(_p)) if os.environ.get("AGG_MERGE") and os.path.exists(_p) else {}; _old.update(res)
    json.dump(_old, open(_p, "w"), indent=1)
    return res


if __name__ == "__main__":
    tags = sys.argv[1].split(",")
    r = main(tags)
    for t in tags:
        for d in DOMS:
            row = r.get(t, {}).get(d, {})
            print(t, d, " ".join(f"{m}:{s['norm_regret']:.4f}(n{s['n']})" for m, s in sorted(row.items()) if "@" not in m))
