"""cost-discounted DIVA (DIVA-kappa). A check is acquired if VOI_point(e) > kappa * c_e, with kappa
selected per (LLM, environment) on that LLM's DEV elicitation (dev labels used, like the tuned Fac-EIG baseline).
This tests whether DIVA-U's gain is equivalent to a tuned acquisition bonus. Writes results/kappa.json:
  {"tag:env": {"kappa": k, "dev": {k: ratio}}}"""
import sys, os, glob, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read, score, summarize, choose_max
from eval_tab import diva_scores
from eval_real import fio_real

KAPPAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
K = "dp_regret"


def kappa_net(sc, fs, kappa):
    cost = {c["label"]: float(c["cost"]) for c in sc["checks"]}
    return {e: v + cost.get(e, 0.0) * (1 - kappa) for e, v in fs["net"].items()}


def kappa_choice(sc, fs, kappa):
    return choose_max(kappa_net(sc, fs, kappa))


BETAS = [0.0, 0.01, 0.02, 0.04, 0.08, 0.16]


def beta_choice(sc, fs, beta):
    """Additive acquisition bonus: acquire if VOI + beta > c (uniform bonus, can trigger acquisitions of zero-VOI checks)."""
    return choose_max({e: v + beta for e, v in fs["net"].items()})


def tune_beta(pairs):
    res = {b: summarize([score(sc, beta_choice(sc, fs, b)) for sc, fs in pairs])[K] for b in BETAS}
    return min(BETAS, key=lambda b: (round(res[b], 6), b)), res


def tune(pairs):
    res = {}
    for k in KAPPAS:
        res[k] = summarize([score(sc, kappa_choice(sc, fs, k)) for sc, fs in pairs])[K]
    best = min(KAPPAS, key=lambda k: (round(res[k], 6), abs(k - 1.0)))
    return best, res


def main():
    out = {}
    for ds in ["taiwan", "german", "cdc", "heart"]:
        dev = {s["id"]: s for s in read(f"{IS}/envs_tab/data/{ds}_dev.jsonl.gz")}
        for f in glob.glob(f"{IS}/outputs/tab_dev/fk_*_{ds}.jsonl.gz"):
            tag = os.path.basename(f)[3:-9][: -(len(ds) + 1)]
            dfr = {r["id"]: r for r in read(f)}
            pairs = [(dev[i], diva_scores(dev[i], dfr[i], "cond", "greedy", "elicit", 0.75)) for i in dev if i in dfr]
            if len(pairs) < 100:
                continue
            k, r = tune(pairs); bb, rb = tune_beta(pairs)
            out[f"{tag}:{ds}"] = {"kappa": k, "dev": r, "beta": bb, "dev_beta": rb}
            print(tag, ds, k, flush=True)
    for tg in ["ret_1", "ret_5", "vol_5", "vol_20"]:
        dname = "ret1" if tg == "ret_1" else tg
        dev = {s["id"]: s for s in read(f"{IS}/envs_real/data/{tg}_dev.jsonl.gz")}
        for f in glob.glob(f"{IS}/outputs/real_dev/fr_*_{dname}.jsonl.gz"):
            tag = os.path.basename(f)[3:-9][: -(len(dname) + 1)]
            if "_" in tag and tag.split("_")[-1] in ("joint", "ratio", "contrast"):
                continue
            dfr = {r["id"]: r for r in read(f)}
            pairs = [(dev[i], fio_real(dev[i], dfr[i], lam=0.75)) for i in dev if i in dfr]
            if len(pairs) < 100:
                continue
            k, r = tune(pairs); bb, rb = tune_beta(pairs)
            out[f"{tag}:{tg}"] = {"kappa": k, "dev": r, "beta": bb, "dev_beta": rb}
            print(tag, tg, k, flush=True)
    json.dump(out, open(f"{IS}/results/kappa.json", "w"), indent=1)


if __name__ == "__main__":
    main()
