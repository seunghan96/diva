"""When does the LLM's knowledge help? Learning curve on knowledge-required records (envs_tab test):
likelihoods P(finding | Y) estimated from n random TRAINING records with a Beta prior that is either
 (a) the LLM-elicited likelihood (DIVA prior, frozen config: conditional form, lambda 0.75) with pseudo-count m per class, or
 (b) uninformative (Laplace, m=2 at 0.5),
then the same decision-value acquisition. n in {0, 10, 25, 50, 100, 200, 1000, all}; 5 seeds per n. Base rate = stated.
Writes results/learning_curve.json."""
import sys, os, json, glob, gzip, ast, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read, score, summarize, choose_max, clip
from eval_tab import diva_scores
from dcore import voi_exact, posterior
import pandas as pd

NS = [0, 10, 25, 50, 100, 200, 1000, -1]
M_LLM = 10.0


def colmap(ds):
    meta = json.load(open(f"{IS}/envs_tab/clean/{ds}_meta.json"))
    f = meta["findings"]; f = ast.literal_eval(f) if isinstance(f, str) else f
    return {v["name"]: k for k, v in f.items()}, meta


def main():
    out = {}; gen_done = set()
    CS = json.load(open(f"{IS}/results/const_strength.json")) if os.path.exists(f"{IS}/results/const_strength.json") else {}
    for f in sorted(glob.glob(f"{IS}/outputs/test_tab/fk_*.jsonl.gz")):
        tag, ds = os.path.basename(f)[3:-9].rsplit("_", 1)
        env = {s["id"]: s for s in read(f"{IS}/envs_tab/data/{ds}_test.jsonl.gz")}
        fr = {r["id"]: r for r in read(f)}
        ids = [i for i in env if i in fr][:1000]
        if len(ids) < 200:
            continue
        cm, meta = colmap(ds)
        df = pd.read_csv(f"{IS}/envs_tab/clean/{ds}.csv.gz")
        tr = df[df["split"] == "train"].reset_index(drop=True)
        res = {}
        for n in NS:
            kinds = ("llm", "flat") + (("generic",) if ds not in gen_done else ())
            for prior_kind in kinds:
                regs = []
                for seed in range(5 if n not in (0, -1) else 1):
                    sub = tr if n == -1 else (tr.sample(n=min(n, len(tr)), random_state=seed) if n > 0 else tr.iloc[:0])
                    y = sub["y"].values
                    rows = []
                    for i in ids:
                        sc = env[i]
                        base = diva_scores(sc, fr[i], "cond", "greedy", "elicit", 0.75)
                        par = {}
                        names = [(c["label"], c["name"]) for c in sc["checks"]] + [(("obs", k), o["name"]) for k, o in enumerate(sc.get("observed", []))]
                        llm_obs = {}
                        if prior_kind == "llm":
                            # LLM likelihoods for observed findings: recompute from the record format
                            from eval_tab import lik_pair, temper
                            for k, rec in enumerate(fr[i].get("obs", [])):
                                lp = lik_pair(rec, "cond", "greedy")
                                llm_obs[k] = temper(lp, 0.75) if lp else (0.5, 0.5)
                        for key, name in names:
                            col = cm.get(name)
                            x = sub[col].values if (col is not None and len(sub)) else np.array([])
                            k1 = float(np.nansum(x[y == 1])) if len(x) else 0.0; n1 = float(np.sum(~np.isnan(x[y == 1]))) if len(x) else 0.0
                            k0 = float(np.nansum(x[y == 0])) if len(x) else 0.0; n0 = float(np.sum(~np.isnan(x[y == 0]))) if len(x) else 0.0
                            if prior_kind == "llm":
                                p1, p0 = (base["par"][key][:2] if not isinstance(key, tuple) else llm_obs.get(key[1], (0.5, 0.5)))
                                m = M_LLM
                            elif prior_kind == "generic":
                                dlt = CS.get(ds, {}).get("delta", 1.0)
                                p1, p0, m = 1 / (1 + np.exp(-dlt / 2)), 1 / (1 + np.exp(dlt / 2)), M_LLM
                            else:
                                p1, p0, m = 0.5, 0.5, 2.0
                            par[key] = (clip((k1 + m * p1) / (n1 + m)), clip((k0 + m * p0) / (n0 + m)))
                        b = posterior(float(sc["prior"]), [(par[("obs", k)][0], par[("obs", k)][1], int(o["value"])) for k, o in enumerate(sc.get("observed", []))])
                        net = {c["label"]: voi_exact(b, par[c["label"]][0], par[c["label"]][1], float(sc["g"]), float(sc["l"])) - float(c["cost"]) for c in sc["checks"]}
                        rows.append(score(sc, choose_max(net)))
                    regs.append(summarize(rows)["dp_regret"])
                res[f"{prior_kind}:{n}"] = float(np.mean(regs))
        res["never"] = summarize([score(env[i], "S") for i in ids])["dp_regret"]
        gen_done.add(ds)
        out[f"{tag}:{ds}"] = res
        print(tag, ds, {k: round(v, 4) for k, v in res.items()}, flush=True)
    json.dump(out, open(f"{IS}/results/learning_curve.json", "w"), indent=1)


if __name__ == "__main__":
    main()
