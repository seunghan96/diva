"""Robustness of the records gold to the reference model (R2-M7 / R4-M5): re-score every policy's first decision on credit and
clinical records under an ALTERNATIVE gold, an empirical naive-Bayes model fitted on the TRAIN records (emp_s1 / emp_s0 per
finding, train base rate), with exact DP over the remaining checks. No LLM is involved in either gold.
Writes results/alt_gold.json: {ds: {tag: {method: {"ref": dp_regret under the reference gold, "nb": under the NB gold, "n"}}}}"""
import sys, os, glob, json, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read, score, choose_max
from dcore import vstar, bayes, pred, posterior, dp_factory
from eval_tab import diva_scores, diva_u

BL = ["cot_need", "bed_llm", "voi_prompt", "pal"]
NMAX = int(os.environ.get("ALT_NMAX", "1000"))


def nb_qvals(sc):
    g, l = float(sc["g"]), float(sc["l"])
    clip = lambda p: min(max(float(p), 0.005), 0.995)
    b = posterior(float(sc["prior"]), [(clip(o["emp_s1"]), clip(o["emp_s0"]), int(o["value"])) for o in sc["observed"]])
    params = {c["label"]: (clip(c["emp_s1"]), clip(c["emp_s0"]), float(c["cost"])) for c in sc["checks"]}
    V = dp_factory(params, g, l)
    labs = list(params)
    q = {"S": vstar(b, g, l)}
    for k in labs:
        s1, s0, c = params[k]; p1 = pred(b, s1, s0); rest = tuple(j for j in labs if j != k)
        q[k] = -c + p1 * V(bayes(b, s1, s0, 1), rest)[0] + (1 - p1) * V(bayes(b, s1, s0, 0), rest)[0]
    return q, l


def main():
    out = collections.defaultdict(dict)
    for ds in ["taiwan", "german", "cdc", "heart"]:
        envl = read(f"{IS}/envs_tab/data/{ds}_test.jsonl.gz")[:NMAX]
        env = {s["id"]: s for s in envl}
        Q = {i: nb_qvals(sc) for i, sc in env.items()}
        for f in glob.glob(f"{IS}/outputs/test_tab/fk_*_{ds}.jsonl.gz"):
            tag = os.path.basename(f)[3:-9][: -(len(ds) + 1)]
            fr = {r["id"]: r for r in read(f)}
            dr = {r["id"]: max(r["dist"], key=r["dist"].get) for r in read(f"{IS}/outputs/test_tab/dir_{tag}_{ds}.jsonl.gz") if r.get("dist")}
            bls = {m: {r["id"]: r["choice"] for r in read(f"{IS}/baselines/outputs/{tag}/{m}/{ds}_test.jsonl.gz") if r.get("choice") is not None} for m in BL}
            ids = [i for i in env if i in fr and i in dr]
            if len(ids) < 200:
                continue
            R = collections.defaultdict(lambda: {"ref": [], "nb": []})
            for i in ids:
                sc = env[i]; fs = diva_scores(sc, fr[i], "cond", "greedy", "elicit", 0.75)
                ch = {"never": "S", "direct": dr[i], "diva": choose_max(fs["net"]), "diva_u": choose_max(diva_u(sc, fr[i], 1.0)["net"]),
                      "always_top": max(fs["net"], key=fs["net"].get)}
                for m in BL:
                    if i in bls[m]:
                        ch[m] = bls[m][i]
                q, l = Q[i]; best = max(q.values())
                for m, c in ch.items():
                    if c not in q:
                        continue
                    R[m]["ref"].append(score(sc, c)["dp_regret"]); R[m]["nb"].append((best - q[c]) / l)
            out[ds][tag] = {m: {"ref": float(np.mean(v["ref"])), "nb": float(np.mean(v["nb"])), "n": len(v["nb"])} for m, v in R.items()}
            print(ds, tag, {m: (round(100 * v["ref"], 2), round(100 * v["nb"], 2)) for m, v in out[ds][tag].items()}, flush=True)
    json.dump(out, open(f"{IS}/results/alt_gold.json", "w"), indent=1)


if __name__ == "__main__":
    main()
