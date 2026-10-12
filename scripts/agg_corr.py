"""Aggregate the correlated-evidence controlled environments (scripts/corr_env.py, reviewer gap M3) into results/corr.json.
Per LLM tag, domain and method: first-decision DP regret (exact, from the TRUE joint model), myopic regret, stop rate, and
the share of first decisions that buy the correlated candidate in the two variants where its partner was already run
(dup_known, cc_known); per variant as well. DIVA / DIVA-DP use the LLM's factored specification (outputs/corr/facnum_*),
DIVA-U equals DIVA as every reliability is stated. LLM-free references: the factored optimum (stated numbers, independence
assumed; myopic and DP) and the joint optimum. A cell is complete when all 300 scenarios are scored."""
import os, sys, json, glob, collections
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from evalx import read, fac_scores, choose_max, score
from dcore import dp_value
from complete import KNOW_LLMS

IS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOMS = ["finance", "credit", "clinical", "legal"]
N = 300
BL = ["never", "cot_need", "sc_cot", "react", "finrisk_ask", "voi_prompt", "voi_paper", "bed_llm", "pal"]


def summ(env, ch):
    ids = [i for i in env if i in ch]
    rows = [(env[i], score(env[i], ch[i])) for i in ids]
    out = {"n": len(rows)}
    if not rows:
        return out
    out["dp_regret"] = float(np.mean([r["dp_regret"] for _, r in rows]))
    out["regret"] = float(np.mean([r["norm_regret"] for _, r in rows]))
    out["stop"] = float(np.mean([r["stopped"] for _, r in rows]))
    kn = [(s, ch[s["id"]]) for s, _ in rows if s["variant"] in ("dup_known", "cc_known")]
    out["buy_corr"] = float(np.mean([c == s["corr"]["redundant"] for s, c in kn])) if kn else float("nan")
    for v in ("dup_known", "cc_known", "dup_cand"):
        vr = [r for s, r in rows if s["variant"] == v]
        out["dp_" + v] = float(np.mean([r["dp_regret"] for r in vr])) if vr else float("nan")
    return out


def main():
    res = {"_meta": {"n": N, "llms": KNOW_LLMS}}
    for d in DOMS:
        env = {s["id"]: s for s in read(f"{IS}/envs_synth/data/corr/{d}_test.jsonl.gz")[:N]}
        ref = {"factored_myopic": {i: s["corr"]["naive_gold"] for i, s in env.items()},
               "factored_dp": {i: s["corr"]["naive_dp_first"] for i, s in env.items()},
               "joint_dp": {i: s["exact"]["dp_first_action"] for i, s in env.items()}}
        res.setdefault("_ref", {})[d] = {k: summ(env, v) for k, v in ref.items()}
        for t in KNOW_LLMS:
            R = collections.defaultdict(dict)
            for r in read(f"{IS}/outputs/corr/facnum_{t}_{d}.jsonl.gz"):
                sc = env.get(r["id"])
                if sc is None:
                    continue
                fs = fac_scores(sc, r)
                R["diva"][r["id"]] = choose_max(fs["net"])
                R["diva_u"][r["id"]] = R["diva"][r["id"]]
                R["always_top"][r["id"]] = max(fs["net"], key=fs["net"].get)
                keys = list(fs["par"]); v, a = dp_value(fs["b"], [fs["par"][k] for k in keys], float(sc["g"]), float(sc["l"]))
                R["diva_dp"][r["id"]] = "S" if a == "S" else keys[a]
            for r in read(f"{IS}/outputs/corr/dir_{t}_{d}.jsonl.gz"):
                if r["id"] in env and r.get("dist"):
                    R["direct"][r["id"]] = max(r["dist"], key=r["dist"].get)
            for m in BL:
                for r in read(f"{IS}/baselines/outputs_corr/{t}/{m}/{d}_test.jsonl.gz"):
                    if r["id"] in env and "error" not in r and r.get("choice") is not None:
                        R[m][r["id"]] = r["choice"]
            res.setdefault(t, {})[d] = {m: summ(env, ch) for m, ch in R.items()}
    json.dump(res, open(f"{IS}/results/corr.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    r = main()
    for d in DOMS:
        print(d, "ref", {k: round(100 * v["dp_regret"], 2) for k, v in r["_ref"][d].items()})
    done = collections.Counter()
    for t in KNOW_LLMS:
        for d in DOMS:
            for m, s in r[t][d].items():
                done[m] += s["n"] >= N
    print("complete (LLM x domain cells of 80):", dict(done))
