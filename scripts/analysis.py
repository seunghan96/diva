"""Analyses on the controlled binary track (CPU only). Writes results/analysis_*.json.
1) decomp: DIVA regret when replacing elicited prior / likelihoods with the true ones (error decomposition).
2) strata: DIVA / Fac-EIG / direct regret, over-acquisition and trap rates per stratum and family (cost/payoff variants).
3) incoherence: raw imagined-belief diagnostics from outputs/probe (corr of elicited vs exact belief change,
   total-probability violation).
4) elicit_err: mean absolute error of elicited numbers vs stated values, per model."""
import sys, os, json, glob, collections, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read, fac_scores, choose_max, score, summarize
from dcore import posterior, voi_exact, eig_exact, bayes

DOMS = ["finance", "credit", "clinical", "legal"]


def envs():
    E = {}
    for d in DOMS:
        for s in read(f"{IS}/envs_synth/data/binary/{d}_test.jsonl.gz"):
            E[s["id"]] = s
    return E


def decomp(E):
    # fact-check 10-12: pool the four domain files of each LLM (previously each file overwrote the LLM's entry, so a row
    # reflected only the domain read last) and count every scenario once
    out = {}
    RT = collections.defaultdict(lambda: collections.defaultdict(list)); seen = collections.defaultdict(set)
    for f in sorted(glob.glob(f"{IS}/outputs/test_bin/facnum_*.jsonl.gz")):
        tag = os.path.basename(f)[7:-9].rsplit("_", 1)[0]
        R = RT[tag]
        for r in read(f):
            sc = E.get(r["id"])
            if sc is None or r["id"] in seen[tag]:
                continue
            seen[tag].add(r["id"])
            true_chk = {c["label"]: [c["s1"], c["s0"]] for c in sc["checks"]}
            true_obs = [[o["s1"], o["s0"]] for o in sc.get("observed", [])]
            variants = {"elicited": r,
                        "true_prior": dict(r, prior=sc["prior"]),
                        "true_lik": dict(r, obs=true_obs, chk=true_chk),
                        "true_all": dict(r, prior=sc["prior"], obs=true_obs, chk=true_chk)}
            for k, v in variants.items():
                fs = fac_scores(sc, v)
                R[k].append(score(sc, choose_max(fs["net"])))
            # elicitation error
            errs = [abs((r["prior"] or 0.5) - sc["prior"])]
            for c in sc["checks"]:
                e = r["chk"].get(c["label"]) or [None, None]
                for x, t in zip(e, [c["s1"], c["s0"]]):
                    if x is not None:
                        errs.append(abs(x - t))
            R["_err"].append({"regret": float(np.mean(errs)), "stop_gold": 0, "stopped": 0, "top1": 0, "placebo": 0, "norm_regret": float(np.mean(errs))})
    for tag, R in RT.items():
        out.setdefault(tag, {})
        for k, rows in R.items():
            s = summarize(rows)
            out[tag][k] = {"norm_regret": s["norm_regret"], "dp_regret": s.get("dp_regret"), "n": s["n"]}
    return out


def strata(E):
    res = json.load(open(f"{IS}/results/bin_main.json")) if os.path.exists(f"{IS}/results/bin_main.json") else {}
    out = {}
    for f in glob.glob(f"{IS}/outputs/test_bin/facnum_*.jsonl.gz"):
        tag = os.path.basename(f)[7:-9].rsplit("_", 1)[0]
        dr = {}
        for d in DOMS:
            for r in read(f"{IS}/outputs/test_dir/dir_{tag}_{d}.jsonl.gz"):
                if r.get("dist"):
                    dr[r["id"]] = max(r["dist"], key=r["dist"].get)
        R = collections.defaultdict(list)
        for r in read(f):
            sc = E.get(r["id"])
            if sc is None:
                continue
            fs = fac_scores(sc, r)
            keys = [sc["stratum"], "fam:" + sc.get("family", "?"), "cost:" + str(sc.get("cost_regime"))]
            for key in keys:
                R[(key, "diva")].append(score(sc, choose_max(fs["net"])))
                R[(key, "fac_eig")].append(score(sc, choose_max(fs["eig"], 0.05)))
                if r["id"] in dr:
                    R[(key, "direct")].append(score(sc, dr[r["id"]]))
                R[(key, "always_stop")].append(score(sc, "S"))
        out[tag] = {f"{k}|{m}": summarize(v) for (k, m), v in R.items()}
    return out


def incoherence(E):
    out = {}
    for f in glob.glob(f"{IS}/outputs/probe/probe_*.jsonl.gz"):
        tag = os.path.basename(f)[6:-9]
        D, DE, viol, B0, P = [], [], [], [], []
        for pr in read(f):
            sc = E.get(pr["id"])
            if sc is None or not pr.get("b0"):
                continue
            b = sc["exact"]["posterior"]; B0.append(pr["b0"]["b"]); P.append(b)
            for c in sc["checks"]:
                bv = pr["bv"].get(c["label"])
                if not bv or not bv["1"] or not bv["0"]:
                    continue
                d = bv["1"]["b"] - bv["0"]["b"]; de = bayes(b, c["s1"], c["s0"], 1) - bayes(b, c["s1"], c["s0"], 0)
                D.append(d); DE.append(de)
                p1 = b * c["s1"] + (1 - b) * c["s0"]
                viol.append(abs(p1 * bv["1"]["b"] + (1 - p1) * bv["0"]["b"] - pr["b0"]["b"]))
        if len(D) > 10:
            out[tag] = dict(corr_update=float(np.corrcoef(D, DE)[0, 1]), slope=float(np.polyfit(DE, D, 1)[0]),
                            mean_abs_update_llm=float(np.mean(np.abs(D))), mean_abs_update_exact=float(np.mean(np.abs(DE))),
                            corr_belief=float(np.corrcoef(B0, P)[0, 1]), tp_violation=float(np.mean(viol)), n=len(D))
    return out


if __name__ == "__main__":
    E = envs()
    which = sys.argv[1:] or ["decomp", "strata", "incoherence"]
    for w in which:
        o = globals()[w](E)
        json.dump(o, open(f"{IS}/results/analysis_{w}.json", "w"), indent=1)
        print(w, "done", len(o))
