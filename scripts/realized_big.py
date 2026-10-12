"""Higher-powered realized-utility replay on real records (reviewer S3, 2026-10-11).
Scenarios: envs_tab/data/{taiwan,cdc}_rz.jsonl.gz = original test (2,000) + fresh (1,000) + big (2,000) scenarios, each from
a distinct test record. Checks reveal the record's actual values; the final action is scored with the record's actual
outcome; utilities are normalized by l and net of costs (as in tab:realized).
Policies per LLM: DIVA and DIVA-U (frozen configuration, elicited specification from outputs/{test,fresh,big}_tab/fk_*),
Never (decide on the specification's own base belief, as in tab:realized), Direct (the LLM acquires and decides itself,
baselines/outputs_rz/{tag}/direct/{ds}_rz.episode.jsonl.gz; on the largest prefix of 5,000 / 3,000 / 2,000 scenarios that every
LLM completed, compared with Never on the same prefix). LLM-free: the reference model's DP policy.
CIs: clustered bootstrap (clusters = scenario group, 1,000 resamples) of the difference to Never.
A cell is reported only when every scenario of the set is present for that LLM (else None -> TBD).
Writes results/realized_big.json."""
import sys, os, json, gzip
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read
from eval_tab import diva_scores
from realized_tab import run_diva
from complete import KNOW_LLMS

DSS = ["taiwan", "cdc"]
TOL = 1e-9
CFG = np.arange(64); BITS = (CFG[:, None] >> np.arange(6)) & 1


def boot(diffs, clusters, B=1000, seed=0):
    """clustered bootstrap CI of the mean (same estimator as extra_controls.boot_ci, vectorized)"""
    rng = np.random.default_rng(seed)
    keys = {}
    for c in clusters:
        keys.setdefault(c, len(keys))
    k = np.array([keys[c] for c in clusters]); K = len(keys)
    s = np.bincount(k, weights=np.asarray(diffs, float), minlength=K); n = np.bincount(k, minlength=K).astype(float)
    idx = rng.integers(0, K, (B, K))
    st = s[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    return float(np.mean(diffs)), float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))


def ref_stop(r):
    """realized utility (normalized by l) of deciding at once with the reference model's own belief (no acquisition)"""
    J = np.asarray(r["joint"]).reshape(64, 2); g, l = float(r["g"]), float(r["l"]); y = int(r["y_true"])
    b = J[:, 1].sum() / J.sum()
    return ((g if y == 0 else -l) if g * (1 - b) - l * b > TOL else 0.0) / l


def ref_dp(r):
    """realized utility (normalized by l, net of costs) of the reference model's DP policy (envs_tab/scripts/check.py)"""
    J = np.asarray(r["joint"]).reshape(64, 2); g, l = float(r["g"]), float(r["l"]); y = int(r["y_true"])
    labs = [c["label"] for c in r["checks"]]
    costs = [float(r["costs"][L]) for L in labs]; vals = [int(c["record_value"]) for c in r["checks"]]
    memo = {}

    def st(m, v):
        w = (CFG & m) == v; tot = J[w].sum(); return w, tot, J[w, 1].sum() / tot
    U = lambda b: max(g * (1 - b) - l * b, 0.0)

    def V(m, v):
        if (m, v) in memo:
            return memo[(m, v)]
        w, tot, b = st(m, v); best = (U(b), None)
        for i in range(6):
            if m >> i & 1:
                continue
            p1 = J[w & (BITS[:, i] == 1)].sum() / tot
            q = -costs[i] + p1 * V(m | 1 << i, v | 1 << i)[0] + (1 - p1) * V(m | 1 << i, v)[0]
            if q > best[0] + TOL:
                best = (q, i)
        memo[(m, v)] = best; return best
    m = v = 0; spent = 0.0
    while True:
        i = V(m, v)[1]
        if i is None:
            break
        m |= 1 << i; v |= vals[i] << i; spent += costs[i]
    b = st(m, v)[2]
    act = g * (1 - b) - l * b > TOL
    return (((g if y == 0 else -l) if act else 0.0) - spent) / l


def fk_rows(tag, ds):
    out = {}
    for part in ("test_tab", "fresh_tab", "big_tab"):
        p = f"{IS}/outputs/{part}/fk_{tag}_{ds}.jsonl.gz"
        if os.path.exists(p):
            for r in read(p):
                out[r["id"]] = r
    return out


def main():
    res = {"ref": {}, "cells": {}}
    for ds in DSS:
        env = read(f"{IS}/envs_tab/data/{ds}_rz.jsonl.gz")
        ids = [s["id"] for s in env]; E = {s["id"]: s for s in env}
        # every policy is evaluated on the largest scenario prefix (5,000, or the 3,000 test + fresh scenarios) whose
        # specification every LLM completed; the file lists test, fresh, then the new scenarios
        FR = {t: fk_rows(t, ds) for t in KNOW_LLMS}
        KS = next((k for k in (len(ids), 3000) if all(all(i in FR[t] for i in ids[:k]) for t in KNOW_LLMS)), 0)
        ids = ids[:KS]
        cl = [E[i].get("group_id") or i for i in ids]
        R = [ref_dp(E[i]) for i in ids]; R0 = [ref_stop(E[i]) for i in ids]
        res["ref"][ds] = dict(n=len(ids), ref_dp=float(np.mean(R)) if R else None, ref_stop=float(np.mean(R0)) if R0 else None,
                              ci_ref_vs_ref_stop=boot([a - b for a, b in zip(R, R0)], cl) if R else None)
        EP = {}
        for tag in KNOW_LLMS:
            ep = {}
            p = f"{IS}/baselines/outputs_rz/{tag}/direct/{ds}_rz.episode.jsonl.gz"
            if os.path.exists(p):
                for r in read(p):
                    if "error" not in r and r.get("decision") is not None:
                        ep[r["id"]] = r
            EP[tag] = ep
        # Direct (LLM in the loop) is reported on the largest scenario prefix (5,000 / 3,000 / 2,000) every LLM completed
        KD = next((k for k in sorted({len(ids), 3000, 2000}, reverse=True) if k <= len(ids) and all(all(i in EP[t] for i in ids[:k]) for t in KNOW_LLMS)), 0)
        res["ref"][ds]["n_direct_set"] = KD
        for tag in KNOW_LLMS:
            fr = FR[tag]; ep = EP[tag]
            cell = dict(n=len(ids), n_spec=sum(i in fr for i in ids), n_direct=sum(i in ep for i in ids))
            if ids and cell["n_spec"] == len(ids):
                D, DU, S = [], [], []
                for i in ids:
                    sc = E[i]; fs = diva_scores(sc, fr[i], "cond", "greedy", "elicit", 0.75)
                    D.append(run_diva(sc, fs)[0]); DU.append(run_diva(sc, fs, unc=1.0)[0])
                    g, l = float(sc["g"]), float(sc["l"]); b = fs["b"]; y = int(sc["y_true"])
                    S.append(((g if y == 0 else -l) if (1 - b) * g - b * l > 0 else 0.0) / l)
                cell.update(diva=float(np.mean(D)), diva_u=float(np.mean(DU)), never=float(np.mean(S)),
                            ci_diva=boot([a - b for a, b in zip(D, S)], cl), ci_diva_u=boot([a - b for a, b in zip(DU, S)], cl),
                            ci_ref=boot([a - b for a, b in zip(R, S)], cl))
                if KD:
                    Dr = []; dids = ids[:KD]; Sd = S[:KD]; cld = cl[:KD]
                    for i in dids:
                        sc, r = E[i], ep[i]; g, l = float(sc["g"]), float(sc["l"]); y = int(sc["y_true"])
                        act = r["decision"] == sc["actions"][0]
                        Dr.append((((g if y == 0 else -l) if act else 0.0) - float(r["cost"])) / l)
                    cell.update(direct=float(np.mean(Dr)), never_dset=float(np.mean(Sd)), ci_direct=boot([a - b for a, b in zip(Dr, Sd)], cld),
                                direct_acq=float(np.mean([ep[i]["n_acq"] for i in dids])), n_direct_set=KD)
            res["cells"][f"{tag}:{ds}"] = cell
            print(tag, ds, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in cell.items() if not k.startswith("ci")}, flush=True)
    json.dump(res, open(f"{IS}/results/realized_big.json", "w"), indent=1)


if __name__ == "__main__":
    main()
