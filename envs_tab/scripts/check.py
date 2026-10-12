"""Sanity checks + summary tables for envs_tab data (independent re-derivation of the gold from the stored joint)."""
import sys
sys.path.append("<path>")
import gzip, json, itertools
from collections import Counter
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss

B = "./envs_tab"
TOL = 1e-9


def load(p):
    return [json.loads(l) for l in gzip.open(p, "rt")]


def brute_dp(J, costs, g, l):
    """Independent DP over (observed index tuple, values) with explicit enumeration of the 64 configs."""
    cfgs = list(itertools.product([0, 1], repeat=6))  # cfg tuple (x_A..x_F); index = sum x_k << k
    idx = [sum(b << k for k, b in enumerate(c)) for c in cfgs]
    J = np.asarray(J).reshape(64, 2)
    U = lambda b: max(g * (1 - b) - l * b, 0.0)
    memo = {}

    def V(state):
        if state in memo:
            return memo[state]
        cons = [i for c, i in zip(cfgs, idx) if all(c[k] == v for k, v in state)]
        tot = J[cons].sum(); b = J[cons, 1].sum() / tot
        best = U(b)
        done = {k for k, _ in state}
        for k in range(6):
            if k in done:
                continue
            p1 = sum(J[i].sum() for c, i in zip(cfgs, idx) if c[k] == 1 and i in cons) / tot
            q = -costs[k] + p1 * V(tuple(sorted(state + ((k, 1),)))) + (1 - p1) * V(tuple(sorted(state + ((k, 0),))))
            best = max(best, q)
        memo[state] = best
        return best
    return V(())


def main():
    out = {}
    for ds in ["taiwan", "cdc", "german", "heart"]:
        for split in ["dev", "test"]:
            rows = load(f"{B}/data/{ds}_{split}.jsonl.gz")
            ex = [r["exact"] for r in rows]
            minvoi = min(min(e["voi"].values()) for e in ex)
            mono = all(e["dp_value"] >= e["myopic_value"] - 1e-9 >= e["value_of_stopping"] - 2e-9 for e in ex)
            rng = np.random.default_rng(0)
            errs = []
            for i in rng.choice(len(rows), 5, replace=False):
                r = rows[i]
                v = brute_dp(r["joint"], [r["costs"][L] for L in "ABCDEF"], r["g"], r["l"])
                errs.append(abs(v - r["exact"]["dp_value"]))
            y = np.array([r["y_true"] for r in rows]); p = np.array([e["posterior"] for e in ex])
            pdir = np.array([e["posterior_direct"] for e in ex])
            oth = Counter(("other_stop" if e["gold_choice"] == "S" else "other_acq") for e in ex if e["stratum"] == "other")
            res = {"n": len(rows), "min_voi": minvoi, "dp>=myopic>=stop": mono, "dp_bruteforce_maxerr": max(errs),
                   "root_belief_auc": roc_auc_score(y, p) if len(set(y)) == 2 else None,
                   "root_belief_brier": brier_score_loss(y, p), "direct_belief_brier": brier_score_loss(y, pdir),
                   "strata": dict(Counter(e["stratum"] for e in ex)), "other_breakdown": dict(oth),
                   "gold_stop": float(np.mean([e["gold_choice"] == "S" for e in ex])),
                   "dp_stop": float(np.mean([e["dp_first_action"] == "S" for e in ex])),
                   "myopic_dp_disagree": float(np.mean([e["myopic_dp_disagree"] for e in ex])),
                   "nonmyopic_acquire": float(np.mean([e["nonmyopic_acquire"] for e in ex])),
                   "n_placebo_scen": int(sum(any(c["is_placebo"] for c in r["checks"]) for r in rows)),
                   "cost_regimes": dict(Counter(r["cost_regime"] for r in rows))}
            out[f"{ds}_{split}"] = res
            print(ds, split, json.dumps(res, default=float))
    json.dump(out, open(f"{B}/data/checks_summary.json", "w"), indent=1, default=float)


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


def realized(ds, split="test"):
    """Realized utility on the actual records: DP-optimal policy under the reference joint vs stop-now vs myopic."""
    sys.path.insert(0, B + "/scripts")
    rows = load(f"{B}/data/{ds}_{split}.jsonl.gz")
    CFG = np.arange(64); BITS = (CFG[:, None] >> np.arange(6)) & 1
    res = {"stop_now": [], "dp_policy": [], "myopic_policy": [], "oracle_all_free": []}
    for r in rows:
        J = np.asarray(r["joint"]).reshape(64, 2); g, l = r["g"], r["l"]; y = r["y_true"]
        costs = [r["costs"][L] for L in "ABCDEF"]; vals = [c["record_value"] for c in r["checks"]]
        memo = {}

        def st(m, v):
            w = (CFG & m) == v; tot = J[w].sum(); return w, tot, J[w, 1].sum() / tot
        U = lambda b: max(g * (1 - b) - l * b, 0.0)

        def V(m, v):
            if (m, v) in memo: return memo[(m, v)]
            w, tot, b = st(m, v); best = (U(b), None)
            for i in range(6):
                if m >> i & 1: continue
                p1 = J[w & (BITS[:, i] == 1)].sum() / tot
                q = -costs[i] + p1 * V(m | 1 << i, v | 1 << i)[0] + (1 - p1) * V(m | 1 << i, v)[0]
                if q > best[0] + TOL: best = (q, i)
            memo[(m, v)] = best; return best

        def realize(policy):
            m = v = 0; spent = 0.0
            while True:
                i = policy(m, v)
                if i is None: break
                m |= 1 << i; v |= vals[i] << i; spent += costs[i]
            b = st(m, v)[2]
            act = g * (1 - b) - l * b > TOL
            return (g if y == 0 else -l) * act - spent

        def myopic(m, v):
            w, tot, b = st(m, v); v0 = U(b); best, arg = TOL, None
            for i in range(6):
                if m >> i & 1: continue
                p1 = J[w & (BITS[:, i] == 1)].sum() / tot
                b1 = J[w & (BITS[:, i] == 1), 1].sum() / J[w & (BITS[:, i] == 1)].sum()
                b0 = J[w & (BITS[:, i] == 0), 1].sum() / J[w & (BITS[:, i] == 0)].sum()
                net = p1 * U(b1) + (1 - p1) * U(b0) - v0 - costs[i]
                if net > best: best, arg = net, i
            return arg
        res["stop_now"].append(realize(lambda m, v: None))
        res["dp_policy"].append(realize(lambda m, v: V(m, v)[1]))
        res["myopic_policy"].append(realize(myopic))
        res["oracle_all_free"].append(realize(lambda m, v: next((i for i in range(6) if not m >> i & 1), None)) + sum(costs))
    out = {k: (float(np.mean(v)), float(np.std(v) / np.sqrt(len(v)))) for k, v in res.items()}
    d = np.array(res["dp_policy"]) - np.array(res["stop_now"])
    out["dp_minus_stop"] = (float(d.mean()), float(d.std() / np.sqrt(len(d))))
    return out


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "realized":
    allr = {}
    for ds in ["taiwan", "cdc", "german", "heart"]:
        allr[ds] = realized(ds)
        print(ds, json.dumps(allr[ds]))
    json.dump(allr, open(f"{B}/data/realized_utility_test.json", "w"), indent=1)
