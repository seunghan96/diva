"""Independent correctness checks of the exact labels (no shared code path with core.py's vectorised VOI).

  1. VOI / EIG recomputed with plain-Python loops over states and outcomes (Bayes rule written out).
  2. Monte Carlo check of VOI (sample state -> outcome -> posterior decision) within 4 standard errors.
  3. Invariants: VOI >= 0; placebo items have VOI = EIG = 0; items about a decision-irrelevant factor (ms track)
     have VOI = 0 exactly; DP value >= max(stop value, myopic value); stratum S2 => gold stop.
  4. MS track: utilities recomputed from the stated text numbers (W, const) and the stated priors.
  5. eval_api: the oracle policy (follow DP, decide Bayes action) gets policy_regret = 0 on every episode.
usage: python selftest.py data/ms/finance_dev.jsonl.gz [more files] [--n 200]
"""
import argparse, gzip, itertools, json, math, random, sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from core import Model
from eval_api import evaluate


def slow_voi_eig(m, obs):
    K = len(m["prior"])
    w = list(m["prior"])
    for i, o in obs:
        w = [w[s] * m["items"][i]["lik"][s][o] for s in range(K)]
    Z = sum(w)
    b = [x / Z for x in w]
    A = len(m["utility"])
    eu0 = max(sum(m["utility"][a][s] * b[s] for s in range(K)) for a in range(A))
    H = lambda p: -sum(x * math.log2(x) for x in p if x > 0)
    done = {i for i, _ in obs}
    out = {}
    for i, it in enumerate(m["items"]):
        if i in done:
            continue
        O = len(it["lik"][0])
        v, e = 0.0, H(b)
        for o in range(O):
            po = sum(b[s] * it["lik"][s][o] for s in range(K))
            if po <= 0:
                continue
            bo = [b[s] * it["lik"][s][o] / po for s in range(K)]
            v += po * max(sum(m["utility"][a][s] * bo[s] for s in range(K)) for a in range(A))
            e -= po * H(bo)
        out[it["label"]] = (v - eu0, e)
    return out


def mc_voi(m, obs, i, rng, n=4000):
    M = Model(m)
    b = M.belief(obs)
    v0 = (M.U @ b).max()
    vals = []
    for _ in range(n):
        s = rng.choice(len(b), p=b)
        L = M.L[i][s]
        o = rng.choice(len(L), p=L / L.sum())
        bo = M.belief(list(obs) + [(i, o)])
        a = int(np.argmax(M.U @ bo))
        vals.append(M.U[a, s])
    vals = np.array(vals)
    return vals.mean() - v0, vals.std() / math.sqrt(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--mc", type=int, default=20)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    fails = []
    stats = dict(rows=0, items=0, mc=0, oracle_eps=0)
    for path in a.files:
        rows = [json.loads(l) for l in gzip.open(path, "rt")]
        random.Random(0).shuffle(rows)
        for r in rows[:a.n]:
            stats["rows"] += 1
            m, ex = r["model"], r["exact"]
            obs = [tuple(x) for x in m.get("observed", [])]
            slow = slow_voi_eig(m, obs)
            for lab in r["label_order"]:
                stats["items"] += 1
                sv, se = slow[lab]
                if abs(sv - ex["voi"][lab]) > 1e-6 * max(1, abs(sv)) or abs(se - ex["eig"][lab]) > 1e-6:
                    fails.append((r["id"], lab, "slow_mismatch", sv, ex["voi"][lab], se, ex["eig"][lab]))
                if ex["voi"][lab] < 0:
                    fails.append((r["id"], lab, "neg_voi"))
            for c in r["checks"]:
                if c.get("is_placebo") and (ex["voi"][c["label"]] != 0 or ex["eig"][c["label"]] > 1e-9):
                    fails.append((r["id"], c["label"], "placebo_nonzero"))
                if r["track"] == "ms" and not c["decision_relevant_factor"] and ex["voi"][c["label"]] != 0:
                    fails.append((r["id"], c["label"], "irrelevant_factor_voi_nonzero", ex["voi"][c["label"]]))
            if ex["dp_value"] < max(ex["value_of_stopping"], ex["myopic_value"]) - 1e-9:
                fails.append((r["id"], "dp_below_myopic"))
            if r["stratum"] == "S2" and ex["gold_choice"] != "S":
                fails.append((r["id"], "S2_not_stop"))
            if r["track"] == "ms":
                # recompute utilities from stated components
                st = m["states"]
                keys = [f["key"] for f in r["factors"]]
                for ai, act in enumerate(r["actions"]):
                    for si, s in enumerate(st):
                        u = r["const"][act] + sum(r["W"][act][keys[f]][s[f]] for f in range(len(keys)))
                        if abs(u - m["utility"][ai][si]) > 1e-9:
                            fails.append((r["id"], "utility_mismatch"))
                pr = [float(np.prod([r["factors"][f]["prior"][s[f]] for f in range(len(keys))])) for s in st]
                if max(abs(x - y) for x, y in zip(pr, m["prior"])) > 1e-12:
                    fails.append((r["id"], "prior_mismatch"))
            # MC check on a few rows
            if stats["mc"] < a.mc:
                lab = max(ex["voi"], key=ex["voi"].get)
                i = r["label_order"].index(lab)
                est, se = mc_voi(m, obs, i, rng)
                stats["mc"] += 1
                if abs(est - ex["voi"][lab]) > 4 * se + 1e-9:
                    fails.append((r["id"], lab, "mc_mismatch", est, se, ex["voi"][lab]))
            # oracle policy along each episode: regret must be 0
            M = Model(m)
            for e in range(len(r["episodes"])):
                ep = r["episodes"][e]
                o = list(M.observed0)
                reqs = []
                while True:
                    f = M.dp_first(o)
                    if f["move"] == "S":
                        break
                    lab = M.labels[f["item"]]
                    reqs.append(lab)
                    o.append((f["item"], ep["outcomes"][lab]))
                res = evaluate(r, reqs, r["actions"][f["action"]], e)
                stats["oracle_eps"] += 1
                if abs(res["policy_regret"]) > 1e-6:
                    fails.append((r["id"], e, "oracle_regret", res["policy_regret"]))
    print(json.dumps({"stats": stats, "n_fail": len(fails), "fails": fails[:20]}, default=str))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
