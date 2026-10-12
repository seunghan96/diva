"""Evaluate DIVA on the multi-state track: build the LLM-specified model, compute VOI/EIG/DP with envs_synth's exact engine.
usage: eval_ms.py ENV MS_OUT [DIRECT_OUT]"""
import sys, os, json, itertools, collections, importlib.util
import numpy as np
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from evalx import read, score, summarize, choose_max, spearman
spec = importlib.util.spec_from_file_location("es_core", os.path.join(HERE, "..", "envs_synth", "core.py"))
es = importlib.util.module_from_spec(spec); spec.loader.exec_module(es)


LAST_SPEC = [None]


def norm(v, n):
    if v is None or len(v) != n:
        return np.ones(n) / n, 1
    v = np.clip(np.asarray(v, dtype=float), 0, None)
    if v.sum() <= 0:
        return np.ones(n) / n, 1
    v = np.clip(v / v.sum(), 0.005, None)
    return v / v.sum(), 0


def build(sc, fr, stated_utility=True):
    F = sc["factors"]
    levels = [len(f["levels"]) for f in F]
    states = list(itertools.product(*[range(n) for n in levels]))
    fb = 0
    pri = []
    for f in F:
        p, e = norm(fr["prior"].get(f["key"]), len(f["levels"])); fb += e; pri.append(p)
    prior = np.array([np.prod([pri[k][s[k]] for k in range(len(F))]) for s in states])
    U = []
    for a in sc["actions"]:
        row = []
        cst = fr["c"].get(a); cst = cst[0] if cst else 0.0
        for s in states:
            u = cst
            for k, f in enumerate(F):
                w = fr["u"].get(a, {}).get(f["key"])
                if w is None or len(w) != len(f["levels"]):
                    fb += 1; w = [0.0] * len(f["levels"])
                u += w[s[k]]
            row.append(u)
        U.append(row)
    fidx = {f["key"]: k for k, f in enumerate(F)}
    items = []
    for c in sc["checks"]:
        no = len(c["outcomes"])
        k = fidx.get(c.get("factor"))
        rows = []
        for s in states:
            if k is None:
                rows.append(list(np.ones(no) / no)); continue
            v, e = norm(fr["lik"].get(c["label"], {}).get(str(s[k])), no); fb += e
            rows.append(list(v))
        items.append({"label": c["label"], "lik": rows, "cost": float(c["cost"]), "outcomes": c["outcomes"]})
    if stated_utility:  # payoff table is part of the task specification (as g, l in the binary tracks)
        U = sc["model"]["utility"]
    m = {"prior": list(prior / prior.sum()), "utility": U, "items": items, "budget": sc.get("budget")}
    LAST_SPEC[0] = m
    return es.Model(m), fb


def _h(p):
    p = np.asarray(p, dtype=float); p = p[p > 1e-12]
    return float(-(p * np.log2(p)).sum())


def entropy_search(sc, M_spec):
    """Entropy search over the optimal action (Truong et al.): expected reduction of the entropy of argmax_a U(a, state)
    under the specified model; returns {label: bits}."""
    pri = np.asarray(M_spec["prior"], dtype=float); U = np.asarray(M_spec["utility"], dtype=float)
    astar = U.argmax(0)  # optimal action per state
    nA = U.shape[0]
    def ha(post):
        return _h([post[astar == a].sum() for a in range(nA)])
    h0 = ha(pri); out = {}
    for it in M_spec["items"]:
        L = np.asarray(it["lik"], dtype=float)  # states x outcomes
        pv = pri @ L; eh = 0.0
        for o in range(L.shape[1]):
            if pv[o] > 1e-12:
                eh += pv[o] * ha(pri * L[:, o] / pv[o])
        out[it["label"]] = h0 - eh
    return out


def run(env_p, fr_p, dir_p=None, eig_tau=0.05):
    env = {s["id"]: s for s in read(env_p)}
    fr = {r["id"]: r for r in read(fr_p)}
    dr = {r["id"]: r for r in read(dir_p)} if dir_p else {}
    res = collections.defaultdict(list); rho = collections.defaultdict(list); fbn = 0
    for i, sc in env.items():
        if i not in fr:
            continue
        M, fb = build(sc, fr[i]); fbn += fb > 0
        labs = [c["label"] for c in sc["checks"]]
        net, eig = {}, {}
        for j, lab in enumerate(labs):
            st = M.item_stats([], j)
            net[lab] = st[0] - M.cost[j]; eig[lab] = st[1]
        res["diva"].append(score(sc, choose_max(net)))
        try:
            d = M.dp_first([])
            a = "S" if d["move"] == "S" else labs[d["item"]]
        except Exception:
            a = choose_max(net)
        res["diva_dp"].append(score(sc, a))
        res["fac_eig"].append(score(sc, choose_max(eig, eig_tau)))
        esv = entropy_search(sc, LAST_SPEC[0])
        res["fac_es"].append(score(sc, choose_max(esv, eig_tau)))
        rho["diva"].append(spearman([net[k] for k in labs], [sc["exact"]["net"][k] for k in labs]))
        rho["fac_eig"].append(spearman([eig[k] for k in labs], [sc["exact"]["net"][k] for k in labs]))
        if i in dr and dr[i].get("dist"):
            d = dr[i]["dist"]; res["direct"].append(score(sc, max(d, key=d.get)))
        res["always_stop"].append(score(sc, "S"))
    out = {m: summarize(r) | {"rho": float(np.nanmean(rho[m])) if rho[m] else float("nan")} for m, r in res.items()}
    out["_fallback_scen"] = fbn
    return out


if __name__ == "__main__":
    o = run(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    print("fallback scenarios:", o.pop("_fallback_scen"))
    for m, s in o.items():
        print(f"{m:12s} n={s['n']} regret={s['norm_regret']:.4f} dpreg={s.get('dp_regret', float('nan')):.4f} capt={s.get('captured', float('nan')):.2f} top1={s['top1']:.2f} over={s['over_acq']:.2f} under={s['under_acq']:.2f} plac={s['placebo']:.3f} rho={s['rho']:.2f}")
