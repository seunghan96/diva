"""Shared helpers of the reviewer analyses F8 (an_f8_*.py): exact DP values of a knowledge-required scenario under its
reference model (tabular records: stored reference joint; SEC filings: stored reference naive-Bayes model) for rescaled
check costs and perturbed payoffs, so that first decisions can be scored against the DP optimum of any cost regime.
Reproduces sc["exact"]["q"], ["dp_value"] and ["value_of_stopping"] at cost scale 1 (checked in an_f8_costsweep.py)."""
import ast
from functools import lru_cache
import numpy as np

M6 = 6


def true_model(sc):
    """(post(acq) -> P(Y=1 | acq), outcomes(acq, k) -> [(v, P(v | acq))])"""
    if "joint" in sc:
        J = np.asarray(sc["joint"], dtype=float).reshape(2 ** M6, 2)
        BITS = (np.arange(2 ** M6)[:, None] >> np.arange(M6)) & 1
        idx = {L: i for i, L in enumerate(sc.get("label_order") or [c["label"] for c in sc["checks"]])}

        def mass(acq):
            w = np.ones(2 ** M6, bool)
            for k, v in acq:
                w &= BITS[:, idx[k]] == v
            return J[w].sum(0)

        def post(acq):
            p = mass(acq); return float(p[1] / p.sum())

        def outcomes(acq, k):
            base = mass(acq).sum()
            return [(v, float(mass(acq + ((k, v),)).sum() / base)) for v in (0, 1)]
        return post, outcomes
    model = ast.literal_eval(sc["model"]) if isinstance(sc["model"], str) else sc["model"]
    pri = np.asarray(model["prior"], dtype=float); items = {it["label"]: np.asarray(it["lik"], dtype=float) for it in model["items"]}

    def pv(acq):
        p = pri.copy()
        for k, v in acq:
            p = p * items[k][:, v]
        return p / p.sum()

    def post(acq):
        return float(pv(acq)[1])

    def outcomes(acq, k):
        p = pv(acq) @ items[k]
        return [(v, float(p[v])) for v in range(items[k].shape[1])]
    return post, outcomes


def exact_values(sc, cost_scale=1.0, g=None, l=None):
    """(q {label: value of acquiring it first, then optimal}, value of stopping now, DP value) under the reference model"""
    g = float(sc["g"]) if g is None else g; l = float(sc["l"]) if l is None else l
    cost = {c["label"]: cost_scale * float(c["cost"]) for c in sc["checks"]}
    post, outcomes = true_model(sc)
    vstop = lambda b: max((1 - b) * g - b * l, 0.0)

    @lru_cache(maxsize=None)
    def V(acq):
        best = vstop(post(acq)); done = {k for k, _ in acq}
        for k in cost:
            if k in done:
                continue
            val = -cost[k] + sum(p * V(tuple(sorted(acq + ((k, v),)))) for v, p in outcomes(acq, k) if p > 1e-12)
            best = max(best, val)
        return best
    q = {}
    for k in cost:
        q[k] = -cost[k] + sum(p * V(((k, v),)) for v, p in outcomes((), k) if p > 1e-12)
    vs = vstop(post(()))
    return q, vs, max(vs, max(q.values()) if q else vs)


def dp_regret(choice, q, vs, dpv, l):
    return (dpv - (vs if choice == "S" else q[choice])) / l


def rescale(sc, cost_scale=1.0, g=None, l=None):
    """copy of the scenario with rescaled costs / payoffs as seen by the policy (text is not used by DIVA)"""
    s = dict(sc)
    s["checks"] = [dict(c, cost=cost_scale * float(c["cost"])) for c in sc["checks"]]
    if "costs" in sc:
        s["costs"] = {k: cost_scale * float(v) for k, v in sc["costs"].items()}
    if g is not None:
        s["g"] = g
    if l is not None:
        s["l"] = l
    return s
