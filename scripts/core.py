"""Exact decision-theoretic quantities for binary-outcome, binary-check scenarios (used by the method and the metrics)."""
import math, itertools
from functools import lru_cache


def eu(b, g, l):
    """Expected utilities (act, pass) at belief b = P(Y=1)."""
    return ((1 - b) * g - b * l, 0.0)


def vstar(b, g, l):
    a, p = eu(b, g, l)
    return max(a, p)


def best_action(b, g, l):
    a, p = eu(b, g, l)
    return "act" if a > p else "pass"


def H(b):
    b = min(max(b, 1e-12), 1 - 1e-12)
    return -(b * math.log2(b) + (1 - b) * math.log2(1 - b))


def bayes(b, s1, s0, v):
    """Posterior P(Y=1) after observing check value v (1 = finding) with P(v=1|Y=1)=s1, P(v=1|Y=0)=s0."""
    l1 = s1 if v else 1 - s1
    l0 = s0 if v else 1 - s0
    z = b * l1 + (1 - b) * l0
    return b * l1 / z if z > 0 else b


def pred(b, s1, s0):
    """P(v=1 | belief b)."""
    return b * s1 + (1 - b) * s0


def voi_exact(b, s1, s0, g, l):
    p1 = pred(b, s1, s0)
    return p1 * vstar(bayes(b, s1, s0, 1), g, l) + (1 - p1) * vstar(bayes(b, s1, s0, 0), g, l) - vstar(b, g, l)


def eig_exact(b, s1, s0):
    p1 = pred(b, s1, s0)
    return H(b) - p1 * H(bayes(b, s1, s0, 1)) - (1 - p1) * H(bayes(b, s1, s0, 0))


def posterior(prior, observed):
    b = prior
    for (s1, s0, v) in observed:
        b = bayes(b, s1, s0, v)
    return b


def dp_value(b, checks, g, l):
    """Optimal expected (utility - costs) from belief b with remaining checks [(s1,s0,cost)], exact recursion.
    Returns (value, first_action) where first_action is index into checks or 'S'."""
    checks = tuple(checks)

    @lru_cache(maxsize=None)
    def V(bq, remaining):
        b_ = bq / 1e9
        vals = {"S": vstar(b_, g, l)}
        for i in remaining:
            s1, s0, c = checks[i]
            p1 = pred(b_, s1, s0)
            rest = tuple(j for j in remaining if j != i)
            vals[i] = -c + p1 * V(round(bayes(b_, s1, s0, 1) * 1e9), rest)[0] + (1 - p1) * V(round(bayes(b_, s1, s0, 0) * 1e9), rest)[0]
        best = max(vals.values())
        cands = [k for k, v in vals.items() if v >= best - 1e-6]
        if "S" in cands:
            return vals["S"], "S"
        # tie-break among (near-)equal continuations by myopic net value
        arg = max(cands, key=lambda i: voi_exact(b_, checks[i][0], checks[i][1], g, l) - checks[i][2])
        return best, arg

    return V(round(b * 1e9), tuple(range(len(checks))))


# ------------------------------------------------------------ estimated VOI from (possibly incoherent) LLM beliefs
def voi_from_beliefs(b0, bv1, bv0, g, l, p1=None, mode="martingale"):
    """Imagined-observation VOI. b0: current belief; bv1/bv0: beliefs if the check shows / does not show its finding.
    p1: predictive P(finding). mode='martingale' derives p1 from Bayes-consistency b0 = p1*bv1 + (1-p1)*bv0
    (clipped to [0,1]); 'given' uses p1 as provided; 'uniform' uses 0.5."""
    if mode == "martingale":
        d = bv1 - bv0
        if abs(d) < 1e-6:
            p1 = 0.5
        else:
            p1 = min(1.0, max(0.0, (b0 - bv0) / d))
    elif mode == "uniform":
        p1 = 0.5
    val = p1 * vstar(bv1, g, l) + (1 - p1) * vstar(bv0, g, l) - vstar(b0, g, l)
    return val, p1


def eig_from_beliefs(b0, bv1, bv0, p1=None, mode="martingale"):
    if mode == "martingale":
        d = bv1 - bv0
        p1 = 0.5 if abs(d) < 1e-6 else min(1.0, max(0.0, (b0 - bv0) / d))
    elif mode == "uniform":
        p1 = 0.5
    return H(b0) - p1 * H(bv1) - (1 - p1) * H(bv0), p1


def dp_factory(params, g, l):
    """params: {label: (s1, s0, cost)}. Returns V(b, remaining_labels_tuple) -> (value, first_action_label_or_'S'),
    memoized across calls (one factory per scenario/model)."""
    @lru_cache(maxsize=None)
    def V(bq, remaining):
        b_ = bq / 1e9
        vals = {"S": vstar(b_, g, l)}
        for k in remaining:
            s1, s0, c = params[k]
            p1 = pred(b_, s1, s0)
            rest = tuple(j for j in remaining if j != k)
            vals[k] = -c + p1 * V(round(bayes(b_, s1, s0, 1) * 1e9), rest)[0] + (1 - p1) * V(round(bayes(b_, s1, s0, 0) * 1e9), rest)[0]
        best = max(vals.values())
        cands = [k for k, v in vals.items() if v >= best - 1e-6]
        if "S" in cands:
            return vals["S"], "S"
        arg = max(cands, key=lambda k: voi_exact(b_, params[k][0], params[k][1], g, l) - params[k][2])
        return best, arg
    return lambda b, remaining: V(round(b * 1e9), tuple(sorted(remaining)))
