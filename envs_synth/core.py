"""Exact decision-theoretic engine for the synthetic information environments.

A scenario model (the `model` field of every row) is fully discrete:
  prior      : list[K]        prior over joint hidden states s = 0..K-1
  utility    : list[A][K]     U(a, s) for every action a and hidden state s
  items      : list of {label, lik: list[K][O], cost, outcomes: list[O] (names)}
               lik[s][o] = P(item outcome o | state s); items are conditionally independent given s.
  budget     : max number of acquisitions (None = unlimited, i.e. number of items)
  observed   : list of [item_index, outcome] already revealed before the episode starts (binary track only)

Everything here is exact enumeration (no sampling, no LLM). Beliefs are kept as unnormalised joint
weights w(s) = prior(s) * prod_j lik_j[s][o_j]; then P(o | history) * E[U(a) | history, o] = U @ (w * lik[:, o]) / sum(w),
which gives exact VOI without dividing by small probabilities.

Definitions (b = current belief, VOI matches the project definition):
  EU(a|b)      = sum_s b(s) U(a,s);  V0(b) = max_a EU(a|b)
  VOI_i(b)     = sum_o P(o|b) max_a EU(a|b,o) - V0(b)            (myopic, one-step, cost-free)
  net_i(b)     = VOI_i(b) - cost_i
  EIG_i(b)     = H(S|b) - sum_o P(o|b) H(S|b,o)                   (bits, information about the hidden state)
  EIGdec_i(b)  = H(a*(S)|b) - sum_o P(o|b) H(a*(S)|b,o)           (bits, information about the full-information
                                                                     optimal action a*(s) = argmax_a U(a,s); Entropy-Search-style)
  DP: V*(h) = max( max_a EU(a|b_h),  max_{i unobserved, budget left} [-cost_i + sum_o P(o|b_h) V*(h + (i,o))] )
"""
import math
import numpy as np

TOL = 1e-9


class Model:
    def __init__(self, m):
        self.prior = np.asarray(m["prior"], dtype=np.float64)
        self.U = np.asarray(m["utility"], dtype=np.float64)          # A x K
        self.items = m["items"]
        self.L = [np.asarray(it["lik"], dtype=np.float64) for it in self.items]   # K x O_i
        self.cost = np.asarray([float(it["cost"]) for it in self.items])
        self.labels = [it["label"] for it in self.items]
        self.lab2i = {l: i for i, l in enumerate(self.labels)}
        self.n = len(self.items)
        self.budget = m.get("budget") or self.n
        self.observed0 = [tuple(x) for x in m.get("observed", [])]
        # full-information optimal action per state (ties -> lowest index), for EIGdec
        self.astar = np.argmax(self.U, axis=0)
        self.A = self.U.shape[0]
        self._dp = {}

    # ------------------------------------------------------------ beliefs
    def weights(self, obs):
        w = self.prior.copy()
        for i, o in obs:
            w = w * self.L[i][:, o]
        return w

    def belief(self, obs):
        w = self.weights(obs)
        return w / w.sum()

    def eu(self, obs):
        return self.U @ self.belief(obs)

    # ------------------------------------------------------------ values
    @staticmethod
    def _H(p):
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    def _Hdec(self, b):
        q = np.bincount(self.astar, weights=b, minlength=self.A)
        return self._H(q)

    def item_stats(self, obs, i):
        """(VOI, EIG, EIGdec, P(o) vector) of item i at history obs."""
        w = self.weights(obs)
        Z = w.sum()
        b = w / Z
        v0 = float((self.U @ b).max())
        Wo = w[:, None] * self.L[i]               # K x O
        po = Wo.sum(0) / Z
        val = float((self.U @ Wo).max(0).sum() / Z)
        voi = val - v0
        if abs(voi) < TOL:
            voi = 0.0
        h0, hd0 = self._H(b), self._Hdec(b)
        eig = h0
        eigd = hd0
        for o in range(Wo.shape[1]):
            if po[o] <= 0:
                continue
            bo = Wo[:, o] / Wo[:, o].sum()
            eig -= po[o] * self._H(bo)
            eigd -= po[o] * self._Hdec(bo)
        eig = 0.0 if abs(eig) < TOL else eig
        eigd = 0.0 if abs(eigd) < TOL else eigd
        return max(voi, 0.0), max(eig, 0.0), max(eigd, 0.0), po

    def node_table(self, obs):
        """All per-item quantities at a history + DP values. Returns dict."""
        obs = tuple(sorted(obs))
        done = {i for i, _ in obs}
        b = self.belief(obs)
        eu = self.U @ b
        out = {"belief": b, "eu": eu, "best_action": int(np.argmax(eu)), "v0": float(eu.max()),
               "voi": {}, "eig": {}, "eig_dec": {}, "net": {}, "q": {}, "p_out": {}}
        nacq = len(obs) - len(self.observed0)
        for i in range(self.n):
            if i in done:
                continue
            voi, eig, eigd, po = self.item_stats(obs, i)
            out["voi"][i], out["eig"][i], out["eig_dec"][i] = voi, eig, eigd
            out["net"][i] = voi - self.cost[i]
            out["p_out"][i] = po
            if nacq < self.budget:
                out["q"][i] = self.q_value(obs, i)
        out["v_star"] = self.value(obs)
        return out

    # ------------------------------------------------------------ DP
    def value(self, obs):
        key = tuple(sorted(obs))
        if key in self._dp:
            return self._dp[key]
        w = self.weights(key)
        Z = w.sum()
        best = float((self.U @ w).max() / Z)
        nacq = len(key) - len(self.observed0)
        if nacq < self.budget:
            done = {i for i, _ in key}
            for i in range(self.n):
                if i in done:
                    continue
                q = self._q(key, i, w, Z)
                if q > best + TOL:
                    best = q
        self._dp[key] = best
        return best

    def _q(self, key, i, w, Z):
        Wo = w[:, None] * self.L[i]
        po = Wo.sum(0) / Z
        q = -self.cost[i]
        for o in range(Wo.shape[1]):
            if po[o] <= 0:
                continue
            q += po[o] * self.value(key + ((i, o),))
        return float(q)

    def q_value(self, obs, i):
        key = tuple(sorted(obs))
        w = self.weights(key)
        return self._q(key, i, w, w.sum())

    def dp_first(self, obs, tol=1e-7):
        """Optimal move at history obs: ('S', action) or ('A', item_index); also ties."""
        key = tuple(sorted(obs))
        v = self.value(key)
        eu = self.eu(key)
        stop = float(eu.max())
        nacq = len(key) - len(self.observed0)
        qs = {}
        if nacq < self.budget:
            done = {i for i, _ in key}
            qs = {i: self.q_value(key, i) for i in range(self.n) if i not in done}
        best_items = [i for i, q in qs.items() if q >= v - tol]
        if stop >= v - tol:            # stopping is (co-)optimal -> prefer stop (cheaper, no information needed)
            return {"move": "S", "action": int(np.argmax(eu)), "v": v, "ties": best_items, "q": qs, "q_stop": stop}
        return {"move": "A", "item": max(best_items, key=lambda i: qs[i]), "v": v, "ties": best_items, "q": qs,
                "q_stop": stop}


def ent_bits(p):
    p = np.asarray(p, float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def sample_episode(model: Model, rng):
    """Sample a true state from the belief after the pre-observed items, and an outcome for every item."""
    b = model.belief(model.observed0)
    s = int(rng.choice(len(b), p=b))
    outs = [int(rng.choice(model.L[i].shape[1], p=model.L[i][s] / model.L[i][s].sum())) for i in range(model.n)]
    return {"state": s, "outcomes": outs}


def summarize_root(model: Model, tol=1e-7):
    """Exact summary at the starting history (used for the `exact` field)."""
    obs = model.observed0
    t = model.node_table(obs)
    first = model.dp_first(obs)
    labels = model.labels
    voi = {labels[i]: t["voi"][i] for i in t["voi"]}
    eig = {labels[i]: t["eig"][i] for i in t["eig"]}
    eigd = {labels[i]: t["eig_dec"][i] for i in t["eig_dec"]}
    net = {labels[i]: t["net"][i] for i in t["net"]}
    maxnet = max(net.values()) if net else 0.0
    if maxnet > tol:
        gold = [l for l, v in net.items() if v >= maxnet - tol]
        gold_choice = max(gold, key=lambda l: (net[l], -labels.index(l)))
    else:
        gold, gold_choice = ["S"], "S"
    return {
        "posterior": t["belief"].tolist(),
        "eu": t["eu"].tolist(),
        "best_action": t["best_action"],
        "v0": t["v0"],
        "voi": voi, "eig": eig, "eig_dec": eigd, "net": net,
        "q": {labels[i]: v for i, v in t["q"].items()},
        "gold_choice": gold_choice,             # myopic: argmax_i VOI_i - c_i if > 0 else "S"
        "gold_ties": gold,
        "dp_value": t["v_star"],
        "dp_first_action": "S" if first["move"] == "S" else labels[first["item"]],
        "dp_first_ties": ["S"] if first["move"] == "S" else [labels[i] for i in first["ties"]],
        "myopic_value": max(t["v0"], t["v0"] + maxnet),
        "value_of_stopping": t["v0"],
    }
