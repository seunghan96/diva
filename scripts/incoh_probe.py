"""Incoherence probe on the controlled binary scenarios (data for Figure fig_incoh, see figs_all.fig_incoh).

For every LLM of complete.PROBE_LLMS with outputs/probe/probe_{tag}.jsonl.gz:
  swings: for each (scenario, check) with both imagined beliefs,
          X = exact Bayesian swing  bayes(b,s1,s0,1) - bayes(b,s1,s0,0)      (b = exact posterior of the scenario)
          Y = imagined swing        bv[label]['1'].b - bv[label]['0'].b       (LLM's elicited beliefs; same pairing as
                                                                               analysis.incoherence)
  top1:   on scenarios where acquiring is optimal (exact gold choice != Stop), the check with the highest decision value
          computed from the imagined beliefs, VOI(b0, b|1, b|0; exact P(finding)) - cost (dcore.voi_from_beliefs,
          mode='given'), equals an exact best check (exact['gold_ties']); ties in the argmax get fractional credit.
          chance = uniform pick among the scenario's checks.
Cache: results/incoh_probe_cache.pkl, recomputed whenever a probe file, a scenario file or this module is newer than it.
"""
import os, sys, glob, gzip, json, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from dcore import bayes, voi_from_beliefs, pred  # noqa: E402
from complete import PROBE_LLMS, lsort  # noqa: E402

DOMS = ["finance", "credit", "clinical", "legal"]
CACHE = f"{IS}/results/incoh_probe_cache.pkl"


def _read(p):
    return [json.loads(l) for l in gzip.open(p, "rt")] if os.path.exists(p) else []


def _inputs():
    return ([f"{IS}/envs_synth/data/binary/{d}_test.jsonl.gz" for d in DOMS] + glob.glob(f"{IS}/outputs/probe/probe_*.jsonl.gz")
            + [os.path.abspath(__file__)])


def _fresh():
    if not os.path.exists(CACHE):
        return False
    t = os.path.getmtime(CACHE)
    return all(os.path.getmtime(p) <= t for p in _inputs() if os.path.exists(p))


def _compute():
    env = {}
    for d in DOMS:
        for s in _read(f"{IS}/envs_synth/data/binary/{d}_test.jsonl.gz"):
            env[s["id"]] = s
    out = {}
    for t in lsort(PROBE_LLMS):
        X, Y, hit, ch = [], [], [], []
        for pr in _read(f"{IS}/outputs/probe/probe_{t}.jsonl.gz"):
            sc = env.get(pr["id"])
            if sc is None or not pr.get("b0"):
                continue
            ex = sc["exact"]; b = ex["posterior"]
            bvs = [pr["bv"].get(c["label"]) for c in sc["checks"]]
            ok = [bool(bv and bv["1"] and bv["0"]) for bv in bvs]
            for c, bv, o in zip(sc["checks"], bvs, ok):
                if o:
                    X.append(bayes(b, c["s1"], c["s0"], 1) - bayes(b, c["s1"], c["s0"], 0)); Y.append(bv["1"]["b"] - bv["0"]["b"])
            if not all(ok) or ex["gold_choice"] == "S":
                continue
            g, l = float(sc["g"]), float(sc["l"])
            net = {c["label"]: voi_from_beliefs(pr["b0"]["b"], bv["1"]["b"], bv["0"]["b"], g, l, p1=pred(b, c["s1"], c["s0"]),
                                                mode="given")[0] - float(c["cost"]) for c, bv in zip(sc["checks"], bvs)}
            ties = {x for x in (ex.get("gold_ties") or [ex["gold_choice"]]) if x != "S"}
            m = max(net.values()); A = [k for k, v in net.items() if v >= m - 1e-9]
            hit.append(len(set(A) & ties) / len(A)); ch.append(len(ties) / len(sc["checks"]))
        if X:
            out[t] = dict(X=np.array(X), Y=np.array(Y), top1=float(np.mean(hit)) if hit else None,
                          chance=float(np.mean(ch)) if ch else None, n_top1=len(hit))
    return out


def load():
    """{tag: dict(X, Y, top1, chance, n_top1)} for the LLMs with probe outputs (missing LLMs are absent)."""
    if _fresh():
        raw = pickle.load(open(CACHE, "rb"))
    else:
        raw = {t: dict(v, X=[float(x) for x in v["X"]], Y=[float(y) for y in v["Y"]]) for t, v in _compute().items()}
        pickle.dump(raw, open(CACHE, "wb"))  # plain lists: readable from both venvs (numpy-version independent)
    return {t: dict(v, X=np.array(v["X"]), Y=np.array(v["Y"])) for t, v in raw.items()}


def summary(P=None):
    P = P if P is not None else load()
    XA = np.concatenate([v["X"] for v in P.values()]); YA = np.concatenate([v["Y"] for v in P.values()])
    t1 = [v["top1"] for v in P.values() if v["top1"] is not None]
    return dict(n_llm=len(P), r=float(np.corrcoef(XA, YA)[0, 1]), slope=float(np.polyfit(XA, YA, 1)[0]),
                shrink=float(np.mean(np.abs(XA)) / np.mean(np.abs(YA))), top1_mean=float(np.mean(t1)),
                top1_min=float(min(t1)), top1_max=float(max(t1)),
                chance=float(np.mean([v["chance"] for v in P.values() if v["chance"] is not None])))


if __name__ == "__main__":
    print(summary())
