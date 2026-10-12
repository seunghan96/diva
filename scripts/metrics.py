"""Single-step acquisition metrics and decision-rule derivations from probe outputs."""
import sys, os, json, gzip, math
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from dcore import voi_from_beliefs, eig_from_beliefs, vstar, H


def spearman(a, b):
    from scipy.stats import spearmanr
    if len(set(a)) < 2 or len(set(b)) < 2:
        return float("nan")
    return spearmanr(a, b).correlation


def derive(sc, pr, tau=0.0, pmode="martingale", eig_tau=0.05, belief_key="b"):
    """Return {method: (choice, scores)} for methods computed from one probe record."""
    g, l = sc["g"], sc["l"]
    out = {}
    if pr.get("direct"):
        d = pr["direct"]
        ch = max(d, key=d.get)
        out["direct"] = (ch, {k: v for k, v in d.items() if k != "S"})
    b0 = pr["b0"][belief_key] if pr.get("b0") else None
    if b0 is None or any(pr["bv"][c["label"]][x] is None for c in sc["checks"] for x in "10"):
        return out
    voi, eig = {}, {}
    for c in sc["checks"]:
        bv = pr["bv"][c["label"]]
        b1, bz = bv["1"][belief_key], bv["0"][belief_key]
        p1 = pr.get("pred", {}).get(c["label"]) if pmode == "given" else None
        if pmode == "given" and p1 is None:
            p1 = 0.5
        v, _ = voi_from_beliefs(b0, b1, bz, g, l, p1=p1, mode=pmode)
        e, _ = eig_from_beliefs(b0, b1, bz, p1=p1, mode=pmode)
        voi[c["label"]] = v - c["cost"]
        eig[c["label"]] = e
    best = max(voi, key=voi.get)
    out["iovoi"] = (best if voi[best] > tau else "S", voi)
    be = max(eig, key=eig.get)
    out["eig"] = (be if eig[be] > eig_tau else "S", eig)
    # salience-free EIG minus cost in bits is not commensurate; EIG-acquire-always variant for acquisition-only metrics
    out["eig_always"] = (be, eig)
    return out


def score_choice(sc, choice):
    ex = sc["exact"]
    net = ex["net"]
    best_val = max(0.0, max(net.values()))
    val = 0.0 if choice == "S" else net[choice]
    regret = best_val - val
    return dict(regret=regret, norm_regret=regret / sc["l"], top1=float(choice == ex["gold_choice"]),
                stop_gold=float(ex["gold_choice"] == "S"), stopped=float(choice == "S"),
                placebo=float(choice != "S" and any(c["label"] == choice and c.get("is_placebo") for c in sc["checks"])),
                eig_trap=float(sc["stratum"] == "S1" and choice == max(ex["eig"], key=ex["eig"].get)))


def summarize(rows):
    """rows: list of dicts from score_choice (+ stratum)."""
    import collections
    R = collections.defaultdict(list)
    for r in rows:
        for k, v in r.items():
            if isinstance(v, (int, float)):
                R[k].append(v)
    s = {k: float(np.mean(v)) for k, v in R.items()}
    acq = [r for r in rows if r["stop_gold"] == 0]
    stp = [r for r in rows if r["stop_gold"] == 1]
    s["over_acq"] = float(np.mean([1 - r["stopped"] for r in stp])) if stp else float("nan")
    s["under_acq"] = float(np.mean([r["stopped"] for r in acq])) if acq else float("nan")
    s1 = [r for r in rows if r.get("stratum") == "S1"]
    s["eig_trap_S1"] = float(np.mean([r["eig_trap"] for r in s1])) if s1 else float("nan")
    s["n"] = len(rows)
    return s


def rank_fidelity(sc, scores):
    labs = [c["label"] for c in sc["checks"]]
    return spearman([scores[k] for k in labs], [sc["exact"]["net"][k] for k in labs])
