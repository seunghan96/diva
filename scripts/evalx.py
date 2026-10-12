"""Unified first-step evaluation for envs_synth / envs_tab / envs_real style scenarios.
Gold: myopic (net, gold_ties) AND sequential (DP Q-values: exact.q[label], exact.value_of_stopping, exact.dp_value).
Tolerance 1e-9 everywhere; tied-optimal choices count as correct; no silent drops (failed elicitations fall back to
0.5 and are counted in 'fallback')."""
import sys, os, json, gzip, math, collections
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from dcore import posterior, voi_exact, eig_exact, dp_value

TOL = 1e-9


def clip(x):
    return min(max(float(x), 0.01), 0.99)


def fac_scores(sc, fr):
    """Scores from a factored elicitation record. Returns dict with voi/eig/net and chosen labels for variants."""
    fb = 0
    def val(x):
        nonlocal fb
        if x is None:
            fb += 1; return 0.5
        return clip(x)
    prior = val(fr.get("prior"))
    obs = []
    for o, ob in zip(fr.get("obs", []), sc.get("observed", [])):
        obs.append((val(o[0]), val(o[1]), int(ob["value"])))
    b = posterior(prior, obs)
    g, l = float(sc["g"]), float(sc["l"])
    voi, eig, net, par = {}, {}, {}, {}
    for c in sc["checks"]:
        s1, s0 = (fr.get("chk", {}).get(c["label"]) or [None, None])
        s1, s0 = val(s1), val(s0)
        par[c["label"]] = (s1, s0, float(c["cost"]))
        voi[c["label"]] = voi_exact(b, s1, s0, g, l)
        eig[c["label"]] = eig_exact(b, s1, s0)
        net[c["label"]] = voi[c["label"]] - float(c["cost"])
    return dict(b=b, voi=voi, eig=eig, net=net, par=par, fallback=fb)


def choose_max(d, tau=0.0):
    k = max(d, key=d.get)
    return k if d[k] > tau + TOL else "S"


def scale(sc):
    if sc.get("l") is not None:
        return float(sc["l"])
    U = np.asarray(sc["model"]["utility"], dtype=float)  # multi-state track: payoff range across actions/states
    return float(U.max() - U.min()) / 2 or 1.0


def score(sc, choice):
    ex = sc["exact"]
    L = scale(sc)
    net = ex["net"]
    best = max(0.0, max(net.values()))
    v = 0.0 if choice == "S" else net[choice]
    ties = ex.get("gold_ties") or [ex.get("gold_choice")]
    gold_stop = (ex.get("gold_choice") == "S")
    r = dict(regret=best - v, norm_regret=(best - v) / L, top1=float(choice in ties),
             stop_gold=float(gold_stop), stopped=float(choice == "S"),
             placebo=float(choice != "S" and any(c["label"] == choice and c.get("is_placebo") for c in sc["checks"])))
    if "dp_first_action" in ex:
        r["dp_stop"] = float("S" in (ex.get("dp_first_ties") or [ex.get("dp_first_action")]) or ex.get("dp_first_action") in (None, "S"))
    st = sc.get("stratum") or ex.get("stratum")
    r["stratum"] = st
    if st == "S1":
        eig = ex["eig"]; ae = max(eig, key=eig.get)
        r["eig_trap"] = float(choice == ae)
    # sequential (DP) gold
    q = ex.get("q")
    if q is not None and ex.get("dp_value") is not None:
        vs = float(ex.get("value_of_stopping", 0.0))
        qc = vs if choice == "S" else float(q[choice])
        vdp = float(ex["dp_value"])
        r["dp_regret"] = (vdp - qc) / L
        dties = ex.get("dp_first_ties") or [ex.get("dp_first_action")]
        dties = ["S" if (t in (None, "S", "STOP", "stop")) else t for t in dties]
        r["dp_top1"] = float(choice in dties)
        r["_qc"], r["_vs"], r["_vdp"] = qc / L, vs / L, vdp / L
    return r


def summarize(rows):
    out = {}
    keys = set(k for r in rows for k, v in r.items() if isinstance(v, (int, float)))
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    stp = [r for r in rows if r["stop_gold"] == 1]
    acq = [r for r in rows if r["stop_gold"] == 0]
    out["over_acq"] = float(np.mean([1 - r["stopped"] for r in stp])) if stp else float("nan")
    out["under_acq"] = float(np.mean([r["stopped"] for r in acq])) if acq else float("nan")
    q = [r for r in rows if "_qc" in r]
    if q:
        num = np.mean([r["_qc"] - r["_vs"] for r in q]); den = np.mean([r["_vdp"] - r["_vs"] for r in q])
        out["captured"] = float(num / den) if den > 1e-12 else float("nan")
    out["n"] = len(rows)
    return out


def spearman(a, b):
    from scipy.stats import spearmanr
    if len(set(a)) < 2 or len(set(b)) < 2:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def read(p):
    if not os.path.exists(p):
        return []
    with gzip.open(p, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]
