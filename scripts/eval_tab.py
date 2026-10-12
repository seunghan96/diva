"""Evaluate DIVA (knowledge-required, binary checks) from run_factored_k.py outputs, plus direct (run_direct.py) and
references. Variants: elicitation form (lik / cond / both) x aggregation (greedy / mean of K samples incl. greedy).
usage: eval_tab.py ENV FK_OUT [DIRECT_OUT]"""
import sys, os, json, math, collections
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from evalx import read, score, summarize, choose_max, spearman, clip
from dcore import posterior, voi_exact, eig_exact


def logit(p):
    p = clip(p); return math.log(p / (1 - p))


def sig(x):
    return 1 / (1 + math.exp(-x))


def agg(v, mode):
    if v is None:
        return None
    v = [x for x in v if x is not None]
    if not v:
        return None
    if mode == "greedy":
        return v[0]
    return sig(sum(logit(x) for x in v) / len(v))


def lik_pair(rec, form, mode):
    """Return (P(f|Y=1), P(f|Y=0)) from an elicitation record for one check."""
    out = []
    if form in ("lik", "both"):
        l1, l0 = agg(rec.get("l1"), mode), agg(rec.get("l0"), mode)
        if l1 is not None and l0 is not None:
            out.append((clip(l1), clip(l0)))
    if form in ("cond", "both"):
        c1, c0, pf = agg(rec.get("c1"), mode), agg(rec.get("c0"), mode), agg(rec.get("pf"), mode)
        if None not in (c1, c0, pf):
            c1, c0, pf = clip(c1), clip(c0), clip(pf)
            a1 = c1 * pf / (c1 * pf + c0 * (1 - pf))
            a0 = (1 - c1) * pf / ((1 - c1) * pf + (1 - c0) * (1 - pf))
            out.append((clip(a1), clip(a0)))
    if not out:
        return None
    if len(out) == 1:
        return out[0]
    return (sig(sum(logit(x[0]) for x in out) / len(out)), sig(sum(logit(x[1]) for x in out) / len(out)))


def temper(lp, lam):
    """Shrink an elicited likelihood pair toward its mean by factor lam (lam=1: unchanged); the marginal
    P(finding) at an even prior is preserved and the likelihood ratio moves toward 1."""
    if lam == 1.0:
        return lp
    m = (lp[0] + lp[1]) / 2
    return (clip(m + lam * (lp[0] - m)), clip(m + lam * (lp[1] - m)))


def diva_scores(sc, fr, form="lik", mode="greedy", prior="elicit", lam=1.0):
    fb = 0
    pri = agg(fr.get("prior"), mode)
    if prior == "stated" or pri is None:
        pri = sc["prior"]
    obs = []
    for rec, ob in zip(fr.get("obs", []), sc.get("observed", [])):
        lp = lik_pair(rec, form, mode)
        if lp is None:
            fb += 1; continue
        lp = temper(lp, lam)
        obs.append((lp[0], lp[1], int(ob["value"])))
    b = posterior(clip(pri), obs)
    g, l = float(sc["g"]), float(sc["l"])
    net, eig, par = {}, {}, {}
    for c in sc["checks"]:
        lp = lik_pair(fr["chk"].get(c["label"], {}), form, mode)
        if lp is None:
            fb += 1; lp = (0.5, 0.5)
        lp = temper(lp, lam)
        par[c["label"]] = (lp[0], lp[1], float(c["cost"]))
        net[c["label"]] = voi_exact(b, lp[0], lp[1], g, l) - float(c["cost"])
        eig[c["label"]] = eig_exact(b, lp[0], lp[1])
    return dict(b=b, net=net, eig=eig, par=par, fallback=fb)


def run(env_p, fk_p, dir_p=None, forms=("lik", "cond", "both"), modes=("greedy", "mean"), eig_tau=0.05):
    env = {s["id"]: s for s in read(env_p)}
    fk = {r["id"]: r for r in read(fk_p)}
    dr = {r["id"]: r for r in read(dir_p)} if dir_p else {}
    res = collections.defaultdict(list); rho = collections.defaultdict(list); bri = collections.defaultdict(list)
    for i, sc in env.items():
        if i not in fk:
            continue
        labs = [c["label"] for c in sc["checks"]]
        for form in forms:
            for mode in modes:
                f = diva_scores(sc, fk[i], form, mode)
                nm = f"diva_{form}_{mode}"
                res[nm].append(score(sc, choose_max(f["net"])))
                rho[nm].append(spearman([f["net"][k] for k in labs], [sc["exact"]["net"][k] for k in labs]))
                if form == "lik" and mode == "greedy":
                    res["fac_eig"].append(score(sc, choose_max(f["eig"], eig_tau)))
                    rho["fac_eig"].append(spearman([f["eig"][k] for k in labs], [sc["exact"]["net"][k] for k in labs]))
                bri[nm].append((f["b"] - sc["exact"]["posterior"]) ** 2)
        if i in dr and dr[i].get("dist"):
            d = dr[i]["dist"]
            res["direct"].append(score(sc, max(d, key=d.get)))
            rho["direct"].append(spearman([d.get(k, 0) for k in labs], [sc["exact"]["net"][k] for k in labs]))
        res["always_stop"].append(score(sc, "S"))
        res["oracle"].append(score(sc, sc["exact"]["gold_choice"]))
    out = {}
    for m, rows in res.items():
        s = summarize(rows); s["rho"] = float(np.nanmean(rho[m])) if rho[m] else float("nan")
        if bri[m]:
            s["belief_mse"] = float(np.mean(bri[m]))
        out[m] = s
    return out


if __name__ == "__main__":
    o = run(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    for m, s in sorted(o.items()):
        print(f"{m:18s} n={s['n']} regret={s['norm_regret']:.4f} dpreg={s.get('dp_regret', float('nan')):.4f} capt={s.get('captured', float('nan')):.2f} top1={s['top1']:.2f} over={s['over_acq']:.2f} under={s['under_acq']:.2f} plac={s['placebo']:.3f} rho={s['rho']:.2f} bmse={s.get('belief_mse', float('nan')):.3f}")


def pick(v, j):
    if not isinstance(v, list):
        return v
    if j < len(v) and v[j] is not None:
        return [v[j]]
    return v[:1]


def spec_j(fr, j):
    """j-th sampled specification (j=0 greedy) from a run_factored_k record."""
    g = {"prior": pick(fr.get("prior"), j), "obs": [], "chk": {}}
    for rec in fr.get("obs", []):
        g["obs"].append({k: pick(v, j) for k, v in rec.items()})
    for lab, rec in fr.get("chk", {}).items():
        g["chk"][lab] = {k: pick(v, j) for k, v in rec.items()}
    return g


def diva_robust(sc, fr, form="cond", q=0.25, lam=1.0, nspec=5):
    """Robust DIVA: compute net values under each sampled specification; rank checks by the mean net value and
    acquire only if the q-quantile of the chosen check's net value across specifications is > 0."""
    nets = []
    for j in range(nspec):
        f = diva_scores(sc, spec_j(fr, j), form, "greedy", "elicit", lam)
        nets.append(f["net"])
    labs = list(nets[0])
    mean = {k: float(np.mean([n[k] for n in nets])) for k in labs}
    lo = {k: float(np.quantile([n[k] for n in nets], q)) for k in labs}
    k = max(mean, key=mean.get)
    return (k if lo[k] > 1e-9 else "S"), mean


def diva_u(sc, fr, sigma=1.0, form="cond", lam=0.75):
    """Uncertainty-aware DIVA (PREREG_KNOW3): expected VOI over a Gaussian (scale sigma) on each check's log-odds spread."""
    import math
    from dcore import voi_exact
    f = diva_scores(sc, fr, form, "greedy", "elicit", lam)
    g, l = float(sc["g"]), float(sc["l"])
    GH = [(-math.sqrt(3), 1 / 6), (0.0, 2 / 3), (math.sqrt(3), 1 / 6)]
    lg = lambda x: math.log(clip(x) / (1 - clip(x)))
    sg = lambda x: 1 / (1 + math.exp(-x))
    net = {}
    for c in sc["checks"]:
        s1, s0, cost = f["par"][c["label"]]
        v = sum(w * voi_exact(f["b"], clip(sg(lg(s1) + sigma * z / 2)), clip(sg(lg(s0) - sigma * z / 2)), g, l) for z, w in GH)
        net[c["label"]] = v - cost
    return dict(f, net=net)
