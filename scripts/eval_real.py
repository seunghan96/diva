"""FIO-VOI scores for binned real-data scenarios from run_factored_real.py outputs + metrics vs empirical gold.
usage: eval_real.py ENV FR_OUT"""
import sys, os, json, math, collections
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from evalx import read, score, summarize, choose_max, spearman, clip


def odds(p):
    p = clip(p); return p / (1 - p)


def H(b):
    b = min(max(b, 1e-12), 1 - 1e-12); return -(b * math.log2(b) + (1 - b) * math.log2(1 - b))


def first(x):
    if isinstance(x, list):
        v = [y for y in x if y is not None]
        return v[0] if v else None
    return x


def agg(x, mode="greedy"):
    if not isinstance(x, list):
        return x
    v = [y for y in x if y is not None]
    if not v:
        return None
    if mode == "greedy":
        return v[0]
    lo = [math.log(odds(y)) for y in v]
    m = sum(lo) / len(lo)
    return 1 / (1 + math.exp(-m))


def stated_base_rate(sc):
    """Base rate as STATED in the text (never the reference posterior stored in sc['prior'])."""
    import re
    t = (sc.get("text_parts") or {}).get("base_rate", "") or sc.get("text", "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", t)
    return float(m.group(1)) / 100 if m else 0.5


def fio_real(sc, fr, mode="greedy", prior_mode="elicit", lam=1.0):
    g, l = float(sc["g"]), float(sc["l"])
    fb = 0
    def gv(x):
        nonlocal fb
        v = agg(x, mode)
        if v is None:
            fb += 1; return None
        return clip(v)
    pri = gv(fr["prior"])
    if prior_mode == "stated" or pri is None:
        pri = clip(stated_base_rate(sc))
    o = odds(pri)
    for x in fr.get("obs", []):
        c = gv(x)
        if c is not None:
            c = clip(pri + lam * (c - pri))
            o *= odds(c) / odds(pri)
    b0 = o / (1 + o)
    def vstar(b):
        return max((1 - b) * g - b * l, 0.0)
    voi, eig, par = {}, {}, {}
    for c in sc["checks"]:
        raw = fr["chk"].get(c["label"], [])
        if raw and isinstance(raw[0], list) and raw[0] and raw[0][0] == "RATIO":
            rho = float(raw[0][1]) ** lam
            lo = min(0.99, max(0.01, pri * 2 / (1 + rho))); hi = min(0.99, max(0.01, rho * lo))
            raw = [[lo]] + [[pri]] * (len(raw) - 2) + [[hi]]
            cs = [x[0] for x in raw]
        else:
            cs = [gv(x) for x in raw]
        nb = len(c.get("bins", [])) or len(cs)
        if not (raw and isinstance(fr["chk"].get(c["label"], [[None]])[0], list) and fr["chk"][c["label"]][0] and fr["chk"][c["label"]][0][0] == "RATIO"):
            cs = [pri if v is None else clip(pri + lam * (v - pri)) for v in cs]
        if not cs:
            voi[c["label"]] = 0.0; eig[c["label"]] = 0.0; continue
        l1 = np.array([v / nb for v in cs]); l0 = np.array([(1 - v) / nb for v in cs])
        l1 /= l1.sum(); l0 /= l0.sum()
        par[c["label"]] = (l1.tolist(), l0.tolist(), float(c["cost"]))
        pb = b0 * l1 + (1 - b0) * l0
        post = b0 * l1 / pb
        voi[c["label"]] = float(sum(pb[i] * vstar(post[i]) for i in range(len(pb))) - vstar(b0))
        eig[c["label"]] = float(H(b0) - sum(pb[i] * H(post[i]) for i in range(len(pb))))
    net = {k: voi[k] - float(c["cost"]) for k, c in zip(voi, sc["checks"])}
    return dict(b0=b0, voi=voi, eig=eig, net=net, fallback=fb, par=par)


def run(env_p, fr_p, mode="greedy", eig_tau=0.05, dir_p=None):
    env = {s["id"]: s for s in read(env_p)}
    fr = {r["id"]: r for r in read(fr_p)}
    dr = {r["id"]: r for r in read(dir_p)} if dir_p else {}
    res = collections.defaultdict(list); rho = collections.defaultdict(list)
    for i, sc in env.items():
        if i not in fr:
            continue
        f = fio_real(sc, fr[i], mode)
        ch = {"fiovoi": choose_max(f["net"]), "fac_eig": choose_max(f["eig"], eig_tau), "always_stop": "S",
              "oracle": sc["exact"]["gold_choice"]}
        for m, c in ch.items():
            res[m].append(score(sc, c))
        if i in dr and dr[i].get("dist"):
            res["direct"].append(score(sc, max(dr[i]["dist"], key=dr[i]["dist"].get)))
        labs = [c["label"] for c in sc["checks"]]
        for m, s in (("fiovoi", f["net"]), ("fac_eig", f["eig"])):
            rho[m].append(spearman([s[k] for k in labs], [sc["exact"]["net"][k] for k in labs]))
    out = {}
    for m, rows in res.items():
        s = summarize(rows); s["rho"] = float(np.nanmean(rho[m])) if rho[m] else float("nan"); out[m] = s
    return out


if __name__ == "__main__":
    o = run(sys.argv[1], sys.argv[2], "greedy", 0.05, sys.argv[3] if len(sys.argv) > 3 else None)
    for m, s in o.items():
        print(f"{m:12s} n={s['n']} regret={s['norm_regret']:.4f} dpreg={s.get('dp_regret', float('nan')):.4f} top1={s['top1']:.2f} over={s['over_acq']:.2f} under={s['under_acq']:.2f} plac={s['placebo']:.3f} rho={s['rho']:.2f}")


def _pick(v, j):
    if isinstance(v, list) and v and isinstance(v[0], list):
        return [_pick(x, j) for x in v]
    if isinstance(v, list):
        return [v[j]] if j < len(v) and v[j] is not None else v[:1]
    return v


def spec_j(fr, j):
    return {"prior": _pick(fr["prior"], j), "obs": [_pick(o, j) for o in fr.get("obs", [])],
            "chk": {k: [_pick(b, j) for b in v] for k, v in fr["chk"].items()}}


def fio_real_robust(sc, fr, q=0.25, nspec=5, lam=1.0):
    nets = []
    for j in range(nspec):
        f = fio_real(sc, spec_j(fr, j), lam=lam)
        nets.append(f["net"])
    labs = list(nets[0])
    mean = {k: float(np.mean([n[k] for n in nets])) for k in labs}
    lo = {k: float(np.quantile([n[k] for n in nets], q)) for k in labs}
    k = max(mean, key=mean.get)
    return (k if lo[k] > 1e-9 else "S"), mean


def fio_real_u(sc, fr, sigma=1.0, lam=0.75):
    """Uncertainty-aware DIVA for binned checks (PREREG_KNOW3)."""
    import math, collections
    GH = [(-math.sqrt(3), 1 / 6), (0.0, 2 / 3), (math.sqrt(3), 1 / 6)]

    def shifted(delta):
        out = {"prior": fr["prior"], "obs": fr["obs"], "chk": {}}
        for k, v in fr["chk"].items():
            vv = [x[0] if isinstance(x, list) else x for x in v]
            if any((x is None or isinstance(x, str)) for x in vv) or len(vv) < 2:
                out["chk"][k] = v; continue
            n = len(vv); new = []
            for j, x in enumerate(vv):
                t = (j - (n - 1) / 2) / ((n - 1) / 2)
                x = clip(x); lo = math.log(x / (1 - x)) + delta * t
                new.append([1 / (1 + math.exp(-lo))])
            out["chk"][k] = new
        return out
    nets = collections.defaultdict(float); base = None
    for z, w in GH:
        fz = fio_real(sc, shifted(sigma * z), lam=lam)
        if z == 0.0:
            base = fz
        for k2, v in fz["net"].items():
            nets[k2] += w * v
    return dict(base, net=dict(nets))
