"""DEV-ONLY: 'flat-specification fallback'. DIVA-U decides whether to acquire. The check is DIVA-U's own argmax unless
the elicited specification is flat (max log-odds spread over candidate checks < t), in which case the numeric answers
carry no ranking information and the LLM's direct ranking over checks is used. Grid t; criterion: mean over all dev
cells of the average of DP and myopic regret ratios to never asking. Writes results/dev_flat_switch.json."""
import sys, os, glob, json, math, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read, score, summarize, choose_max, clip
from eval_tab import diva_scores, diva_u
from eval_real import fio_real, fio_real_u, agg
T = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 99.0]
lg = lambda p: math.log(clip(p) / (1 - clip(p)))


def spread_tab(f):
    return max([abs(lg(a) - lg(b)) for a, b, _ in f["par"].values()] or [0.0])


def spread_sec(sc, fr, lam=0.75):
    pri = agg(fr["prior"], "greedy"); pri = clip(pri if pri is not None else 0.5)
    best = 0.0
    for c in sc["checks"]:
        raw = fr["chk"].get(c["label"], [])
        vals = []
        for x in raw:
            v = agg(x, "greedy") if isinstance(x, list) and (not x or not isinstance(x[0], str)) else None
            if v is not None:
                vals.append(clip(pri + lam * (v - pri)))
        if len(vals) >= 2:
            best = max(best, lg(max(vals)) - lg(min(vals)))
    return best


def cells():
    out = []
    for ds in ["taiwan", "german", "cdc", "heart"]:
        for f in glob.glob(f"{IS}/outputs/tab_dev/fk_*_{ds}.jsonl.gz"):
            tag = os.path.basename(f)[3:-9][: -(len(ds) + 1)]
            out.append(("tab", ds, f"{IS}/envs_tab/data/{ds}_dev.jsonl.gz", f, f"{IS}/outputs/tab_dev/dir_{tag}_{ds}.jsonl.gz", tag))
    for tg in ["ret_1", "ret_5", "vol_5", "vol_20"]:
        dn = "ret1" if tg == "ret_1" else tg
        for f in glob.glob(f"{IS}/outputs/real_dev/fr_*_{dn}.jsonl.gz"):
            tag = os.path.basename(f)[3:-9][: -(len(dn) + 1)]
            if tag.split("_")[-1] in ("joint", "ratio", "contrast"):
                continue
            out.append(("sec", tg, f"{IS}/envs_real/data/{tg}_dev.jsonl.gz", f, f"{IS}/outputs/real_dev/dir_{tag}_{dn}.jsonl.gz", tag))
    return out


def run(split="dev", cell_list=None, ts=T):
    res = collections.defaultdict(dict)
    for kind, e, ep, fp, dp, tag in cell_list:
        if not os.path.exists(dp):
            continue
        env = {s["id"]: s for s in read(ep)}; fr = {r["id"]: r for r in read(fp)}
        dr = {r["id"]: r["dist"] for r in read(dp) if r.get("dist")}
        ids = [i for i in env if i in fr and i in dr]
        if len(ids) < 100:
            continue
        V = collections.defaultdict(list); sp = []
        for i in ids:
            sc = env[i]
            if kind == "tab":
                f = diva_scores(sc, fr[i], "cond", "greedy", "elicit", 0.75); s_ = spread_tab(f); fu = diva_u(sc, fr[i], 1.0)
            else:
                s_ = spread_sec(sc, fr[i]); fu = fio_real_u(sc, fr[i], 1.0)
            sp.append(s_)
            labs = [c["label"] for c in sc["checks"]]
            dd = {k: v for k, v in dr[i].items() if k in labs}
            cu = choose_max(fu["net"])
            for t in ts:
                c = cu if (cu == "S" or s_ >= t or not dd) else max(dd, key=dd.get)
                V[t].append(score(sc, c))
            V["never"].append(score(sc, "S"))
        nv = summarize(V["never"])
        res[e][tag] = {str(t): {"dp": summarize(V[t])["dp_regret"], "myo": summarize(V[t])["norm_regret"]} for t in ts}
        res[e][tag]["never"] = {"dp": nv["dp_regret"], "myo": nv["norm_regret"]}
        res[e][tag]["spread_median"] = float(np.median(sp))
    return res


if __name__ == "__main__":
    r = run("dev", cells())
    crit = {}
    for t in T:
        v = []
        for e, c in r.items():
            for tag, x in c.items():
                if x["never"]["dp"] > 0 and x["never"]["myo"] > 0:
                    v.append(0.5 * (x[str(t)]["dp"] / x["never"]["dp"] + x[str(t)]["myo"] / x["never"]["myo"]))
        crit[t] = float(np.mean(v))
    print("criterion", crit)
    for e, c in r.items():
        print(e, "median spread", round(np.median([x["spread_median"] for x in c.values()]), 2),
              {t: (round(100 * np.mean([x[str(t)]["dp"] for x in c.values()]), 2), round(100 * np.mean([x[str(t)]["myo"] for x in c.values()]), 2)) for t in T})
    json.dump({"cells": r, "criterion": crit}, open(f"{IS}/results/dev_flat_switch.json", "w"), indent=1)
