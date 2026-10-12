"""additional analyses answered on FIRST decisions from existing outputs (no LLM calls), all 20 LLMs, first 300 test scenarios:
 (C) cost-aware EIG on the elicited specification, EIG - lambda * c / l (acquire the arg max if positive, else stop), with ONE
     lambda selected on the development splits for all environments and LLMs (pooled mean DP regret);
 (E) payoff and cost misspecification: DIVA / DIVA-U decide with g, l and every cost multiplied by independent factors drawn
     uniformly from [1-p, 1+p] (p = 0.25, 0.5; fixed seed per scenario) and are scored with the true values;
 (F) cost regimes on the knowledge-required environments: all costs scaled by 0.25..4 for the decision AND the reference
     DP optimum (recomputed exactly, an_f8_common.exact_values); the elicited specification has no cost input;
 (G) stop-threshold baselines: Direct stops when its stop probability >= tau, Ask-P when its ask probability < tau, else
     they take their most likely check; tau is selected per LLM and environment group by 2-fold cross-fitting over scenario
     groups (no development outputs exist for these baselines), which favours the baselines.
Writes results/f8_first.json."""
import os, sys, json, zlib, collections
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE); sys.path.insert(0, HERE)
from evalx import read, score, choose_max
from an_common import load, spec_scores, spec_u, KNOW_LLMS, CDOMS, TAB, SEC
from an_f8_common import exact_values, dp_regret, rescale

LAMS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
PERT = [0.25, 0.5]
SCALES = [0.25, 0.5, 1.0, 2.0, 4.0]
TAUS = ["orig", 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]
ENVS = TAB + SEC + list(CDOMS)
K = "dp_regret"
EXACT = {}  # (env, id, scale) -> (q, vs, dpv), filled before the pool forks


def eigc(sc, eig, lam):
    L = float(sc["l"]); s = {c["label"]: eig[c["label"]] - lam * float(c["cost"]) / L for c in sc["checks"]}
    k = max(s, key=s.get)
    return k if s[k] > 0 else "S"


def dev_rows(e, tag):
    if e in CDOMS:
        env, sp = f"{IS}/envs_synth/data/binary/{e}_dev.jsonl.gz", f"{IS}/outputs/dev_bin/facnum_{tag}_{e}.jsonl.gz"
    elif e in TAB:
        env, sp = f"{IS}/envs_tab/data/{e}_dev.jsonl.gz", f"{IS}/outputs/tab_dev/fk_{tag}_{e}.jsonl.gz"
    else:
        env, sp = f"{IS}/envs_real/data/{e}_dev.jsonl.gz", f"{IS}/outputs/real_dev/fr_{tag}_{'ret1' if e == 'ret_1' else e}.jsonl.gz"
    if not os.path.exists(sp):
        return None
    fr = {r["id"]: r for r in read(sp)}
    return [(sc, fr[sc["id"]]) for sc in read(env) if sc["id"] in fr][:300]


def perturb(sc, p):
    rng = np.random.default_rng(zlib.crc32(f"{sc['id']}|{p}".encode()))
    f = lambda: float(rng.uniform(1 - p, 1 + p))
    s = rescale(sc, 1.0, g=float(sc["g"]) * f(), l=float(sc["l"]) * f())
    s["checks"] = [dict(c, cost=float(c["cost"]) * f()) for c in sc["checks"]]
    return s


def fold(sc):
    return zlib.crc32(str(sc.get("group_id") or sc["id"]).encode()) % 2


def one(args):
    e, tag = args
    rows = load(e, tag)
    if rows is None:
        return e, tag, None
    out = {"n": len(rows)}
    # (C) dev curve and test curve over lambda
    dv = dev_rows(e, tag)
    if dv:
        eigs = [(sc, spec_scores(e, sc, fr)["eig"]) for sc, fr in dv]
        out["eig_dev"] = [float(np.mean([score(sc, eigc(sc, eg, lam))[K] for sc, eg in eigs])) for lam in LAMS]
        out["n_dev"] = len(eigs)
    R = collections.defaultdict(list)
    ask = {}
    if True:
        p = f"{IS}/baselines/outputs/{tag}/finrisk_ask/{e}_test.jsonl.gz"
        if os.path.exists(p):
            ask = {r["id"]: r for r in read(p) if r.get("choice") is not None}
    for sc, fr, dist in rows:
        fs = spec_scores(e, sc, fr); nu = spec_u(e, sc, fr)
        R["diva"].append(score(sc, choose_max(fs["net"]))[K])
        R["diva_u"].append(score(sc, choose_max(nu if nu is not None else fs["net"]))[K])
        for i, lam in enumerate(LAMS):
            R[f"eigc{i}"].append(score(sc, eigc(sc, fs["eig"], lam))[K])
        for p_ in PERT:  # (E)
            s2 = perturb(sc, p_)
            R[f"pert_diva_{p_}"].append(score(sc, choose_max(spec_scores(e, s2, fr)["net"]))[K])
            n2 = spec_u(e, s2, fr)
            R[f"pert_divau_{p_}"].append(score(sc, choose_max(n2 if n2 is not None else spec_scores(e, s2, fr)["net"]))[K])
        if e not in CDOMS:  # (F)
            for s_ in SCALES:
                q, vs, dpv = EXACT[(e, sc["id"], s_)]
                s2 = rescale(sc, s_); l = float(sc["l"])
                R[f"cs_never_{s_}"].append(dp_regret("S", q, vs, dpv, l))
                R[f"cs_diva_{s_}"].append(dp_regret(choose_max(spec_scores(e, s2, fr)["net"]), q, vs, dpv, l))
                R[f"cs_divau_{s_}"].append(dp_regret(choose_max(spec_u(e, s2, fr)), q, vs, dpv, l))
                R[f"cs_dpstop_{s_}"].append(float(vs >= dpv - 1e-9))
        # (G) per-scenario regrets of every threshold (fold kept for cross-fitting)
        labs = [c["label"] for c in sc["checks"]]
        top = max(labs, key=lambda k: dist.get(k, 0.0)); pS = float(dist.get("S", 0.0))
        orig = max(dist, key=dist.get)
        R["fold"].append(fold(sc))
        R["direct"].append(score(sc, orig if orig in labs or orig == "S" else top)[K])
        R["thr_direct"].append([score(sc, (orig if orig in labs or orig == "S" else top) if t == "orig" else ("S" if pS >= t else top))[K] for t in TAUS])
        a = ask.get(sc["id"])
        if a is not None:
            sco = a.get("scores") or {}; pa = float((a.get("info") or {}).get("p_ask", 1.0 if a["choice"] != "S" else 0.0))
            atop = max(labs, key=lambda k: sco.get(k, 0.0)) if sco else (a["choice"] if a["choice"] in labs else labs[0])
            R["ask_p"].append(score(sc, a["choice"])[K])
            R["thr_ask"].append([score(sc, a["choice"] if t == "orig" else ("S" if pa < t else atop))[K] for t in TAUS])
        else:
            R["ask_p"].append(None); R["thr_ask"].append(None)
    out["vals"] = {k: v for k, v in R.items()}
    return e, tag, out


def exact_job(args):
    e, i, sc = args
    return [((e, sc["id"], s_), exact_values(sc, s_)) for s_ in SCALES]


def main():
    from an_common import env_rows
    jobs = [(e, i, sc) for e in TAB + SEC for i, sc in enumerate(env_rows(e)[:300])]
    with Pool(int(os.environ.get("F8_PROC", "48"))) as pool:
        for lst in pool.imap_unordered(exact_job, jobs, chunksize=8):
            for k, v in lst:
                EXACT[k] = v
    print("exact values", len(EXACT), flush=True)
    res = {}
    with Pool(int(os.environ.get("F8_PROC", "48"))) as pool:
        for e, tag, o in pool.imap_unordered(one, [(e, t) for e in ENVS for t in KNOW_LLMS]):
            res.setdefault(e, {})[tag] = o
            print(e, tag, None if o is None else (o["n"], o.get("n_dev")), flush=True)
    json.dump({"lams": LAMS, "pert": PERT, "scales": SCALES, "taus": [str(t) for t in TAUS], "res": res}, open(f"{IS}/results/f8_first.json", "w"))


if __name__ == "__main__":
    main()
