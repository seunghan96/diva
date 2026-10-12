"""CPU-only controls requested by R2 (dev-tuned where there is a knob; tuned on DEV, applied once to TEST):
Controlled binary: Fac-EIG-cost (argmax EIG - lam*cost/l, acquire if > tau), dev-tuned (lam, tau) per model.
Knowledge tab:     knowledge-free DIVA (elicited likelihood pairs shuffled across checks, 5 seeds),
                   empirical naive Bayes from TRAINING records (no LLM; emp_s1/emp_s0),
                   threshold heuristic (acquire cheapest check if |b0 - b_dagger| < delta, dev-tuned delta),
                   Fac-EIG-cost dev-tuned.
Writes results/extra_controls.json; every comparison on identical row sets, with clustered bootstrap 95% CIs
(cluster = group_id or id) of DIVA minus control."""
import sys, os, json, glob, random, collections, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read, fac_scores, choose_max, score, summarize
from eval_tab import diva_scores, diva_u
from dcore import voi_exact, eig_exact, posterior

K = "dp_regret"
LAMS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
TAUS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]


def eig_cost_choice(sc, eig, lam, tau):
    L = float(sc["l"])
    sc_ = {c["label"]: eig[c["label"]] - lam * float(c["cost"]) / L for c in sc["checks"]}
    k = max(sc_, key=sc_.get)
    return k if sc_[k] > tau else "S"


def tune_eig(pairs):
    """pairs: list of (sc, eig dict) on DEV -> best (lam, tau) by mean dp_regret."""
    best = None
    for lam, tau in itertools.product(LAMS, TAUS):
        r = np.mean([score(sc, eig_cost_choice(sc, e, lam, tau))[K] for sc, e in pairs])
        if best is None or r < best[0]:
            best = (r, lam, tau)
    return best[1], best[2]


def boot_ci(diffs, clusters, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    by = collections.defaultdict(list)
    for d, c in zip(diffs, clusters):
        by[c].append(d)
    keys = list(by); arr = [np.array(by[k]) for k in keys]
    stats = []
    for _ in range(B):
        idx = rng.integers(0, len(keys), len(keys))
        tot = sum(arr[i].sum() for i in idx); n = sum(len(arr[i]) for i in idx)
        stats.append(tot / max(n, 1))
    return float(np.mean(diffs)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def controlled():
    out = {}
    for f in glob.glob(f"{IS}/outputs/dev_bin/facnum_*_finance.jsonl.gz"):
        tag = os.path.basename(f)[7:-9].rsplit("_", 1)[0]
        dev = []
        for d in ["finance", "credit", "clinical", "legal"]:
            env = {s["id"]: s for s in read(f"{IS}/envs_synth/data/binary/{d}_dev.jsonl.gz")}
            for r in read(f"{IS}/outputs/dev_bin/facnum_{tag}_{d}.jsonl.gz"):
                if r["id"] in env:
                    dev.append((env[r["id"]], fac_scores(env[r["id"]], r)["eig"]))
        if len(dev) < 400:
            continue
        lam, tau = tune_eig(dev)
        rows = collections.defaultdict(list); diffs = []; cl = []
        for d in ["finance", "credit", "clinical", "legal"]:
            env = {s["id"]: s for s in read(f"{IS}/envs_synth/data/binary/{d}_test.jsonl.gz")}
            for r in read(f"{IS}/outputs/test_bin/facnum_{tag}_{d}.jsonl.gz"):
                sc = env.get(r["id"])
                if sc is None:
                    continue
                fs = fac_scores(sc, r)
                a = score(sc, choose_max(fs["net"])); b = score(sc, eig_cost_choice(sc, fs["eig"], lam, tau))
                rows["diva"].append(a); rows["fac_eig_cost"].append(b)
                diffs.append(a[K] - b[K]); cl.append(sc.get("group_id") or sc["id"])
        out[tag] = {"lam": lam, "tau": tau, **{m: summarize(v) for m, v in rows.items()}, "diff_ci": boot_ci(diffs, cl)}
    return out


def knowledge():
    out = {}
    for f in glob.glob(f"{IS}/outputs/test_tab/fk_*.jsonl.gz"):
        tag, ds = os.path.basename(f)[3:-9].rsplit("_", 1)
        test = {s["id"]: s for s in read(f"{IS}/envs_tab/data/{ds}_test.jsonl.gz")}
        fr = {r["id"]: r for r in read(f)}
        ids = [i for i in test if i in fr]
        if len(ids) < 200:
            continue
        # dev tuning of the threshold heuristic and EIG-cost on the same model's dev elicitation (if present)
        devf = f"{IS}/outputs/tab_dev/fk_{tag}_{ds}.jsonl.gz"
        dev = {s["id"]: s for s in read(f"{IS}/envs_tab/data/{ds}_dev.jsonl.gz")}
        dfr = {r["id"]: r for r in read(devf)}
        dids = [i for i in dev if i in dfr]
        def heur(sc, b, delta):
            bd = float(sc["g"]) / (float(sc["g"]) + float(sc["l"]))
            if abs(b - bd) >= delta:
                return "S"
            return min(sc["checks"], key=lambda c: float(c["cost"]))["label"]
        deltas = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
        if len(dids) >= 60:
            dpairs = [(dev[i], diva_scores(dev[i], dfr[i], "cond", "greedy", "elicit", 0.75)) for i in dids]
            delta = min(deltas, key=lambda dl: np.mean([score(sc, heur(sc, f_["b"], dl))[K] for sc, f_ in dpairs]))
            lam, tau = tune_eig([(sc, f_["eig"]) for sc, f_ in dpairs])
            untuned = False
        else:
            delta, lam, tau = 0.05, 0.1, 0.02
            untuned = True
        rows = collections.defaultdict(list); diffs = collections.defaultdict(list); cl = []
        for i in ids:
            sc = test[i]
            f_ = diva_scores(sc, fr[i], "cond", "greedy", "elicit", 0.75)
            a = score(sc, choose_max(f_["net"])); rows["diva"].append(a)
            # knowledge-free: shuffle likelihood pairs across checks (5 seeds, averaged)
            labs = [c["label"] for c in sc["checks"]]; ks = []
            for seed in range(5):
                rnd = random.Random(hash((i, seed)) & 0xffffffff)
                perm = labs[:]; rnd.shuffle(perm)
                net = {}
                for lab, src in zip(labs, perm):
                    s1, s0, _ = f_["par"][src]; c = float(next(c["cost"] for c in sc["checks"] if c["label"] == lab))
                    net[lab] = voi_exact(f_["b"], s1, s0, float(sc["g"]), float(sc["l"])) - c
                ks.append(score(sc, choose_max(net))[K])
            kf = dict(a); kf[K] = float(np.mean(ks)); rows["knowledge_free"].append(kf)
            # DIVA-U and its knowledge-free version (shuffled specifications, same uncertainty)
            fu = diva_u(sc, fr[i], 1.0); rows["diva_u"].append(score(sc, choose_max(fu["net"])))
            ksu = []
            for seed in range(5):
                rnd = random.Random(hash((i, seed, "u")) & 0xffffffff)
                perm = labs[:]; rnd.shuffle(perm)
                netu = {lab: fu["net"][src] + f_["par"][src][2] - f_["par"][lab][2] for lab, src in zip(labs, perm)}
                ksu.append(score(sc, choose_max(netu))[K])
            kfu = dict(a); kfu[K] = float(np.mean(ksu)); rows["knowledge_free_u"].append(kfu)
            # empirical NB from training records (no LLM)
            b = float(sc["exact"]["posterior"])  # supervised reference: data-fitted belief + training-record NB likelihoods
            netE = {c["label"]: voi_exact(b, min(max(c["emp_s1"], .01), .99), min(max(c["emp_s0"], .01), .99), float(sc["g"]), float(sc["l"])) - float(c["cost"]) for c in sc["checks"]}
            rows["emp_nb"].append(score(sc, choose_max(netE)))
            rows["heuristic"].append(score(sc, heur(sc, f_["b"], delta)))
            rows["fac_eig_cost"].append(score(sc, eig_cost_choice(sc, f_["eig"], lam, tau)))
            rows["always_stop"].append(score(sc, "S"))
            cl.append(sc.get("group_id") or i)
            for m in ["knowledge_free", "emp_nb", "heuristic", "fac_eig_cost", "always_stop", "diva_u", "knowledge_free_u"]:
                diffs[m].append(a[K] - rows[m][-1][K])
        out[f"{tag}:{ds}"] = {"tuned": not untuned, "delta": delta, "lam": lam, "tau": tau, **{m: summarize(v) for m, v in rows.items()},
                               **{f"ci_vs_{m}": boot_ci(v, cl) for m, v in diffs.items()}}
    return out


def sec():
    from eval_real import fio_real
    out = {}
    for f in glob.glob(f"{IS}/outputs/test_real/fr_*.jsonl.gz"):
        b = os.path.basename(f)[3:-9]
        tg = next((t for t in ["ret_1", "ret_5", "vol_5", "vol_20"] if b.endswith(t)), None)
        if tg is None:
            continue
        tag = b[: -len(tg) - 1]
        dname = "ret1" if tg == "ret_1" else tg
        dev = {s["id"]: s for s in read(f"{IS}/envs_real/data/{tg}_dev.jsonl.gz")}
        dfr = {r["id"]: r for r in read(f"{IS}/outputs/real_dev/fr_{tag}_{dname}.jsonl.gz")}
        dids = [i for i in dev if i in dfr]
        if len(dids) < 100:
            continue
        lam, tau = tune_eig([(dev[i], fio_real(dev[i], dfr[i], lam=0.75)["eig"]) for i in dids])
        out[f"{tag}:{tg}"] = {"lam": lam, "tau": tau}
    return out


if __name__ == "__main__":
    res = {"controlled": controlled(), "knowledge": knowledge(), "sec": sec()}
    json.dump(res, open(f"{IS}/results/extra_controls.json", "w"), indent=1)
    for k, v in res["controlled"].items():
        print("bin", k, v["lam"], v["tau"], round(v["diva"][K], 4), round(v["fac_eig_cost"][K], 4), [round(x, 4) for x in v["diff_ci"]])
    for k, v in res["knowledge"].items():
        print("tab", k, {m: round(v[m][K], 4) for m in ["diva", "knowledge_free", "emp_nb", "heuristic", "fac_eig_cost", "always_stop"]})
