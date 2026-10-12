"""Full acquire-then-decide episodes on the knowledge-required tracks (R5-E1), computed EXACTLY (no sampling) under the
reference model: credit/clinical records use the reference joint over (checks, Y) stored in each scenario; SEC filings use
the reference naive-Bayes model stored in each scenario. Policies re-plan after every observation with their FROZEN
elicited specification (no new LLM calls): the observed check is moved into the known findings of the specification.
The final action is taken with the policy's own belief. Regret = (DP value of the reference model - policy value) / l.
Policies: never, DIVA, DIVA-U, DIVA-kappa (dev-tuned cost discount, results/kappa.json), always (acquire every check in
DIVA's order). Writes results/episodes_know.json {env: {tag: {policy: {"regret", "n_acq", "n"}}}}."""
import sys, os, glob, json, ast, copy, collections
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read, choose_max
from eval_tab import diva_scores, diva_u
from eval_real import fio_real, fio_real_u

NMAX = int(os.environ.get("EPK_NMAX", "300"))
KAP = json.load(open(f"{IS}/results/kappa.json")) if os.path.exists(f"{IS}/results/kappa.json") else {}
M6 = 6
ERR = [0]
CONST = json.load(open(f"{IS}/results/const_strength.json")) if os.path.exists(f"{IS}/results/const_strength.json") else {}
_X = json.load(open(f"{IS}/results/extra_controls.json")) if os.path.exists(f"{IS}/results/extra_controls.json") else {}
EXC = dict(_X.get("knowledge", {})); EXC.update(_X.get("sec", {}))
CUR = {"tag": None, "env": None}
LAMACT = float(os.environ["EPK_LAMACT"]) if os.environ.get("EPK_LAMACT") else None
SPLIT = os.environ.get("EPK_SPLIT", "test"); ROW0 = int(os.environ.get("EPK_ROW0", "0"))
POLS = [p for p in os.environ.get("EPK_POLS", "").split(",") if p]
OUTNAME = os.environ.get("EPK_OUT", "episodes_know")
FIRST = [None]
SKIP_TAGS = set(os.environ.get("EPK_SKIP", "").split(",")) - {"", "none"}
# so the episode outputs used by tables never skip LLMs, whatever EPK_SKIP the caller passes.
if OUTNAME in ("episodes_know_k4", "episodes_know"):
    SKIP_TAGS = set()


def spec_after(kind, sc, fr, acq):
    """Specification state after acquiring checks acq = ((label, outcome), ...)."""
    if not acq:
        return sc, fr
    sc2 = dict(sc); fr2 = dict(fr)
    labs = {c["label"]: c for c in sc["checks"]}
    sc2["checks"] = [c for c in sc["checks"] if c["label"] not in dict(acq)]
    if kind == "tab":
        sc2["observed"] = list(sc.get("observed", [])) + [{"value": v, "key": labs[k]["key"]} for k, v in acq]
        fr2["obs"] = list(fr.get("obs", []))[: len(sc.get("observed", []))] + [fr["chk"].get(k, {}) for k, v in acq]
    else:
        sc2["observed"] = list(sc.get("observed", [])) + [{"bin": v} for k, v in acq]
        fr2["obs"] = list(fr.get("obs", []))[: len(sc.get("observed", []))] + [fr["chk"][k][v] if k in fr["chk"] and v < len(fr["chk"][k]) else [None] for k, v in acq]
    return sc2, fr2


def spec_dp_first(b, par, g, l):
    """Exact DP under the specified naive-Bayes model. par: {label: (l1 list, l0 list, cost)} (outcome likelihoods).
    Returns the first action ('S' if stopping is optimal within 1e-9)."""
    from functools import lru_cache
    labs = tuple(sorted(par))
    vstar = lambda q: max((1 - q) * g - q * l, 0.0)

    @lru_cache(maxsize=None)
    def V(bq, rem):
        q = bq / 1e9; best = vstar(q); arg = "S"
        for k in rem:
            l1, l0, c = par[k]; rest = tuple(x for x in rem if x != k); val = -c
            for v in range(len(l1)):
                pv = q * l1[v] + (1 - q) * l0[v]
                if pv > 1e-12:
                    val += pv * V(round(q * l1[v] / pv * 1e9), rest)[0]
            if val > best + 1e-9:
                best, arg = val, k
        return best, arg
    return V(round(b * 1e9), labs)[1]


def policy_step(kind, pol, sc, fr, kap):
    """Return (choice label or 'S', belief) for the current specification state."""
    gen = pol.startswith("gen_")
    if gen:  # LLM-free generic specification (const_strength.py), delta selected on development data
        from const_strength import spec as gspec
        cs = CONST.get(CUR["env"], {})
        fr = gspec(kind, sc, cs.get("delta_u" if pol == "gen_u" else "delta", 1.0))
        pol = {"gen_dp": "diva_dp", "gen_u": "diva_u", "gen_diva": "diva"}[pol]
    form, lam = ("lik", 1.0) if gen else ("cond", 0.75)
    if kind == "tab":
        f = diva_scores(sc, fr, form, "greedy", "elicit", lam); b = f["b"]
    else:
        f = fio_real(sc, fr, lam=lam); b = f["b0"]
    if pol == "beta":
        bet = KAP.get(f"{CUR['tag']}:{CUR['env']}", {}).get("beta", 0.0)
        return (choose_max({e: v + bet for e, v in f["net"].items()}) if sc["checks"] else "S"), b
    if pol == "faceig":
        from extra_controls import eig_cost_choice
        ex = EXC.get(f"{CUR['tag']}:{CUR['env']}", {})
        return (eig_cost_choice(sc, f["eig"], ex.get("lam", 0.1), ex.get("tau", 0.02)) if sc["checks"] else "S"), b
    if LAMACT is not None:  # belief used for the final action (calibration variant); acquisition decisions unchanged
        b = diva_scores(sc, fr, "cond", "greedy", "elicit", LAMACT)["b"] if kind == "tab" else fio_real(sc, fr, lam=LAMACT)["b0"]
    if not sc["checks"] or pol == "never":
        return "S", b
    if pol == "diva":
        return choose_max(f["net"]), b
    if pol == "always":
        return max(f["net"], key=f["net"].get), b
    if pol == "kappa":
        cost = {c["label"]: float(c["cost"]) for c in sc["checks"]}
        return choose_max({e: v + cost[e] * (1 - kap) for e, v in f["net"].items()}), b
    if pol == "diva_dp":
        g, l = float(sc["g"]), float(sc["l"])
        if kind == "tab":
            par = {k: ([1 - a, a], [1 - b_, b_], c) for k, (a, b_, c) in f["par"].items()}
        else:
            par = f["par"]
        return spec_dp_first(b, par, g, l), b
    if pol == "diva_u":
        fu = (diva_u(sc, fr, 1.0, form, lam) if kind == "tab" else fio_real_u(sc, fr, 1.0, lam))
        return choose_max(fu["net"]), b
    raise ValueError(pol)


def episode_value(kind, pol, sc, fr, kap):
    g, l = float(sc["g"]), float(sc["l"]); thr = g / (g + l)
    labels = [c["label"] for c in sc["checks"]]; cost = {c["label"]: float(c["cost"]) for c in sc["checks"]}
    if kind == "tab":
        J = np.asarray(sc["joint"], dtype=float).reshape(2 ** M6, 2)
        BITS = (np.arange(2 ** M6)[:, None] >> np.arange(M6)) & 1
        idx = {L: i for i, L in enumerate(sc.get("label_order") or labels)}
        def true_post(acq):
            w = np.ones(2 ** M6, bool)
            for k, v in acq:
                w &= BITS[:, idx[k]] == v
            p = J[w].sum(0); return p / p.sum(), p.sum()
        def outcomes(acq, k):
            base = true_post(acq)[1]
            return [(v, true_post(acq + ((k, v),))[1] / base) for v in (0, 1)]
    else:
        model = ast.literal_eval(sc["model"]) if isinstance(sc["model"], str) else sc["model"]
        pri = np.asarray(model["prior"], dtype=float); items = {it["label"]: np.asarray(it["lik"], dtype=float) for it in model["items"]}
        def post(acq):
            p = pri.copy()
            for k, v in acq:
                p = p * items[k][:, v]
            return p / p.sum()
        def true_post(acq):
            return post(acq), None
        def outcomes(acq, k):
            p = post(acq); pv = p @ items[k]
            return [(v, float(pv[v])) for v in range(items[k].shape[1])]
    n_acq = [0.0]

    def V(acq, prob):
        s2, f2 = spec_after(kind, sc, fr, acq)
        if pol == "direct1" and not acq:
            pp = "diva"; ch, b = policy_step(kind, "diva", s2, f2, kap); ch = FIRST[0]
        else:
            pp = pol if pol not in ("ask1", "direct1") else ("always" if (pol == "ask1" and not acq) else "diva")
            ch, b = policy_step(kind, pp, s2, f2, kap)
        if ch == "S" or ch not in cost:
            pt = true_post(acq)[0]; bt = pt[1]
            u_act = g * (1 - bt) - l * bt
            return np.array([u_act if b < thr else 0.0, max(u_act, 0.0)])
        n_acq[0] += prob
        val = np.array([-cost[ch], -cost[ch]])
        for v, pv in outcomes(acq, ch):
            if pv > 1e-12:
                val = val + pv * V(acq + ((ch, v),), prob * pv)
        return val
    v = V((), 1.0)
    return (sc["exact"]["dp_value"] - v[0]) / l, (sc["exact"]["dp_value"] - v[1]) / l, n_acq[0]


def cell(args):
    kind, envn, ep, fp, tag = args
    CUR["tag"], CUR["env"] = tag, envn
    envl = read(ep)[ROW0:NMAX]; fr = {r["id"]: r for r in read(fp)}
    kap = KAP.get(f"{tag}:{envn}", {}).get("kappa")
    if SPLIT == "dev":
        dp_ = (f"{IS}/outputs/tab_dev/dir_{tag}_{envn}.jsonl.gz" if kind == "tab" else f"{IS}/outputs/real_dev/dir_{tag}_{'ret1' if envn == 'ret_1' else envn}.jsonl.gz")
    else:
        dp_ = (f"{IS}/outputs/test_tab/dir_{tag}_{envn}.jsonl.gz" if kind == "tab" else f"{IS}/outputs/test_real/dir_{tag}_{envn}.jsonl.gz")
    DR = {r["id"]: max(r["dist"], key=r["dist"].get) for r in read(dp_) if r.get("dist")} if os.path.exists(dp_) else {}
    pols = (["never", "diva", "diva_u", "ask1", "always"] + (["kappa"] if kap is not None else []) + (["direct1"] if DR else [])) if not POLS else POLS
    R = collections.defaultdict(lambda: {"regret": [], "regret_ref": [], "n_acq": []})
    for sc in envl:
        if sc["id"] not in fr:
            continue
        for p in pols:
            if p == "direct1":
                if sc["id"] not in DR:
                    continue
                FIRST[0] = DR[sc["id"]]
            try:
                r, rr, na = episode_value(kind, p, sc, fr[sc["id"]], kap)
            except Exception as e:
                ERR[0] += 1
                continue
            R[p]["regret"].append(r); R[p]["regret_ref"].append(rr); R[p]["n_acq"].append(na); R[p].setdefault("ids", []).append(sc["id"])
    out = {p: {"regret": float(np.mean(v["regret"])), "regret_ref": float(np.mean(v["regret_ref"])), "n_acq": float(np.mean(v["n_acq"])), "n": len(v["regret"]),
               "vals_ref": [round(x, 6) for x in v["regret_ref"]], "vals": [round(x, 6) for x in v["regret"]], "ids": v["ids"]} for p, v in R.items() if v["regret"]}
    out["_errors"] = ERR[0]
    print(envn, tag, out.get("_errors"), {p: (round(100 * v["regret"], 2), round(100 * v["regret_ref"], 2), round(v["n_acq"], 2), v["n"]) for p, v in out.items() if p != "_errors"}, flush=True)
    return envn, tag, out


def main():
    jobs = []
    for ds in ["taiwan", "german", "cdc", "heart"]:
        pat = f"{IS}/outputs/tab_dev/fk_*_{ds}.jsonl.gz" if SPLIT == "dev" else f"{IS}/outputs/test_tab/fk_*_{ds}.jsonl.gz"
        for f in glob.glob(pat):
            tag = os.path.basename(f)[3:-9][: -(len(ds) + 1)]
            if tag in SKIP_TAGS:
                continue
            jobs.append(("tab", ds, f"{IS}/envs_tab/data/{ds}_{SPLIT}.jsonl.gz", f, tag))
    for tg in ["ret_1", "ret_5", "vol_5", "vol_20"]:
        dn = ("ret1" if tg == "ret_1" else tg) if SPLIT == "dev" else tg
        pat = f"{IS}/outputs/real_dev/fr_*_{dn}.jsonl.gz" if SPLIT == "dev" else f"{IS}/outputs/test_real/fr_*_{tg}.jsonl.gz"
        for f in glob.glob(pat):
            tag = os.path.basename(f)[3:-9][: -(len(dn) + 1)]
            if tag in SKIP_TAGS or tag.split("_")[-1] in ("joint", "ratio", "contrast"):
                continue
            jobs.append(("sec", tg, f"{IS}/envs_real/data/{tg}_{SPLIT}.jsonl.gz", f, tag))
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        jobs = jobs[:1]; global NMAX; NMAX = 20
    out = collections.defaultdict(dict)
    with Pool(int(os.environ.get("EPK_PROC", "24"))) as pool:
        for envn, tag, o in pool.imap_unordered(cell, jobs):
            out[envn][tag] = o
    # "Always ask" as an episode policy = ask at every step until no check remains, then decide (exact, reference posterior):
    # identical to "always"; stored under the canonical key always_top as well
    for e_ in out.values():
        for r_ in e_.values():
            if isinstance(r_, dict) and "always" in r_:
                r_["always_top"] = r_["always"]
    json.dump(out, open(f"{IS}/results/{OUTNAME}.json" if not (len(sys.argv) > 1) else f"{IS}/results/episodes_know_smoke.json", "w"), indent=1)


if __name__ == "__main__":
    main()
