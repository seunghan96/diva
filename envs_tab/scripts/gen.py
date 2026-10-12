"""Knowledge-required tabular environments: scenarios from real held-out records, gold from the masked reference models.

Coherent preposterior (single joint per scenario): for the 6 candidate checks C and the observed evidence E,
  p(x_C | E)      = average over R=6 random chain orders of  prod_j P(x_{o_j} | E, x_{o_<j})   (masked f-models),
  P(Y=1 | E, x_C) = calibrated masked Y-model at each of the 64 leaves,
  J[x_C, y]       = p(x_C | E) P(y | E, x_C).
Every belief P(Y | E, x_S) and predictive P(x_e = v | E, x_S) used by VOI / EIG / DP is a marginal of J, so the
martingale property holds exactly (sum_v p(v) b_v = b, VOI >= 0), and the 3^6-state DP is consistent.
The direct reference-model belief P(Y | E) is also stored (exact.posterior_direct) with the coherence gap.
Run: python -I scripts/gen.py <dataset>"""
import sys
sys.path.append("<path>")
sys.path.insert(0, "./envs_tab/scripts")
import gzip, json, pickle, zlib
from collections import Counter
from multiprocessing import Pool
import numpy as np
import pandas as pd
import lightgbm as lgb
from features import DATASETS

B = "./envs_tab"
TOL = 1e-9          # one tolerance for ties / stop everywhere
CLIP = 1e-3         # all model probabilities clipped to [CLIP, 1-CLIP]
M = 6               # candidate checks per scenario
R_ORDERS = 6        # chain orders averaged for p(x_C | E)
G_SET, L_SET = [1, 2, 3], [2, 3, 5, 8]
COST_REGIMES = [0.0, 0.05, 0.1, 0.2, 0.4]
COST_MULT = [0.5, 1.0, 1.0, 1.5, 2.0]
LABELS = "ABCDEF"
N_NAT = {"taiwan": {"test": 2000, "dev": 500}, "cdc": {"test": 2000, "dev": 500},
         "german": {"test": 1000, "dev": 300}, "heart": {"test": 1000, "dev": 300}}
N_STRAT = {"taiwan": {"test": 1000, "dev": 250}, "cdc": {"test": 1000, "dev": 250},
           "german": {"test": 500, "dev": 150}, "heart": {"test": 500, "dev": 150}}
POOL_MULT = 8       # extra draws (same split records) from which the stratified subset is selected
STRATA_Q = {"S1": 0.35, "S2": 0.25, "S3": 0.25, "other": 0.15}
CFG = np.arange(2 ** M)
BITS = (CFG[:, None] >> np.arange(M)) & 1          # BITS[cfg, k] = value of candidate k (label order)


# ------------------------------------------------------------------ models
class Ref:
    def __init__(self, ds, feats):
        d = f"{B}/models/{ds}"
        self.y = lgb.Booster(model_file=f"{d}/ymodel.txt")
        self.calib = pickle.load(open(f"{d}/ycalib.pkl", "rb"))
        self.f = {j: lgb.Booster(model_file=f"{d}/fmodel_{f}.txt") for j, f in enumerate(feats)}

    def py(self, X):
        p = np.clip(self.y.predict(X, num_threads=8), 1e-6, 1 - 1e-6)
        if self.calib[0] == "platt":
            p = 1 / (1 + np.exp(-(self.calib[1] * np.log(p / (1 - p)) + self.calib[2])))
        else:
            p = np.interp(p, self.calib[1], self.calib[2])
        return np.clip(p, CLIP, 1 - CLIP)

    def pf(self, j, X):
        return np.clip(self.f[j].predict(X, num_threads=8), CLIP, 1 - CLIP)


# ------------------------------------------------------------------ text (envs_synth style, no reliabilities)
def num(x):
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return s if s else "0"


def pts(x):
    return f"{num(x)} point" + ("" if float(x) == 1 else "s")


def verbalize(D, base_rate, g, l, observed, checks, rng):
    context = f"You are {D['role']}. {D['context']}"
    payoff = "Payoffs: " + D["gain_text"].format(g=pts(g), l=pts(l))
    k = int(round(100 * base_rate))
    base = rng.choice([f"Base rate: among {D['population']}, about {k}% end up in the adverse case ({D['adverse']}).",
                       f"Base rate: in about {k}% of {D['population']}, {D['adverse']}."])
    known = [f"- {o['name']} (already done): " + (f"it found that {o['finding_text']}." if o["value"]
                                                  else f"it did not find that {o['finding_text']}.") for o in observed]
    known_txt = ("Already known:\n" + "\n".join(known)) if known else "Already known: nothing beyond the base rate."
    opts = {}
    for c in checks:
        cost = "free" if c["cost"] == 0 else f"costs {pts(c['cost'])}"
        opts[c["label"]] = f"({c['label']}) {c['name']} ({cost}): reports whether {c['finding_text']}."
    stop = f"(S) Stop and decide now ({D['act']} or {D['pass']})."
    options = ("You may run any of the following checks (each at most once) before deciding.\n" +
               "\n".join(opts[c["label"]] for c in checks) + "\n" + stop)
    parts = {"context": context, "payoff": payoff, "base_rate": base, "known": known_txt, "options": options,
             "option_lines": opts, "stop_line": stop, "budget": "",
             "note_optional": "How informative each check is, is not stated; judge it from your own knowledge."}
    return parts, "\n\n".join([context, payoff, base, known_txt, options])


def add_known(parts, check, value):
    """Render one extra known-finding line in the same style (for imagined-observation prompts)."""
    line = f"- {check['name']} (already done): " + (f"it found that {check['finding_text']}." if value
                                                    else f"it did not find that {check['finding_text']}.")
    k = parts["known"]
    return (k + "\n" + line) if k.startswith("Already known:\n") else ("Already known:\n" + line)


# ------------------------------------------------------------------ scenario draws
def draw(rng, rec_vals, feats_meta, feats, placebos):
    avail = [j for j in range(len(feats)) if not np.isnan(rec_vals[j])]
    nonp = [j for j in avail if feats[j] not in placebos]
    plac = [j for j in avail if feats[j] in placebos]
    n_obs = int(rng.choice([0, 0, 1, 1, 2]))
    use_p = bool(plac) and rng.random() < 0.5
    need = M - (1 if use_p else 0)
    n_obs = max(0, min(n_obs, len(nonp) - need))
    if len(nonp) < need:
        return None
    perm = list(rng.permutation(nonp))
    obs, cands = perm[:n_obs], perm[n_obs:n_obs + need]
    if use_p:
        cands.append(int(rng.choice(plac)))
    cands = [int(c) for c in rng.permutation(cands)]
    g, l = int(rng.choice(G_SET)), int(rng.choice(L_SET))
    regime = float(rng.choice(COST_REGIMES))
    costs = [round(regime * min(g, l) * float(rng.choice(COST_MULT)), 2) for _ in cands]
    return {"obs": [int(o) for o in obs], "cands": cands, "g": g, "l": l, "regime": regime, "costs": costs,
            "orders": [list(map(int, rng.permutation(M))) for _ in range(R_ORDERS)]}


def build_joints(ref, F, recs, drws):
    """Batched model queries -> J[s] (64 x 2) and the direct root belief for each scenario."""
    req = {j: [] for j in range(F)}     # model j -> list of (scen, order, step, rows)
    yrows, rootrows = [], []
    for s, (x, d) in enumerate(zip(recs, drws)):
        e0 = np.full(F, np.nan)
        for o in d["obs"]:
            e0[o] = x[o]
        rootrows.append(e0)
        leaf = np.repeat(e0[None], 2 ** M, 0)
        for k, c in enumerate(d["cands"]):
            leaf[:, c] = BITS[:, k]
        yrows.append(leaf)
        for r, perm in enumerate(d["orders"]):
            for j in range(M):
                t = np.arange(2 ** j)
                rows = np.repeat(e0[None], 2 ** j, 0)
                for i in range(j):
                    rows[:, d["cands"][perm[i]]] = (t >> i) & 1
                req[d["cands"][perm[j]]].append((s, r, j, rows))
    S = len(drws)
    out = {}
    for fj, lst in req.items():
        if not lst:
            continue
        X = np.concatenate([q[3] for q in lst])
        p = ref.pf(fj, X)
        pos = 0
        for (s, r, j, rows) in lst:
            out[(s, r, j)] = p[pos:pos + len(rows)]
            pos += len(rows)
    py = ref.py(np.concatenate(yrows)).reshape(S, 2 ** M)
    proot = ref.py(np.stack(rootrows))
    Js = []
    for s, d in enumerate(drws):
        px = np.zeros(2 ** M)
        for r, perm in enumerate(d["orders"]):
            pr = np.ones(2 ** M)
            for j in range(M):
                tj = sum(BITS[:, perm[i]] << i for i in range(j)) if j else np.zeros(2 ** M, int)
                p1 = out[(s, r, j)][tj]
                pr *= np.where(BITS[:, perm[j]] == 1, p1, 1 - p1)
            px += pr / R_ORDERS
        Js.append(np.stack([px * (1 - py[s]), px * py[s]], 1))
    return Js, proot


# ------------------------------------------------------------------ exact gold on the joint
def H2(b):
    b = min(max(b, 1e-15), 1 - 1e-15)
    return float(-(b * np.log2(b) + (1 - b) * np.log2(1 - b)))


def solve(J, costs, g, l):
    memo = {}
    U = lambda b: max(g * (1 - b) - l * b, 0.0)

    def stats(m, v):
        w = ((CFG & m) == v)
        tot = J[w].sum()
        return w, tot, J[w, 1].sum() / tot

    def value(m, v):
        key = (m, v)
        if key in memo:
            return memo[key]
        w, tot, b = stats(m, v)
        best = U(b)
        for i in range(M):
            if m >> i & 1:
                continue
            q = qval(m, v, i, w, tot)
            if q > best + TOL:
                best = q
        memo[key] = best
        return best

    def qval(m, v, i, w=None, tot=None):
        if w is None:
            w, tot, _ = stats(m, v)
        p1 = J[w & (BITS[:, i] == 1)].sum() / tot
        return -costs[i] + p1 * value(m | 1 << i, v | 1 << i) + (1 - p1) * value(m | 1 << i, v)

    w, tot, b = stats(0, 0)
    v0 = U(b)
    voi, eig, net, q, pfind = {}, {}, {}, {}, {}
    for i in range(M):
        p1 = J[BITS[:, i] == 1].sum() / tot
        b1 = J[BITS[:, i] == 1, 1].sum() / J[BITS[:, i] == 1].sum()
        b0 = J[BITS[:, i] == 0, 1].sum() / J[BITS[:, i] == 0].sum()
        vi = p1 * U(b1) + (1 - p1) * U(b0) - v0
        ei = H2(b) - p1 * H2(b1) - (1 - p1) * H2(b0)
        vi = 0.0 if abs(vi) < TOL else vi
        ei = 0.0 if abs(ei) < TOL else ei
        L = LABELS[i]
        voi[L], eig[L], net[L], pfind[L] = vi, ei, vi - costs[i], p1
        q[L] = qval(0, 0, i)
    V = value(0, 0)
    maxnet = max(net.values())
    if maxnet > TOL:
        gold_ties = [L for L in LABELS if net[L] >= maxnet - TOL]
        gold = max(gold_ties, key=lambda L: (net[L], -LABELS.index(L)))
    else:
        gold_ties, gold = ["S"], "S"
    if v0 >= V - TOL:
        dp_first, dp_ties = "S", ["S"]
    else:
        dp_ties = [L for L in LABELS if q[L] >= V - TOL]
        dp_first = max(dp_ties, key=lambda L: (q[L], -LABELS.index(L)))
    return {"posterior": float(b), "eu_act": float(g * (1 - b) - l * b), "eu_pass": 0.0,
            "threshold": g / (g + l), "voi": voi, "eig": eig, "net": net, "q": q, "p_finding": pfind,
            "gold_choice": gold, "gold_ties": gold_ties, "dp_value": float(V), "dp_first_action": dp_first,
            "dp_first_ties": dp_ties, "myopic_value": float(max(v0, v0 + maxnet)), "value_of_stopping": float(v0)}


def stratum(ex, tol=TOL):
    voi, eig, net = ex["voi"], ex["eig"], ex["net"]
    mv, me, maxnet = max(voi.values()), max(eig.values()), max(net.values())
    a_voi = {L for L, v in voi.items() if v >= mv - tol}
    a_eig = {L for L, v in eig.items() if v >= me - tol}
    if maxnet <= tol and ex["dp_first_action"] == "S" and me > 0.05:
        return "S2"
    if mv > tol and not (a_voi & a_eig) and max(voi[L] for L in a_eig) <= 0.5 * mv:
        return "S1"
    if mv > tol and (a_voi & a_eig) and maxnet > tol:
        return "S3"
    return "other"


def _solve_task(a):
    J, costs, g, l = a
    return solve(J, costs, g, l)


# ------------------------------------------------------------------ main
def make_rows(ds, split, family, recs_df, rids, seeds, ref, meta, feats, placebos, D):
    F = len(feats)
    X = recs_df.set_index("rid").loc[rids, feats].values.astype(float)
    ys = recs_df.set_index("rid").loc[rids, "y"].values.astype(int)
    drws, keep = [], []
    for k, (x, sd) in enumerate(zip(X, seeds)):
        d = draw(np.random.default_rng(sd), x, meta["findings"], feats, placebos)
        if d is not None:
            drws.append(d); keep.append(k)
    X, ys = X[keep], ys[keep]
    rids = [rids[k] for k in keep]; seeds = [seeds[k] for k in keep]
    Js, proot = build_joints(ref, F, X, drws)
    with Pool(16) as pool:
        exs = pool.map(_solve_task, [(J, d["costs"], d["g"], d["l"]) for J, d in zip(Js, drws)], chunksize=8)
    rows = []
    br = meta["base_rate_train"]
    for i, (x, d, J, ex) in enumerate(zip(X, drws, Js, exs)):
        rng = np.random.default_rng(seeds[i] + 1)
        checks = []
        for k, c in enumerate(d["cands"]):
            fm = meta["findings"][feats[c]]
            checks.append({"label": LABELS[k], "name": fm["name"], "finding_text": fm["finding_text"], "key": feats[c],
                           "cost": d["costs"][k], "is_placebo": bool(fm["is_placebo"]),
                           "emp_s1": round(fm["emp_s1"], 4), "emp_s0": round(fm["emp_s0"], 4),
                           "record_value": int(x[c])})
        observed = []
        for o in d["obs"]:
            fm = meta["findings"][feats[o]]
            observed.append({"name": fm["name"], "finding_text": fm["finding_text"], "key": feats[o], "value": int(x[o]),
                             "emp_s1": round(fm["emp_s1"], 4), "emp_s0": round(fm["emp_s0"], 4)})
        parts, text = verbalize(D, br, d["g"], d["l"], observed, checks, rng)
        ex["best_action"] = D["act"] if ex["eu_act"] > TOL else D["pass"]
        ex["posterior_direct"] = float(proot[i])
        ex["coherence_gap"] = float(ex["posterior"] - proot[i])
        ex["stratum"] = stratum(ex)
        ex["myopic_dp_disagree"] = ex["gold_choice"] not in ex["dp_first_ties"] and ex["dp_first_action"] not in ex["gold_ties"]
        ex["nonmyopic_acquire"] = ex["gold_choice"] == "S" and ex["dp_first_action"] != "S"
        rows.append({
            "id": f"{ds}-{split}-{family}-{i:05d}", "track": "tabular", "domain": D["domain"], "dataset": ds,
            "family": family, "group_id": f"{ds}-{split}-r{rids[i]}", "record_id": int(rids[i]),
            "stratum": ex["stratum"], "prior": round(br, 4), "g": d["g"], "l": d["l"], "cost_regime": d["regime"],
            "budget": M, "costs": {LABELS[k]: d["costs"][k] for k in range(M)}, "checks": checks, "observed": observed,
            "label_order": list(LABELS), "actions": [D["act"], D["pass"]], "states": [D["benign_short"], D["adverse_short"]],
            "text_parts": parts, "text": text, "exact": ex, "y_true": int(ys[i]), "split": split, "seed": int(seeds[i]),
            "joint": [float(v) for v in J.reshape(-1)],
        })
    return rows


def write(path, rows):
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main(ds):
    meta = json.load(open(f"{B}/clean/{ds}_meta.json"))
    df = pd.read_csv(f"{B}/clean/{ds}.csv.gz")
    feats = list(meta["findings"].keys())
    placebos = set(meta["placebos"])
    D = DATASETS[ds]
    ref = Ref(ds, feats)
    summary = {}
    for split, src in [("dev", "cal"), ("test", "test")]:
        base_seed = zlib.crc32(f"{ds}-{split}".encode())
        rng = np.random.default_rng(base_seed)
        recs = df[df.split == src]
        all_rids = recs.rid.values
        n = N_NAT[ds][split]
        if n <= len(all_rids):
            rids = list(rng.choice(all_rids, n, replace=False))
        else:  # small sets: every record used, then reused with fresh draws (same group_id)
            reps = int(np.ceil(n / len(all_rids)))
            rids = list(np.concatenate([rng.permutation(all_rids) for _ in range(reps)])[:n])
        seeds = [int(s) for s in rng.integers(0, 2 ** 31 - 1, size=len(rids))]
        nat = make_rows(ds, split, "natural", recs, rids, seeds, ref, meta, feats, placebos, D)
        write(f"{B}/data/{ds}_{split}.jsonl.gz", nat)
        # stratified subset: natural rows + extra draws from the same split's records, filled to quotas
        npool = POOL_MULT * len(rids)
        prids = list(rng.choice(all_rids, npool, replace=npool > len(all_rids)))
        pseeds = [int(s) for s in rng.integers(0, 2 ** 31 - 1, size=npool)]
        pool = make_rows(ds, split, "pool", recs, prids, pseeds, ref, meta, feats, placebos, D)
        cand = nat + pool
        by = {k: [r for r in cand if r["stratum"] == k] for k in STRATA_Q}
        ns = N_STRAT[ds][split]
        strat = []
        for k, q in STRATA_Q.items():
            want = int(round(q * ns))
            lst = by[k]
            idx = rng.permutation(len(lst))[:want]
            strat += [lst[j] for j in idx]
        strat = [strat[j] for j in rng.permutation(len(strat))]
        write(f"{B}/data/{ds}_{split}_strat.jsonl.gz", strat)
        summary[split] = {"n_natural": len(nat), "n_unique_records": len(set(r["record_id"] for r in nat)),
                          "strata_natural": dict(Counter(r["stratum"] for r in nat)),
                          "n_strat": len(strat), "strata_strat": dict(Counter(r["stratum"] for r in strat)),
                          "strata_pool_available": {k: len(v) for k, v in by.items()}}
        diag = diagnostics(nat)
        summary[split]["diagnostics"] = diag
        print(ds, split, json.dumps(summary[split]), flush=True)
    json.dump(summary, open(f"{B}/data/{ds}_summary.json", "w"), indent=1)


def diagnostics(rows):
    ex = [r["exact"] for r in rows]
    pl = [r["exact"]["voi"][c["label"]] for r in rows for c in r["checks"] if c["is_placebo"]]
    plf = [r["exact"]["voi"][c["label"]] / max(r["exact"]["voi"].values()) for r in rows for c in r["checks"]
           if c["is_placebo"] and max(r["exact"]["voi"].values()) > TOL]
    npl = [r["exact"]["voi"][c["label"]] for r in rows for c in r["checks"] if not c["is_placebo"]]
    plE = [r["exact"]["eig"][c["label"]] for r in rows for c in r["checks"] if c["is_placebo"]]
    nplE = [r["exact"]["eig"][c["label"]] for r in rows for c in r["checks"] if not c["is_placebo"]]
    gap = np.array([e["coherence_gap"] for e in ex])
    return {
        "frac_gold_stop": float(np.mean([e["gold_choice"] == "S" for e in ex])),
        "frac_dp_stop": float(np.mean([e["dp_first_action"] == "S" for e in ex])),
        "myopic_dp_disagree": float(np.mean([e["myopic_dp_disagree"] for e in ex])),
        "nonmyopic_acquire": float(np.mean([e["nonmyopic_acquire"] for e in ex])),
        "frac_with_placebo": float(np.mean([any(c["is_placebo"] for c in r["checks"]) for r in rows])),
        "placebo_voi_mean": float(np.mean(pl)) if pl else None,
        "placebo_voi_frac_pos": float(np.mean(np.array(pl) > TOL)) if pl else None,
        "placebo_voi_rel_to_max_mean": float(np.mean(plf)) if plf else None,
        "placebo_eig_mean": float(np.mean(plE)) if plE else None,
        "nonplacebo_voi_mean": float(np.mean(npl)), "nonplacebo_voi_frac_pos": float(np.mean(np.array(npl) > TOL)),
        "nonplacebo_eig_mean": float(np.mean(nplE)),
        "coherence_gap_abs_mean": float(np.abs(gap).mean()), "coherence_gap_abs_p95": float(np.quantile(np.abs(gap), 0.95)),
        "posterior_mean": float(np.mean([e["posterior"] for e in ex])), "y_mean": float(np.mean([r["y_true"] for r in rows])),
        "dp_gain_over_stop_mean": float(np.mean([e["dp_value"] - e["value_of_stopping"] for e in ex])),
    }


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        main(ds)
