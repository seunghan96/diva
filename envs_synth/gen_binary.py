"""Binary-outcome track (lead's DESIGN.md section 2 interface): Y in {0,1} (1 = adverse), actions {act, pass},
U(act,0)=+g, U(act,1)=-l, U(pass,.)=0, binary checks with stated reliabilities, exact VOI / EIG / DP.

usage: python gen_binary.py --domain finance --split test --n 3000 [--workers 32]
output: data/binary/{domain}_{split}.jsonl.gz  (rows interleaved so any prefix is stratum-balanced)
"""
import argparse, gzip, json, math, os, random, sys, time
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from core import Model, summarize_root, sample_episode
from domains_bin import DOMAINS

SPLIT_SEED = {"dev": 1000, "test": 2000, "pilot": 3000}
FAM_SEED = {"base": 1, "cost_variant": 2, "payoff_pair": 3, "many_item": 4}
DOM_SEED = {"finance": 11, "credit": 22, "clinical": 33, "legal": 44}
PRIORS = [round(0.10 + 0.05 * k, 2) for k in range(11)]          # 0.10..0.60
G_SET, L_SET = [1, 2, 3], [2, 3, 5, 8]
COST_REGIMES = [0.0, 0.05, 0.1, 0.2, 0.4]
LABELS = "ABCDEFGHIJKLMN"
N_EPISODES = 8

# family mix (fractions of the split); strata quotas are enforced on the whole split
FAMILY_MIX = {"base": 0.62, "cost_variant": 0.12, "payoff_pair": 0.10, "many_item": 0.16}
STRATA_TARGET = {"S1": 0.38, "S2": 0.27, "S3": 0.27, "other": 0.08}


# ------------------------------------------------------------------ number formats
def pct_text(p, rng):
    """Exact share of a population in one of several surface formats ("about 30%", "3 in 10")."""
    k = int(round(p * 100))
    fmts = [f"about {k}%", f"{k}%", f"roughly {k} percent"]
    if k % 10 == 0 and 0 < k < 100:
        fmts.append(f"{k // 10} in 10")
    elif k % 5 == 0 and 0 < k < 100:
        fmts.append(f"{k // 5} in 20")
    return rng.choice(fmts)


def rate_text(p, rng):
    """Exact frequency phrased as how often a check fires ("about 70% of the time", "in 7 out of 10 cases")."""
    k = int(round(p * 100))
    fmts = [f"about {k}% of the time", f"{k}% of the time", f"in roughly {k} percent of cases"]
    if k % 10 == 0 and 0 < k < 100:
        fmts.append(f"in {k // 10} out of 10 cases")
    elif k % 5 == 0 and 0 < k < 100:
        fmts.append(f"in {k // 5} out of 20 cases")
    return rng.choice(fmts)


def pts(x):
    return f"{num(x)} point" + ("" if float(x) == 1 else "s")


def num(x):
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return s if s else "0"


# ------------------------------------------------------------------ check reliabilities
def sample_check(rng, kind=None):
    """(s1, s0) on a 0.05 grid; s1 = P(finding|Y=1) >= s0 = P(finding|Y=0)."""
    kind = kind or rng.choice(["strong", "strong", "moderate", "moderate", "weak", "one_high", "one_low"])
    g = lambda a, b: round(rng.randint(int(round(a * 20)), int(round(b * 20))) / 20, 2)
    for _ in range(200):
        if kind == "strong":
            s1, s0 = g(0.55, 0.95), g(0.05, 0.25)
        elif kind == "moderate":
            s1, s0 = g(0.4, 0.8), g(0.15, 0.45)
        elif kind == "weak":
            s1 = g(0.2, 0.8); s0 = round(s1 - rng.choice([0.05, 0.1]), 2)
        elif kind == "one_high":          # finding is common either way: informative, rarely decision-changing
            s1, s0 = g(0.85, 0.95), g(0.6, 0.8)
        elif kind == "one_low":           # finding is rare either way
            s1, s0 = g(0.15, 0.35), g(0.05, 0.1)
        elif kind == "placebo":
            s1 = g(0.15, 0.85); s0 = s1
        if 0.05 <= s0 <= s1 <= 0.95 and (kind == "placebo" or s1 > s0):
            return s1, s0, kind
    raise RuntimeError(kind)


def build_model(sc):
    pi = sc["prior"]
    U = [[sc["g"], -sc["l"]], [0.0, 0.0]]       # actions: 0 = act, 1 = pass ; states: 0 = benign, 1 = adverse
    items = [{"label": c["label"], "lik": [[1 - c["s0"], c["s0"]], [1 - c["s1"], c["s1"]]], "cost": c["cost"],
              "outcomes": ["no finding", "finding"]} for c in sc["checks"]]
    # pre-observed checks are appended as extra (already revealed) items
    obs_items = [{"label": f"_obs{j}", "lik": [[1 - o["s0"], o["s0"]], [1 - o["s1"], o["s1"]]], "cost": 0.0,
                  "outcomes": ["no finding", "finding"]} for j, o in enumerate(sc["observed"])]
    n = len(items)
    m = {"prior": [1 - pi, pi], "utility": U, "items": items + obs_items,
         "observed": [[n + j, o["value"]] for j, o in enumerate(sc["observed"])],
         "budget": sc.get("budget") or n}
    return m


def stratum(ex, tol=1e-7):
    voi, eig, net = ex["voi"], ex["eig"], ex["net"]
    mv = max(voi.values())
    me = max(eig.values())
    a_voi = {l for l, v in voi.items() if v >= mv - tol}
    a_eig = {l for l, v in eig.items() if v >= me - tol}
    maxnet = max(net.values())
    if maxnet <= tol and ex["dp_first_action"] == "S" and me > 0.05:
        return "S2"
    if mv > tol and not (a_voi & a_eig) and max(voi[l] for l in a_eig) <= 0.5 * mv:
        return "S1"
    if mv > tol and (a_voi & a_eig) and maxnet > tol:
        return "S3"
    return "other"


# ------------------------------------------------------------------ scenario sampling
def sample_scenario(rng, dom, n_checks=6, budget=None, n_obs=None, regime=None, kinds=None):
    D = DOMAINS[dom]
    lib = list(range(len(D["checks"])))
    rng.shuffle(lib)
    n_obs = rng.choice([0, 0, 1, 1, 2]) if n_obs is None else n_obs
    g, l = rng.choice(G_SET), rng.choice(L_SET)
    pi = rng.choice(PRIORS)
    regime = rng.choice(COST_REGIMES) if regime is None else regime
    unit = min(g, l)
    checks = []
    kinds = kinds or []
    for j in range(n_checks):
        kind = "placebo" if j == 0 else (kinds[j] if j < len(kinds) else None)
        s1, s0, kind = sample_check(rng, kind)
        cost = round(regime * unit * rng.choice([0.5, 1.0, 1.0, 1.5, 2.0]), 2)
        name, ftxt = D["checks"][lib[j]]
        checks.append({"name": name, "finding_text": ftxt, "s1": s1, "s0": s0, "cost": cost,
                       "is_placebo": kind == "placebo", "kind": kind})
    rng.shuffle(checks)
    for j, c in enumerate(checks):
        c["label"] = LABELS[j]
    observed = []
    for j in range(n_obs):
        name, ftxt = D["checks"][lib[n_checks + j]]
        s1, s0, kind = sample_check(rng, rng.choice(["strong", "moderate", "weak"]))
        # value drawn from the prior predictive so pre-observed evidence is realistic
        p_find = pi * s1 + (1 - pi) * s0
        observed.append({"name": name, "finding_text": ftxt, "value": int(rng.random() < p_find), "s1": s1, "s0": s0})
    return {"domain": dom, "prior": pi, "g": g, "l": l, "cost_regime": regime, "checks": checks,
            "observed": observed, "budget": budget}


def verbalize(sc, rng):
    D = DOMAINS[sc["domain"]]
    adverse = D["adverse"]
    context = f"You are {D['role']}. {D['context']}"
    payoff = "Payoffs: " + D["gain_text"].replace("{g} points", "{g}").replace("{l} points", "{l}").format(
        g=pts(sc["g"]), l=pts(sc["l"]))
    base_rate = rng.choice([
        f"Base rate: among {D['population']}, {pct_text(sc['prior'], rng)} end up in the adverse case ({adverse}).",
        f"Base rate: in {int(round(sc['prior'] * 100))}% of {D['population']}, {adverse}.",
    ])
    known = []
    for o in sc["observed"]:
        res = (f"it found that {o['finding_text']}" if o["value"] else f"it did not find that {o['finding_text']}")
        known.append(f"- {o['name']} (already done): {res}. If {adverse}, this check reports this finding "
                     f"{rate_text(o['s1'], rng)}; otherwise {rate_text(o['s0'], rng)}.")
    known_txt = ("Already known:\n" + "\n".join(known)) if known else "Already known: nothing beyond the base rate."
    opts = {}
    for c in sc["checks"]:
        cost = "free" if c["cost"] == 0 else f"costs {pts(c['cost'])}"
        opts[c["label"]] = (f"({c['label']}) {c['name']} ({cost}): reports whether {c['finding_text']}. "
                            f"If {adverse}, it reports this {rate_text(c['s1'], rng)}; otherwise "
                            f"{rate_text(c['s0'], rng)}.")
    budget = ""
    if sc.get("budget") and sc["budget"] < len(sc["checks"]):
        budget = f"You may run at most {sc['budget']} checks in total."
    options = ("You may run any of the following checks (each at most once) before deciding" +
               (f"; {budget[0].lower() + budget[1:]}" if budget else ".") + "\n" +
               "\n".join(opts[c["label"]] for c in sc["checks"]) +
               f"\n(S) Stop and decide now ({D['act']} or {D['pass']}).")
    parts = {"context": context, "payoff": payoff, "base_rate": base_rate, "known": known_txt, "options": options,
             "option_lines": opts, "stop_line": f"(S) Stop and decide now ({D['act']} or {D['pass']}).", "budget": budget}
    text = "\n\n".join([context, payoff, base_rate, known_txt, options])
    return parts, text


def finalize(sc, rng, family, group_id=None, extra=None):
    m = build_model(sc)
    M = Model(m)
    ex = summarize_root(M)
    D = DOMAINS[sc["domain"]]
    pi_post = ex["posterior"][1]
    exact = {"posterior": pi_post, "posterior_vec": ex["posterior"],
             "eu_act": ex["eu"][0], "eu_pass": ex["eu"][1],
             "best_action": D["act"] if ex["best_action"] == 0 else D["pass"],
             "threshold": sc["g"] / (sc["g"] + sc["l"])}
    labs = [c["label"] for c in sc["checks"]]
    for k in ["voi", "eig", "eig_dec", "net", "q"]:
        exact[k] = {l: ex[k][l] for l in labs if l in ex[k]}
    for k in ["gold_choice", "gold_ties", "dp_value", "dp_first_action", "dp_first_ties", "myopic_value",
              "value_of_stopping"]:
        exact[k] = ex[k]
    exact["stratum"] = stratum(exact)
    eps = [sample_episode(M, rng) for _ in range(N_EPISODES)]
    parts, text = verbalize(sc, rng)
    row = {"track": "binary", "domain": sc["domain"], "family": family, "group_id": group_id,
           "stratum": exact["stratum"], "prior": sc["prior"], "g": sc["g"], "l": sc["l"],
           "cost_regime": sc["cost_regime"], "budget": sc.get("budget") or len(sc["checks"]),
           "costs": {c["label"]: c["cost"] for c in sc["checks"]},
           "checks": [{k: c[k] for k in ["label", "name", "finding_text", "s1", "s0", "cost", "is_placebo", "kind"]}
                      for c in sc["checks"]],
           "observed": sc["observed"], "label_order": labs,
           "actions": [D["act"], D["pass"]], "states": [D["benign_short"], D["adverse_short"]],
           "text_parts": parts, "text": text, "model": m, "exact": exact,
           "episodes": [{"state": e["state"], "y": e["state"],
                         "outcomes": {labs[i]: e["outcomes"][i] for i in range(len(labs))}} for e in eps]}
    if extra:
        row.update(extra)
    return row


# ------------------------------------------------------------------ families
def gen_base(rng, dom, want=None):
    """Sample until the stratum is `want` (None = accept anything)."""
    for t in range(5000):
        kinds = None
        if want == "S2":
            # sufficiency: a concentrated base rate / large costs make stopping optimal while checks still inform
            pass
        sc = sample_scenario(rng, dom)
        row = finalize(sc, rng, "base")
        if want is None or row["stratum"] == want:
            row["n_tries"] = t + 1
            return row
    return None


def gen_cost_variant(rng, dom, gid):
    """Same scenario, costs scaled x0.25 / x1 / x4 (regime > 0)."""
    sc = sample_scenario(rng, dom, regime=rng.choice([0.05, 0.1, 0.2]))
    rows = []
    for mult in [0.25, 1.0, 4.0]:
        s2 = json.loads(json.dumps(sc))
        for c in s2["checks"]:
            c["cost"] = round(c["cost"] * mult, 3)
        s2["cost_regime"] = round(sc["cost_regime"] * mult, 4)
        rows.append(finalize(s2, rng, "cost_variant", gid, {"cost_mult": mult}))
    return rows


def gen_payoff_pair(rng, dom, gid):
    """Same evidence, two payoff structures (decision thresholds) -> different VOI ranking."""
    sc = sample_scenario(rng, dom)
    rows = []
    pairs = [(g, l) for g in G_SET for l in L_SET if (g, l) != (sc["g"], sc["l"])]
    g2, l2 = rng.choice(pairs)
    for g, l in [(sc["g"], sc["l"]), (g2, l2)]:
        s2 = json.loads(json.dumps(sc))
        s2["g"], s2["l"] = g, l
        rows.append(finalize(s2, rng, "payoff_pair", gid))
    return rows


def gen_many(rng, dom):
    n = rng.choice([10, 10, 12])
    sc = sample_scenario(rng, dom, n_checks=n, budget=rng.choice([3, 4]))
    return finalize(sc, rng, "many_item")


def work(args):
    dom, split, family, idx, want = args
    seed = (DOM_SEED[dom] * 1_000_003 + SPLIT_SEED[split] * 7919 + idx * 31 + FAM_SEED[family] * 104729) % (2 ** 32)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    gid = f"{dom}-{split}-bin-{family}-{idx}"
    if family == "base":
        rows = [gen_base(_NPWrap(rng, np_rng), dom, want)]
    elif family == "cost_variant":
        rows = gen_cost_variant(_NPWrap(rng, np_rng), dom, gid)
    elif family == "payoff_pair":
        rows = gen_payoff_pair(_NPWrap(rng, np_rng), dom, gid)
    else:
        rows = [gen_many(_NPWrap(rng, np_rng), dom)]
    out = []
    for k, r in enumerate(rows):
        if r is None:
            continue
        r["id"] = f"{dom}-{split}-bin-{family}-{idx:05d}-{k}"
        r["split"] = split
        r["seed"] = seed
        out.append(r)
    return out


class _NPWrap:
    """random.Random for scenario sampling; numpy Generator.choice(n, p=...) for episodes."""
    def __init__(self, rng, np_rng):
        self.r, self.np = rng, np_rng

    def choice(self, a, p=None):
        if p is not None:
            return self.np.choice(a, p=p)
        return self.r.choice(a)

    def __getattr__(self, k):
        return getattr(self.r, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    n = a.n
    n_cv = int(n * FAMILY_MIX["cost_variant"]) // 3
    n_pp = int(n * FAMILY_MIX["payoff_pair"]) // 2
    n_mi = int(n * FAMILY_MIX["many_item"])
    jobs = [(a.domain, a.split, "cost_variant", i, None) for i in range(n_cv)]
    jobs += [(a.domain, a.split, "payoff_pair", i, None) for i in range(n_pp)]
    jobs += [(a.domain, a.split, "many_item", i, None) for i in range(n_mi)]
    with Pool(a.workers) as pool:
        fixed = [r for rs in pool.map(work, jobs, chunksize=4) for r in rs]
        cnt = {k: 0 for k in STRATA_TARGET}
        for r in fixed:
            cnt[r["stratum"]] += 1
        n_base = n - len(fixed)
        need = {}
        for k, f in STRATA_TARGET.items():
            need[k] = max(0, int(round(f * n)) - cnt[k])
        # rescale needs to exactly n_base
        tot = sum(need.values())
        wants = []
        for k in ["S1", "S2", "S3", "other"]:
            wants += [k] * int(round(need[k] * n_base / max(tot, 1)))
        while len(wants) < n_base:
            wants.append("S1")
        wants = wants[:n_base]
        bjobs = [(a.domain, a.split, "base", i, w) for i, w in enumerate(wants)]
        base = [r for rs in pool.map(work, bjobs, chunksize=8) for r in rs]
    rows = fixed + base
    # interleave by stratum so any prefix is balanced; groups stay contiguous
    rng = random.Random(SPLIT_SEED[a.split] + DOM_SEED[a.domain])
    units = {}
    for r in rows:
        key = r["group_id"] or r["id"]
        units.setdefault(key, []).append(r)
    ul = list(units.values())
    rng.shuffle(ul)
    rows = [r for u in ul for r in u]
    out = a.out or os.path.join(HERE, "data", "binary", f"{a.domain}_{a.split}.jsonl.gz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with gzip.open(out, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(json.dumps({"out": out, "n": len(rows), "secs": round(time.time() - t0, 1),
                      "strata": Counter(r["stratum"] for r in rows), "family": Counter(r["family"] for r in rows),
                      "gold_stop": sum(r["exact"]["gold_choice"] == "S" for r in rows)}))


if __name__ == "__main__":
    main()
