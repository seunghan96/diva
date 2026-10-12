"""Multi-state track: factored hidden state (3 factors, 12-18 joint states), 3 actions, decision contexts
(finance horizons, loan types, clinical settings, litigation postures), multi-outcome evidence items, exact VOI/EIG/DP.

usage: python gen_ms.py --domain finance --split test --n 3000 [--workers 32]
output: data/ms/{domain}_{split}.jsonl.gz
"""
import argparse, gzip, itertools, json, math, os, random, sys, time
from collections import Counter
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from core import Model, summarize_root, sample_episode
from domains_ms import MS_DOMAINS
from gen_binary import stratum, _NPWrap

SPLIT_SEED = {"dev": 5000, "test": 6000}
DOM_SEED = {"finance": 101, "credit": 202, "clinical": 303, "legal": 404}
FAM_SEED = {"base": 1, "trap": 2, "sufficient": 3, "flip": 4, "context_pair": 5, "cost_variant": 6,
            "placebo_heavy": 7, "many_item": 8}
FAMILY_MIX = {"base": 0.15, "trap": 0.17, "sufficient": 0.13, "flip": 0.13, "context_pair": 0.15,
              "cost_variant": 0.09, "placebo_heavy": 0.06, "many_item": 0.12}
LABELS = "ABCDEFGHIJKLMNOP"
N_EPISODES = 8
COST_MULTS = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0]
N_ROBUST = 24

VERBAL = [  # (upper bound of the stated prior, phrase, plausible range a reader could hold)
    (0.10, "very unlikely", (0.02, 0.12)),
    (0.25, "unlikely", (0.08, 0.28)),
    (0.40, "a real possibility", (0.22, 0.45)),
    (0.60, "about as likely as not", (0.35, 0.62)),
    (0.80, "likely", (0.55, 0.82)),
    (1.01, "very likely", (0.78, 0.98)),
]


def verbal_bin(p):
    for ub, phrase, rngs in VERBAL:
        if p <= ub + 1e-9:
            return phrase, rngs


def round_grid(p, step=0.05, minp=0.05):
    """Round a probability vector to a step grid (each >= minp), summing to 1 (largest remainder)."""
    p = np.maximum(np.asarray(p, float), 1e-9)
    p = p / p.sum()
    n = len(p)
    units = int(round(1 / step))
    mins = int(round(minp / step))
    free = units - mins * n
    raw = np.maximum(p * units - mins, 0)
    raw = raw / raw.sum() * free if raw.sum() > 0 else np.full(n, free / n)
    base = np.floor(raw).astype(int)
    rem = free - base.sum()
    order = np.argsort(-(raw - base))
    base[order[:rem]] += 1
    return [round((b + mins) * step, 2) for b in base]


def fmt_money(x, D):
    if D["unit"] == "$k":
        v = round(x, 1)
        s = f"{abs(v):.1f}".rstrip("0").rstrip(".")
        return ("-" if v < 0 else "") + f"${s}k"
    v = round(x, 1)
    s = f"{abs(v):.1f}".rstrip("0").rstrip(".")
    return ("-" if v < 0 else "") + f"{s} points"


def signed(x, D):
    s = fmt_money(x, D)
    return s if s.startswith("-") or x == 0 else "+" + s


# ------------------------------------------------------------------ sampling
def sample_priors(rng, D, mode):
    pri = []
    for f in D["factors"]:
        base = np.array(f["prior"])
        if mode == "concentrated":
            k = rng.randrange(len(base))
            v = np.full(len(base), 0.05)
            v[k] = 1 - 0.05 * (len(base) - 1)
            mix = rng.choice([0.0, 0.05, 0.1])
            p = (1 - mix * 2) * v + mix * 2 * base
        else:
            alpha = {"diffuse": 3.0, "normal": 8.0}[mode]
            p = np.random.default_rng(rng.randrange(2 ** 31)).dirichlet(alpha * base)
        pri.append(round_grid(p))
    return pri


def sample_lik(rng, n_levels, n_out, strength):
    d = {"weak": rng.uniform(0.8, 2.0), "moderate": rng.uniform(3.0, 6.0), "strong": rng.uniform(8.0, 16.0)}[strength]
    rows = []
    for l in range(n_levels):
        x = l / (n_levels - 1)
        sc = np.array([-d * (x - o / (n_out - 1)) ** 2 for o in range(n_out)])
        sc += np.array([rng.gauss(0, 0.3) for _ in range(n_out)])
        p = np.exp(sc - sc.max())
        rows.append(round_grid(p))
    return rows


def sample_scenario(rng, dom, family, ctx=None, n_items=None, prior_mode=None, cost_mult=None):
    D = MS_DOMAINS[dom]
    ctx = ctx if ctx is not None else rng.randrange(len(D["contexts"]))
    C = D["contexts"][ctx]
    prior_mode = prior_mode or rng.choice(["diffuse", "normal", "normal", "concentrated"])
    priors = sample_priors(rng, D, prior_mode)
    size = rng.choice([0.5, 1, 2, 5]) if dom == "finance" else 1.0
    # ---- utility components (jittered, rounded); zeros stay zero
    jit = lambda v, step: 0.0 if v == 0 else round(v * rng.uniform(0.7, 1.3) / step) * step
    W = {}
    for a in D["actions"]:
        W[a] = {}
        for f in D["factors"]:
            k = f["key"]
            if a == D["actions"][D["baseline"]]:
                W[a][k] = [0.0] * len(f["levels"])
            elif dom == "finance":
                W[a][k] = None   # filled from returns below
            else:
                W[a][k] = [jit(v, 1.0 if dom == "clinical" else 5.0) for v in C["W"][a][k]]
    ret = None
    if dom == "finance":
        ret = {f["key"]: [jit(v, 0.5) for v in C["ret"][f["key"]]] for f in D["factors"]}
        for f in D["factors"]:
            k = f["key"]
            W["BUY"][k] = [round(size * 10 * r, 2) for r in ret[k]]
            W["SELL"][k] = [-round(size * 10 * r, 2) for r in ret[k]]
    const = {a: D["const"][a] * (size if dom == "finance" else 1.0) for a in D["actions"]}
    # ---- items
    relevant = {f["key"]: any(any(v != 0 for v in W[a][f["key"]]) for a in D["actions"]) for f in D["factors"]}
    lib = list(range(len(D["items"])))
    rng.shuffle(lib)
    n_plac = {"placebo_heavy": 3, "many_item": 2}.get(family, 1)
    if n_items is None:
        n_items = rng.choice([5, 6, 6, 7]) if family != "many_item" else min(rng.choice([12, 13, 14]), len(D["items"]) + len(D["placebos"]))
    n_real = n_items - n_plac
    if family == "trap":
        irr = [i for i in lib if not relevant[D["items"][i][1]]]
        rel = [i for i in lib if relevant[D["items"][i][1]]]
        k_irr = min(len(irr), rng.choice([1, 2]))
        chosen = irr[:k_irr] + rel[:n_real - k_irr]
    else:
        chosen = lib[:n_real]
    pl = list(range(len(D["placebos"])))
    rng.shuffle(pl)
    regime = cost_mult if cost_mult is not None else rng.choice(COST_MULTS)
    unit_cost = D["cost_scale"] * (size if dom == "finance" else 1.0)
    cstep = 0.1 if dom == "clinical" else 0.5
    items = []
    for i in chosen:
        name, fk, outs = D["items"][i]
        fi = [f["key"] for f in D["factors"]].index(fk)
        if family == "trap" and not relevant[fk]:
            strength = "strong"
        else:
            strength = rng.choice(["weak", "moderate", "moderate", "strong"])
        lik = sample_lik(rng, len(D["factors"][fi]["levels"]), len(outs), strength)
        for _ in range(20):
            if len({tuple(r) for r in lik}) > 1:
                break
            lik = sample_lik(rng, len(D["factors"][fi]["levels"]), len(outs), strength)
        items.append({"name": name, "factor": fk, "outcomes": outs, "lik_by_level": lik, "strength": strength,
                      "is_placebo": False, "decision_relevant_factor": relevant[fk]})
    for j in pl[:n_plac]:
        name, outs = D["placebos"][j]
        fk = rng.choice(D["factors"])["key"]
        fi = [f["key"] for f in D["factors"]].index(fk)
        row = round_grid(np.random.default_rng(rng.randrange(2 ** 31)).dirichlet([2.0] * len(outs)))
        items.append({"name": name, "factor": fk, "outcomes": outs,
                      "lik_by_level": [row[:] for _ in D["factors"][fi]["levels"]], "strength": "placebo",
                      "is_placebo": True, "decision_relevant_factor": relevant[fk]})
    for it in items:
        it["cost"] = round(regime * unit_cost * rng.choice([0.5, 1.0, 1.0, 1.5, 2.0]) / cstep) * cstep
        it["cost"] = round(it["cost"], 2)
    rng.shuffle(items)
    for j, it in enumerate(items):
        it["label"] = LABELS[j]
    budget = rng.choice([3, 3, 4]) if family == "many_item" else None
    return {"domain": dom, "context": C["key"], "ctx_index": ctx, "prior_mode": prior_mode, "size_musd": size,
            "factor_priors": priors, "W": W, "ret": ret, "const": const, "items": items, "cost_regime": regime,
            "budget": budget}


def build_model(sc):
    D = MS_DOMAINS[sc["domain"]]
    F = D["factors"]
    levels = [range(len(f["levels"])) for f in F]
    states = list(itertools.product(*levels))
    prior = [float(np.prod([sc["factor_priors"][fi][s[fi]] for fi in range(len(F))])) for s in states]
    U = []
    for a in D["actions"]:
        U.append([sc["const"][a] + sum(sc["W"][a][F[fi]["key"]][s[fi]] for fi in range(len(F))) for s in states])
    keys = [f["key"] for f in F]
    items = []
    for it in sc["items"]:
        fi = keys.index(it["factor"])
        items.append({"label": it["label"], "lik": [it["lik_by_level"][s[fi]] for s in states],
                      "cost": it["cost"], "outcomes": it["outcomes"]})
    return {"prior": prior, "utility": U, "items": items, "observed": [],
            "budget": sc["budget"] or len(items), "states": [list(s) for s in states]}


# ------------------------------------------------------------------ text
HEADERS = {
    "finance": ["EARNINGS DESK MEMO (issuer name withheld)", "PRE-EARNINGS POSITIONING NOTE: {sector} issuer",
                "Portfolio committee brief: earnings event, {sector} sector"],
    "credit": ["CREDIT COMMITTEE MEMO: small-business application", "Loan underwriting summary ({sector} borrower)",
               "Credit file: new application from a {sector} business"],
    "clinical": ["ED / urgent-care note: adult with acute respiratory illness", "Clinical decision note",
                 "Triage summary: cough, fever and breathlessness"],
    "legal": ["LITIGATION STRATEGY MEMO (privileged)", "Case assessment: breach-of-contract claim against the client",
              "Settlement decision brief"],
}
SECTORS = {"finance": ["consumer retail", "semiconductor", "industrial machinery", "software", "specialty chemicals", "medical devices"],
           "credit": ["restaurant", "logistics", "dental practice", "machine shop", "landscaping", "wholesale food"],
           "clinical": [""], "legal": [""]}
SITUATION = {
    "finance": "The company reports quarterly results shortly. You manage a ${size}M sleeve and must choose {acts} for a holding period of {horizon}.",
    "credit": "The bank is considering {horizon}. You must choose {acts}.",
    "clinical": "You are seeing the patient in {horizon}. You must choose {acts}.",
    "legal": "Your client is the defendant in {horizon}. The plaintiff has offered to settle. You must recommend {acts}.",
}
ROLE = {"finance": "You are a portfolio manager at an investment fund.", "credit": "You are a credit officer at a commercial bank.",
        "clinical": "You are the treating physician.", "legal": "You are counsel for the defendant company."}


def join_or(xs):
    return ", ".join(xs[:-1]) + " or " + xs[-1] if len(xs) > 1 else xs[0]


def prior_lines(sc, D, rng, qual):
    lines = []
    for fi, f in enumerate(D["factors"]):
        p = sc["factor_priors"][fi]
        order = sorted(range(len(p)), key=lambda l: -p[l])
        if not qual:
            body = ", ".join(f"{f['levels'][l]} {int(round(p[l] * 100))}%" for l in order)
            lines.append(rng.choice([f"- {f['name'][0].upper() + f['name'][1:]}: {body}.",
                                     f"- Our current probabilities for {f['name']}: {body}."]))
        else:
            parts = [f"{f['levels'][l]} is {verbal_bin(p[l])[0]}" for l in order]
            lines.append(f"- {f['name'][0].upper() + f['name'][1:]}: " + "; ".join(parts) + ".")
    tail = "These factors are independent of each other."
    return "Current assessment (before any further requests):\n" + "\n".join(lines) + "\n" + tail


def payoff_text(sc, D, rng):
    C = D["contexts"][sc["ctx_index"]]
    acts = D["actions"]
    base = acts[D["baseline"]]
    keys = [f["key"] for f in D["factors"]]
    out = []
    if sc["domain"] == "finance":
        rel = []
        for f in D["factors"]:
            r = sc["ret"][f["key"]]
            if any(v != 0 for v in r):
                rel.append(f"{f['short']} (" + ", ".join(f"{lv} {'+' if v > 0 else ''}{v:g}%" for lv, v in zip(f["levels"], r)) + ")")
        out.append(f"Return model for {C['horizon']}: the stock's return is the sum of contributions from " + "; ".join(rel) +
                   ". No other factor moves the return over this horizon.")
        out.append(f"BUY earns the return on ${sc['size_musd']:g}M; SELL earns the negative of the return on ${sc['size_musd']:g}M; "
                   f"HOLD earns $0. BUY or SELL also costs {fmt_money(-sc['const']['BUY'], D)} in trading costs. "
                   "(A 1% return on $1M is $10k.)")
    else:
        out.append(f"Payoffs in {D['unit_word']}, relative to {base}: each option's value is the sum of the contributions below"
                   + (" plus a fixed amount." if any(sc["const"][a] for a in acts) else "."))
        for a in acts:
            if a == base:
                continue
            comps = []
            for f in D["factors"]:
                w = sc["W"][a][f["key"]]
                if any(v != 0 for v in w):
                    comps.append(f"{f['short']}: " + ", ".join(f"{lv} {signed(v, D)}" for lv, v in zip(f["levels"], w)))
            fixed = f" Fixed amount: {signed(sc['const'][a], D)}." if sc["const"][a] else ""
            out.append(f"- {a}: " + ("; ".join(comps) if comps else "no factor contributions") + "." + fixed)
        out.append(f"{base} is worth 0. Factors not listed for an option do not change its value.")
    out.append(C["why"])
    return "\n".join(out)


def item_line(it, D, rng):
    f = [x for x in D["factors"] if x["key"] == it["factor"]][0]
    cost = "free" if it["cost"] == 0 else f"cost {fmt_money(it['cost'], D)}"
    rows = []
    for lv, row in zip(f["levels"], it["lik_by_level"]):
        rows.append(f"if {f['short']} is {lv}: " + ", ".join(f"'{o}' {int(round(p * 100))}%" for o, p in zip(it["outcomes"], row)))
    return (f"({it['label']}) {it['name']} ({cost}). Possible results: " + " / ".join(f"'{o}'" for o in it["outcomes"]) +
            ". How often each result appears: " + "; ".join(rows) + ".")


def verbalize(sc, rng):
    D = MS_DOMAINS[sc["domain"]]
    C = D["contexts"][sc["ctx_index"]]
    hdr = rng.choice(HEADERS[sc["domain"]]).format(sector=rng.choice(SECTORS[sc["domain"]]))
    acts = join_or([D["action_text"][a] for a in D["actions"]])
    sit = ROLE[sc["domain"]] + " " + SITUATION[sc["domain"]].format(size=f"{sc['size_musd']:g}", horizon=C["horizon"], acts=acts)
    pri_num = prior_lines(sc, D, rng, qual=False)
    pri_qual = prior_lines(sc, D, rng, qual=True)
    pay = payoff_text(sc, D, rng)
    opts = {it["label"]: item_line(it, D, rng) for it in sc["items"]}
    n = len(sc["items"])
    budget = f"You may make at most {sc['budget']} requests in total." if sc["budget"] and sc["budget"] < n else ""
    intro = ("Available requests (each at most once; costs are in " + D["unit_word"].split(" (")[0] + " and are subtracted from the payoff"
             + ("; " + budget[0].lower() + budget[1:-1] if budget else "") + "):")
    stop = f"(S) Stop requesting and decide now ({join_or(D['actions'])})."
    options = intro + "\n" + "\n".join(opts[it["label"]] for it in sc["items"]) + "\n" + stop
    parts = {"header": hdr, "situation": sit, "beliefs_numeric": pri_num, "beliefs_qualitative": pri_qual,
             "payoff": pay, "options": options, "option_lines": opts, "stop_line": stop, "budget": budget}
    t_num = "\n\n".join([hdr, sit, pri_num, pay, options])
    t_qual = "\n\n".join([hdr, sit, pri_qual, pay, options])
    return parts, t_num, t_qual


# ------------------------------------------------------------------ exact + robustness
def exact_of(sc):
    m = build_model(sc)
    M = Model(m)
    ex = summarize_root(M)
    labs = [it["label"] for it in sc["items"]]
    D = MS_DOMAINS[sc["domain"]]
    exact = {"posterior": ex["posterior"], "eu": dict(zip(D["actions"], ex["eu"])),
             "best_action": D["actions"][ex["best_action"]]}
    for k in ["voi", "eig", "eig_dec", "net", "q"]:
        exact[k] = {l: ex[k][l] for l in labs if l in ex[k]}
    for k in ["gold_choice", "gold_ties", "dp_value", "dp_first_action", "dp_first_ties", "myopic_value", "value_of_stopping"]:
        exact[k] = ex[k]
    # factor-marginal entropy reduction per item (bits) = EIG (items inform one factor; factors independent)
    exact["stratum"] = stratum(exact)
    return m, M, exact


def robustness(sc, exact, rng, n=N_ROBUST, with_dp=True):
    """Re-solve with priors drawn inside the qualitative phrase ranges; fraction of draws agreeing with the gold."""
    agree = Counter()
    for _ in range(n):
        s2 = dict(sc)
        pri = []
        for p in sc["factor_priors"]:
            v = np.array([rng.uniform(*verbal_bin(x)[1]) for x in p])
            pri.append(list(v / v.sum()))
        s2["factor_priors"] = pri
        m = build_model(s2)
        M = Model(m)
        t = M.node_table([])
        D = MS_DOMAINS[sc["domain"]]
        labs = M.labels
        net = {labs[i]: v for i, v in t["net"].items()}
        mx = max(net.values())
        gold = "S" if mx <= 1e-7 else max(net, key=net.get)
        voi = {labs[i]: v for i, v in t["voi"].items()}
        agree["gold_choice"] += (gold in exact["gold_ties"])
        agree["best_action"] += (D["actions"][t["best_action"]] == exact["best_action"])
        agree["top_voi"] += (max(voi, key=voi.get) == max(exact["voi"], key=exact["voi"].get))
        if with_dp:
            fm = M.dp_first([])
            dpa = "S" if fm["move"] == "S" else labs[fm["item"]]
            agree["dp_first"] += (dpa in exact["dp_first_ties"])
    r = {k: v / n for k, v in agree.items()}
    r["robust"] = bool(r["gold_choice"] >= 0.9 and r["best_action"] >= 0.9 and r.get("dp_first", 1.0) >= 0.9)
    return r


def finalize(sc, rng, family, group_id=None, extra=None, exact_pack=None):
    m, M, exact = exact_pack or exact_of(sc)
    D = MS_DOMAINS[sc["domain"]]
    labs = [it["label"] for it in sc["items"]]
    eps = [sample_episode(M, rng) for _ in range(N_EPISODES)]
    parts, t_num, t_qual = verbalize(sc, rng)
    rob = robustness(sc, exact, random.Random(rng.randrange(2 ** 31)), with_dp=(family != "many_item"))
    # designed-contrast flags (computed from exact values, not from intent)
    voi, eig = exact["voi"], exact["eig"]
    flags = {
        "has_voi0_eigpos": any(voi[l] == 0 and eig[l] > 0.01 for l in labs),
        "argmax_eig_voi0": (lambda l: voi[l] == 0 and eig[l] > 0)(max(eig, key=eig.get)) and max(voi.values()) > 0,
        "gold_stop": exact["gold_choice"] == "S",
        "dp_stop": exact["dp_first_action"] == "S",
        "decision_flip": max(voi.values()) > 0,
        "n_placebo": sum(it["is_placebo"] for it in sc["items"]),
    }
    row = {"track": "ms", "domain": sc["domain"], "family": family, "group_id": group_id, "stratum": exact["stratum"],
           "context": sc["context"], "prior_mode": sc["prior_mode"], "cost_regime": sc["cost_regime"],
           "budget": sc["budget"] or len(labs), "actions": D["actions"],
           "factors": [{"key": f["key"], "name": f["name"], "levels": f["levels"], "prior": sc["factor_priors"][fi]}
                       for fi, f in enumerate(D["factors"])],
           "W": sc["W"], "const": sc["const"], "ret": sc["ret"], "size_musd": sc["size_musd"],
           "costs": {it["label"]: it["cost"] for it in sc["items"]},
           "checks": [{k: it[k] for k in ["label", "name", "factor", "outcomes", "lik_by_level", "cost", "strength",
                                          "is_placebo", "decision_relevant_factor"]} for it in sc["items"]],
           "label_order": labs, "text_parts": parts, "text": t_num, "text_numeric": t_num, "text_qualitative": t_qual,
           "qual_robustness": rob, "flags": flags, "model": m, "exact": exact,
           "episodes": [{"state": e["state"], "state_levels": m["states"][e["state"]],
                         "outcomes": {labs[i]: e["outcomes"][i] for i in range(len(labs))}} for e in eps]}
    if extra:
        row.update(extra)
    return row


# ------------------------------------------------------------------ families
def accept(family, ex, sc):
    voi, eig, net = ex["voi"], ex["eig"], ex["net"]
    if family == "trap":
        la = max(eig, key=eig.get)
        return voi[la] == 0 and eig[la] > 0.05 and max(voi.values()) > 0
    if family == "sufficient":
        return ex["stratum"] == "S2"
    if family == "flip":
        return ex["stratum"] == "S3"
    return True


def gen_single(rng, dom, family, want=None):
    for t in range(4000):
        kw = {}
        if family == "sufficient":
            kw["prior_mode"] = rng.choice(["concentrated", "concentrated", "normal"])
            kw["cost_mult"] = rng.choice([0.5, 1.0, 2.0, 3.0])
        if family == "flip":
            kw["prior_mode"] = rng.choice(["diffuse", "normal"])
            kw["cost_mult"] = rng.choice([0.0, 0.25, 0.5, 1.0])
        if family == "trap":
            kw["cost_mult"] = rng.choice([0.0, 0.25, 0.5, 1.0])
        sc = sample_scenario(rng, dom, family, **kw)
        pack = exact_of(sc)
        ex = pack[2]
        if not accept(family, ex, sc):
            continue
        if want is not None and ex["stratum"] != want:
            continue
        r = finalize(sc, rng, family, exact_pack=pack)
        r["n_tries"] = t + 1
        return [r]
    return []


def gen_context_pair(rng, dom, gid):
    D = MS_DOMAINS[dom]
    sc = sample_scenario(rng, dom, "context_pair", ctx=0)
    rows = []
    for ci in range(len(D["contexts"])):
        s = json.loads(json.dumps(sc))
        s["ctx_index"], s["context"] = ci, D["contexts"][ci]["key"]
        # same jitter pattern: rescale the context's base W with the scenario's jitter ratios is not needed;
        # use the context's base W rounded (no jitter) so the pair differs only by context
        if dom == "finance":
            s["ret"] = {k: list(v) for k, v in D["contexts"][ci]["ret"].items()}
            for f in D["factors"]:
                s["W"]["BUY"][f["key"]] = [round(s["size_musd"] * 10 * r, 2) for r in s["ret"][f["key"]]]
                s["W"]["SELL"][f["key"]] = [-round(s["size_musd"] * 10 * r, 2) for r in s["ret"][f["key"]]]
        else:
            for a, comp in D["contexts"][ci]["W"].items():
                for k, v in comp.items():
                    s["W"][a][k] = [float(x) for x in v]
        relevant = {f["key"]: any(any(v != 0 for v in s["W"][a][f["key"]]) for a in D["actions"]) for f in D["factors"]}
        for it in s["items"]:
            it["decision_relevant_factor"] = relevant[it["factor"]]
        rows.append(finalize(s, rng, "context_pair", gid))
    return rows


def gen_cost_variant(rng, dom, gid):
    sc = sample_scenario(rng, dom, "cost_variant", cost_mult=rng.choice([0.5, 1.0]))
    rows = []
    for mult in [0.25, 1.0, 4.0]:
        s = json.loads(json.dumps(sc))
        for it in s["items"]:
            it["cost"] = round(it["cost"] * mult, 3)
        s["cost_regime"] = sc["cost_regime"] * mult
        rows.append(finalize(s, rng, "cost_variant", gid, {"cost_mult": mult}))
    return rows


def work(args):
    dom, split, family, idx, want = args
    seed = (DOM_SEED[dom] * 1_000_003 + SPLIT_SEED[split] * 7919 + idx * 31 + FAM_SEED[family] * 104729) % (2 ** 32)
    rng = _NPWrap(random.Random(seed), np.random.default_rng(seed))
    gid = f"{dom}-{split}-ms-{family}-{idx}"
    if family == "context_pair":
        rows = gen_context_pair(rng, dom, gid)
    elif family == "cost_variant":
        rows = gen_cost_variant(rng, dom, gid)
    else:
        rows = gen_single(rng, dom, family, want)
    for k, r in enumerate(rows):
        r["id"] = f"{dom}-{split}-ms-{family}-{idx:05d}-{k}"
        r["split"], r["seed"] = split, seed
    return rows


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
    D = MS_DOMAINS[a.domain]
    jobs = []
    for fam, frac in FAMILY_MIX.items():
        if fam == "base":
            continue
        k = int(n * frac)
        if fam == "context_pair":
            k //= len(D["contexts"])
        elif fam == "cost_variant":
            k //= 3
        jobs += [(a.domain, a.split, fam, i, None) for i in range(k)]
    with Pool(a.workers) as pool:
        rows = [r for rs in pool.imap_unordered(work, jobs, chunksize=2) for r in rs]
        n_base = n - len(rows)
        wants = ["S1", "S2", "S3", "other"] * (n_base // 4 + 1)
        bjobs = [(a.domain, a.split, "base", i, w) for i, w in enumerate(wants[:n_base])]
        rows += [r for rs in pool.imap_unordered(work, bjobs, chunksize=2) for r in rs]
    rows.sort(key=lambda r: r["id"])
    rng = random.Random(SPLIT_SEED[a.split] + DOM_SEED[a.domain])
    units = {}
    for r in rows:
        units.setdefault(r["group_id"] or r["id"], []).append(r)
    ul = list(units.values())
    rng.shuffle(ul)
    rows = [r for u in ul for r in u]
    out = a.out or os.path.join(HERE, "data", "ms", f"{a.domain}_{a.split}.jsonl.gz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with gzip.open(out, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"out": out, "n": len(rows), "secs": round(time.time() - t0, 1),
                      "strata": Counter(r["stratum"] for r in rows), "family": Counter(r["family"] for r in rows),
                      "gold_stop": sum(r["flags"]["gold_stop"] for r in rows),
                      "argmax_eig_voi0": sum(r["flags"]["argmax_eig_voi0"] for r in rows),
                      "qual_robust": sum(r["qual_robustness"]["robust"] for r in rows)}))


if __name__ == "__main__":
    main()
