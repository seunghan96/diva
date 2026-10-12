"""Correlated-evidence variant of the controlled binary environments (reviewer gap M3).
The original controlled checks are conditionally independent given Y, which is exactly the factored assumption of DIVA.
Here two checks share information beyond Y; the exact decision values (VOI, DP) are computed with the TRUE joint model
(hidden state s = (Y, Z), Z a shared latent), while every stated reliability is the exact marginal P(finding | Y).
Three variants (100 test scenarios each per domain, 300 per domain, interleaved):
  dup_known : check X was already run; candidate X' reads the same underlying records (X = Z, X' = Z with accuracy a in
              {1.0, 0.9, 0.8}); the text states that X' uses the same records and how often it agrees with X.
  cc_known  : check X was already run; candidate X2 (a different check) shares a nuisance source W ~ Bern(0.5):
              P(finding | y, w) = s_y + d (2w - 1), so both err in the same direction; the text states the shared source.
  dup_cand  : X and X' are both candidates (nothing run yet); X' agrees with X in a of cases (stated).
Within each variant, half of the scenarios are 'tempting': a factored model with the stated numbers would choose a
different first action from the true optimum (myopic or DP), the other half are not (quota by rejection sampling, as the
strata of the original environments).
The row field `model` holds the factored (stated-marginal) model, used by baselines only for the payoff matrix;
`model_true` holds the joint model; `exact` is computed from `model_true`.
usage: python scripts/corr_env.py [--n_per 100] [--workers 32]   -> envs_synth/data/corr/{domain}_test.jsonl.gz"""
import argparse, gzip, json, os, random, sys
from multiprocessing import Pool
import numpy as np

IS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(IS, "envs_synth"))
from core import Model, summarize_root  # noqa: E402
from gen_binary import (sample_scenario, sample_check, verbalize, build_model, _NPWrap, LABELS, DOM_SEED,  # noqa: E402
                        N_EPISODES)
from domains_bin import DOMAINS  # noqa: E402

VARIANTS = ["dup_known", "cc_known", "dup_cand"]
VAR_SEED = {"dup_known": 71, "cc_known": 72, "dup_cand": 73}
ACC = [1.0, 0.9, 0.8]


def _lik(p):
    return [1 - p, p]


def true_model(sc, corr):
    """Joint model over s = 2*y + z. corr: {'kind', 'x' (label or _obs index), 'x2' (label), 'acc' or 'd'}."""
    pi, g, l = sc["prior"], sc["g"], sc["l"]
    kind = corr["kind"]
    if kind == "cc":
        pz = lambda y, z: 0.5
    else:  # latent signal Z with the stated reliabilities of X
        s1, s0 = corr["zs1"], corr["zs0"]
        pz = lambda y, z: (s1 if y else s0) if z else (1 - (s1 if y else s0))
    states = [(y, z) for y in (0, 1) for z in (0, 1)]
    prior = [(pi if y else 1 - pi) * pz(y, z) for y, z in states]
    U = [[g if y == 0 else -l for y, z in states], [0.0] * 4]

    def lik_of(c, role):
        rows = []
        for y, z in states:
            if role == "x" and kind in ("dup",):
                p = 1.0 if z else 0.0
            elif role == "x2" and kind == "dup":
                a = corr["acc"]; p = a if z else 1 - a
            elif role in ("x", "x2") and kind == "cc":
                d = corr["d"][role]; p = (c["s1"] if y else c["s0"]) + d * (2 * z - 1)
            else:
                p = c["s1"] if y else c["s0"]
            rows.append(_lik(p))
        return rows

    items = [{"label": c["label"], "lik": lik_of(c, c.get("_role")), "cost": c["cost"], "outcomes": ["no finding", "finding"]}
             for c in sc["checks"]]
    n = len(items)
    obs_items = [{"label": f"_obs{j}", "lik": lik_of(o, o.get("_role")), "cost": 0.0, "outcomes": ["no finding", "finding"]}
                 for j, o in enumerate(sc["observed"])]
    return {"prior": prior, "utility": U, "items": items + obs_items,
            "observed": [[n + j, o["value"]] for j, o in enumerate(sc["observed"])], "budget": n,
            "states": [list(s) for s in states]}


def exact_y(M, sc):
    """`exact` field in the format of the original rows, from the joint model (beliefs reduced to P(Y=1))."""
    ex = summarize_root(M)
    labs = [c["label"] for c in sc["checks"]]
    b = np.asarray(ex["posterior"]); py1 = float(b[2] + b[3])
    out = {"posterior": py1, "eu_act": ex["eu"][0], "eu_pass": ex["eu"][1],
           "best_action": DOMAINS[sc["domain"]]["act"] if ex["best_action"] == 0 else DOMAINS[sc["domain"]]["pass"],
           "threshold": sc["g"] / (sc["g"] + sc["l"])}
    out["voi"] = {l: ex["voi"][l] for l in labs}
    out["eig"] = {l: ex["eig_dec"][l] for l in labs}  # information about Y (= the full-information optimal action)
    out["eig_state"] = {l: ex["eig"][l] for l in labs}
    out["net"] = {l: ex["net"][l] for l in labs}
    out["q"] = {l: ex["q"][l] for l in labs}
    for k in ["gold_choice", "gold_ties", "dp_value", "dp_first_action", "dp_first_ties", "myopic_value", "value_of_stopping"]:
        out[k] = ex[k]
    return out


def naive_exact(sc):
    """Same summary under the factored model with the stated marginals (what an ideal factored method computes)."""
    M = Model(build_model(sc))
    return summarize_root(M)


def rebuild_text(sc, parts, notes):
    opts = parts["option_lines"]
    for lab, note in notes.items():
        opts[lab] = opts[lab] + " " + note
    head = parts["options"].split("\n")[0]
    parts["options"] = head + "\n" + "\n".join(opts[c["label"]] for c in sc["checks"]) + "\n" + parts["stop_line"]
    text = "\n\n".join([parts["context"], parts["payoff"], parts["base_rate"], parts["known"], parts["options"]])
    return parts, text


def gen_one(rng, dom, var, want_tempt):
    D = DOMAINS[dom]
    for t in range(4000):
        sc = sample_scenario(rng, dom, n_obs=rng.choice([0, 0, 1]))
        lib_used = {c["name"] for c in sc["checks"]} | {o["name"] for o in sc["observed"]}
        free = [nm for nm in D["checks"] if nm[0] not in lib_used]
        cands = [c for c in sc["checks"] if not c["is_placebo"]]
        notes = {}
        if var in ("dup_known", "dup_cand"):
            s1, s0, _ = sample_check(rng, rng.choice(["strong", "moderate"]))
            a = rng.choice(ACC)
            name, ftxt = free[0]
            corr = {"kind": "dup", "zs1": s1, "zs0": s0, "acc": a}
            s1b, s0b = round(a * s1 + (1 - a) * (1 - s1), 4), round(a * s0 + (1 - a) * (1 - s0), 4)
            red = cands[0]  # the replaced candidate becomes the second copy X'
            red.update({"name": name + " (second provider)", "finding_text": ftxt, "s1": s1b, "s0": s0b, "kind": "redundant",
                        "_role": "x2"})
            if var == "dup_known":
                pf = sc["prior"] * s1 + (1 - sc["prior"]) * s0
                xo = {"name": name, "finding_text": ftxt, "s1": s1, "s0": s0, "value": None, "_role": "x"}
                sc["observed"].append(xo)
                ref = f"the already-done check '{name}'"
            else:
                x = cands[1]
                x.update({"name": name, "finding_text": ftxt, "s1": s1, "s0": s0, "kind": "redundant_source", "_role": "x"})
                ref = f"the check '{name}'"
            agree = ("always gives the same result as that check" if a == 1.0 else
                     f"gives the same result as that check in {int(round(100 * a))}% of cases")
            notes[red["label"]] = f"It uses the same underlying records as {ref} and {agree}."
        else:  # common cause with an already-done check
            got = []
            for _ in range(2):
                for _ in range(200):
                    s1, s0, _k = sample_check(rng, "moderate")
                    d = min(0.2, min(s1, s0, 1 - s1, 1 - s0) - 0.05)
                    d = int(d * 100) / 100
                    if d >= 0.1:
                        got.append((s1, s0, d)); break
            if len(got) < 2:
                continue
            (xs1, xs0, dx), (ys1, ys0, dy) = got
            name, ftxt = free[0]
            xo = {"name": name, "finding_text": ftxt, "s1": xs1, "s0": xs0, "value": None, "_role": "x"}
            sc["observed"].append(xo)
            red = cands[0]
            red.update({"s1": ys1, "s0": ys0, "kind": "common_cause", "_role": "x2"})
            corr = {"kind": "cc", "d": {"x": dx, "x2": dy}}
            notes[red["label"]] = (f"It relies on the same data source as the already-done check '{name}', so when one of the "
                                   f"two checks errs, the other tends to err in the same direction.")
        # sample the values of already-done checks from the TRUE joint model
        mt = true_model(sc, corr)
        # (values None for now -> sample state and outcomes)
        pri = np.asarray(mt["prior"]); s = int(rng.choice(4, p=pri / pri.sum()))
        n = len(sc["checks"])
        for j, o in enumerate(sc["observed"]):
            L = mt["items"][n + j]["lik"][s]
            o["value"] = int(rng.random() < L[1])
        mt = true_model(sc, corr)
        M = Model(mt)
        ex = exact_y(M, sc)
        nv = naive_exact(sc)
        tempt = (nv["gold_choice"] != ex["gold_choice"]) or (nv["dp_first_action"] != ex["dp_first_action"])
        if tempt != want_tempt:
            continue
        parts, _ = verbalize(sc, rng)
        parts, text = rebuild_text(sc, parts, notes)
        ex["stratum"] = "tempt" if tempt else "other"
        eps = []
        for _ in range(N_EPISODES):
            b = M.belief(M.observed0); st = int(rng.choice(4, p=b))
            eps.append({"state": st // 2, "y": st // 2, "z": st % 2,
                        "outcomes": {c["label"]: int(rng.random() < mt["items"][i]["lik"][st][1]) for i, c in enumerate(sc["checks"])}})
        labs = [c["label"] for c in sc["checks"]]
        row = {"track": "binary", "domain": dom, "family": "corr_" + var, "variant": var, "group_id": None,
               "stratum": ex["stratum"], "prior": sc["prior"], "g": sc["g"], "l": sc["l"], "cost_regime": sc["cost_regime"],
               "budget": len(sc["checks"]), "costs": {c["label"]: c["cost"] for c in sc["checks"]},
               "checks": [{k: c[k] for k in ["label", "name", "finding_text", "s1", "s0", "cost", "is_placebo", "kind"]} for c in sc["checks"]],
               "observed": [{k: o[k] for k in ["name", "finding_text", "value", "s1", "s0"]} for o in sc["observed"]],
               "label_order": labs, "actions": [D["act"], D["pass"]], "states": [D["benign_short"], D["adverse_short"]],
               "text_parts": parts, "text": text, "model": build_model(sc), "model_true": mt, "exact": ex,
               "corr": {"variant": var, "redundant": red["label"], "acc": corr.get("acc"), "d": corr.get("d"),
                        "naive_gold": nv["gold_choice"], "naive_dp_first": nv["dp_first_action"],
                        "x": (next((c["label"] for c in sc["checks"] if c.get("_role") == "x"), None))},
               "episodes": eps, "n_tries": t + 1}
        return row
    return None


def work(args):
    dom, var, idx, want = args
    seed = (DOM_SEED[dom] * 1_000_003 + 2000 * 7919 + idx * 31 + VAR_SEED[var] * 104729 + 99991) % (2 ** 32)
    rng = _NPWrap(random.Random(seed), np.random.default_rng(seed))
    r = gen_one(rng, dom, var, want)
    if r is not None:
        r["id"] = f"{dom}-test-corr-{var}-{idx:05d}"
        r["split"] = "test"; r["seed"] = seed
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per", type=int, default=100)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--domains", default="finance,credit,clinical,legal")
    a = ap.parse_args()
    for dom in a.domains.split(","):
        jobs = [(dom, v, i, i % 2 == 0) for v in VARIANTS for i in range(a.n_per)]
        with Pool(a.workers) as pool:
            rows = pool.map(work, jobs, chunksize=2)
        miss = sum(r is None for r in rows)
        rows = [r for r in rows if r is not None]
        # interleave variants so any prefix is balanced
        by = {v: [r for r in rows if r["variant"] == v] for v in VARIANTS}
        out = [by[v][i] for i in range(max(len(x) for x in by.values())) for v in VARIANTS if i < len(by[v])]
        p = os.path.join(IS, "envs_synth", "data", "corr", f"{dom}_test.jsonl.gz")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with gzip.open(p, "wt") as f:
            f.write("".join(json.dumps(r) + "\n" for r in out))
        from collections import Counter
        print(dom, len(out), "missing", miss, Counter((r["variant"], r["stratum"]) for r in out),
              "dp_stop", sum(r["exact"]["dp_first_action"] == "S" for r in out),
              "true_dp_buys_red", sum(r["exact"]["dp_first_action"] == r["corr"]["redundant"] for r in out), flush=True)


if __name__ == "__main__":
    main()
