"""Tiny binary-track scenario generator in the envs_synth row format, ONLY for unit tests and smoke tests of the
baselines. Not the benchmark (envs_synth/ owns that); never report numbers from it."""
import random

import numpy as np

from . import policies as P
from .common import pct

D = {
    "role": "an equity analyst at an investment fund",
    "context": "A company has just released its quarterly results. You must decide whether to BUY its shares now or PASS.",
    "act": "BUY", "pass": "PASS", "adverse": "the stock underperforms the market over the next quarter",
    "adverse_short": "underperform", "benign_short": "outperform",
    "gain_text": "if you buy and the stock outperforms, you gain {g} points; if you buy and it underperforms, you lose {l} points. Passing yields 0 points either way.",
    "population": "comparable companies in this situation",
    "checks": [("Forward guidance", "management cut its forward guidance"),
               ("Gross margin trend", "gross margin declined versus last year"),
               ("Inventory levels", "inventories grew much faster than sales"),
               ("Insider trading filings", "executives sold a large amount of stock in the last month"),
               ("Analyst estimate revisions", "most analysts revised next-quarter estimates downward"),
               ("Auditor report", "the auditor flagged a material weakness in internal controls"),
               ("Order backlog", "the order backlog shrank versus last quarter")],
}


def make_scenario(i, seed=0, m=6, n_obs=1, n_episodes=4, budget=None):
    rng = random.Random(seed * 100003 + i)
    lib = list(D["checks"])
    rng.shuffle(lib)
    prior = rng.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    g, l = rng.choice([1, 2, 3]), rng.choice([2, 3, 5, 8])
    cmult = rng.choice([0, 0.05, 0.1, 0.2, 0.4])
    checks, observed = [], []
    for j, (name, ft) in enumerate(lib[:m + n_obs]):
        s1, s0 = round(rng.uniform(0.3, 0.95), 2), round(rng.uniform(0.05, 0.7), 2)
        if j == m - 1:
            s0 = s1  # placebo
        if j < m:
            checks.append({"label": "ABCDEFGH"[j], "name": name, "finding_text": ft, "s1": s1, "s0": s0,
                           "cost": round(cmult * min(g, l), 3), "is_placebo": s1 == s0})
        else:
            observed.append({"name": name, "finding_text": ft, "value": int(rng.random() < 0.5), "s1": s1, "s0": s0})
    opts = {c["label"]: (f"({c['label']}) {c['name']} (costs {c['cost']:g} points): reports whether {c['finding_text']}. "
                         f"If {D['adverse']}, it reports this {pct(c['s1'])} of the time; otherwise {pct(c['s0'])}.")
            for c in checks}
    known = "Already known:\n" + "\n".join(
        f"- {o['name']} (already done): {'it found that ' if o['value'] else 'it did not find that '}{o['finding_text']}. "
        f"If {D['adverse']}, this check reports this finding {pct(o['s1'])} of the time; otherwise {pct(o['s0'])}."
        for o in observed)
    items = [{"label": c["label"], "lik": [[1 - c["s0"], c["s0"]], [1 - c["s1"], c["s1"]]], "cost": c["cost"],
              "outcomes": ["no finding", "finding"]} for c in checks]
    obs_items = [{"label": f"_obs{j}", "lik": [[1 - o["s0"], o["s0"]], [1 - o["s1"], o["s1"]]], "cost": 0.0,
                  "outcomes": ["no finding", "finding"]} for j, o in enumerate(observed)]
    model = {"prior": [1 - prior, prior], "utility": [[g, -l], [0.0, 0.0]], "items": items + obs_items,
             "observed": [[m + j, o["value"]] for j, o in enumerate(observed)], "budget": budget or m}
    M = P.core.Model(model)
    ex = P.core.summarize_root(M)
    nrng = np.random.default_rng(seed * 7919 + i)
    eps = [P.core.sample_episode(M, nrng) for _ in range(n_episodes)]
    labs = [c["label"] for c in checks]
    return {"id": f"toy-{seed}-{i}", "track": "binary", "domain": "finance", "family": "toy", "stratum": "toy", "prior": prior, "g": g,
            "l": l, "budget": budget or m, "costs": {c["label"]: c["cost"] for c in checks}, "checks": checks,
            "observed": observed, "label_order": labs, "actions": ["BUY", "PASS"],
            "states": ["outperform", "underperform"],
            "text_parts": {"context": f"You are {D['role']}. " + D["context"],
                           "payoff": "Payoffs: " + D["gain_text"].format(g=g, l=l),
                           "base_rate": f"Base rate: in {pct(prior)} of {D['population']}, {D['adverse']}.",
                           "known": known, "option_lines": opts},
            "model": model, "exact": {**{k: ex[k] for k in ex}, "posterior": float(M.belief(M.observed0)[1])},
            "episodes": [{"state": e["state"], "y": e["state"], "outcomes": {labs[k]: e["outcomes"][k] for k in range(m)}}
                         for e in eps]}
