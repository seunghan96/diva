"""Aggregate results/f8_first.json (an_f8_first.py) into results/f8_summary.json: per environment means over the 20 LLMs
(None unless every LLM is complete) for (C) cost-aware EIG at the single development-selected lambda, (E) misspecified
payoffs/costs, (F) cost regimes, and (G) cross-fitted stop thresholds (per LLM and environment group)."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE); sys.path.insert(0, HERE)
from an_common import KNOW_LLMS, CDOMS, TAB, SEC

D = json.load(open(f"{IS}/results/f8_first.json")); R = D["res"]; LAMS = D["lams"]
ENVS = TAB + SEC + list(CDOMS)
ok = lambda e: e in R and all(R[e].get(t) for t in KNOW_LLMS)
mean_llm = lambda e, k: float(np.mean([np.mean([x for x in R[e][t]["vals"][k] if x is not None]) for t in KNOW_LLMS])) if ok(e) else None
out = {"envs": ENVS}
# (C) one lambda for all environments and LLMs: pooled development DP regret (every environment weighted equally)
dev = [np.mean([R[e][t]["eig_dev"][i] for e in ENVS if ok(e) for t in KNOW_LLMS if "eig_dev" in R[e][t]]) for i in range(len(LAMS))]
i_star = int(np.argmin(dev)); out["lam"] = LAMS[i_star]; out["eig_dev_curve"] = dict(zip(map(str, LAMS), dev))
out["C"] = {e: {"eigc": mean_llm(e, f"eigc{i_star}"), "diva": mean_llm(e, "diva"), "diva_u": mean_llm(e, "diva_u")} for e in ENVS}
out["C_testbest"] = {e: min(mean_llm(e, f"eigc{i}") for i in range(len(LAMS))) if ok(e) else None for e in ENVS}  # oracle lambda per env (diagnostic)
# (E)
out["E"] = {e: {f"{m}_{p}": mean_llm(e, f"pert_{m}_{p}") for m in ("diva", "divau") for p in D["pert"]} for e in ENVS}
# (F)
out["F"] = {e: {f"{m}_{s}": mean_llm(e, f"cs_{m}_{s}") for m in ("never", "diva", "divau", "dpstop") for s in D["scales"]} for e in TAB + SEC}
# (G) cross-fitting: per LLM and group (knowledge-required / controlled), tau chosen on one fold of scenario groups pooled over
# the group's environments, evaluated on the other fold, and vice versa; per environment means of the out-of-fold regrets
G = {}
for grp, envs in (("know", TAB + SEC), ("ctrl", list(CDOMS))):
    for key in ("thr_direct", "thr_ask"):
        per_env = {e: [] for e in envs}
        for t in KNOW_LLMS:
            rows = [(e, f, r) for e in envs if ok(e) for f, r in zip(R[e][t]["vals"]["fold"], R[e][t]["vals"][key]) if r is not None]
            if not rows:
                continue
            for f in (0, 1):
                train = [r for e, ff, r in rows if ff != f]
                j = int(np.argmin(np.mean(np.array(train), axis=0)))
                for e, ff, r in rows:
                    if ff == f:
                        per_env[e].append((t, r[j]))
        for e in envs:
            by = {}
            for t, v in per_env[e]:
                by.setdefault(t, []).append(v)
            G.setdefault(e, {})[key] = float(np.mean([np.mean(v) for v in by.values()])) if len(by) == len(KNOW_LLMS) else None
for e in ENVS:
    G[e]["direct"] = mean_llm(e, "direct"); G[e]["ask_p"] = mean_llm(e, "ask_p")
out["G"] = G
json.dump(out, open(f"{IS}/results/f8_summary.json", "w"), indent=1)
if __name__ == "__main__":
    f = lambda x: "  -  " if x is None else f"{100 * x:5.2f}"
    print("lambda*", out["lam"], {k: round(100 * v, 2) for k, v in out["eig_dev_curve"].items()})
    for sec, rows in (("C", ["eigc", "diva", "diva_u"]), ("E", ["diva_0.25", "diva_0.5", "divau_0.25", "divau_0.5"])):
        for r in rows:
            print(sec, f"{r:11s}", " ".join(f(out[sec][e][r]) for e in ENVS))
    print("C oracle-lam", " ".join(f(out["C_testbest"][e]) for e in ENVS))
    for r in ("never", "diva", "divau", "dpstop"):
        for s in D["scales"]:
            print("F", r, s, " ".join(f(out["F"][e][f"{r}_{s}"]) for e in TAB + SEC))
    for r in ("direct", "thr_direct", "ask_p", "thr_ask"):
        print("G", f"{r:10s}", " ".join(f(G[e][r]) for e in ENVS))
