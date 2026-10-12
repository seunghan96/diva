"""Episodes + budget curves for factored-model policies (no LLM calls needed after elicitation), using the baselines'
episode() with common-random-number worlds. Policies: fiovoi (myopic), fiovoi_dp, fac_eig (tau from dev), and forced
budgets k = 0..K by VOI order (fiovoi_budget) and by EIG order (eig_budget). Also oracle variants using TRUE params.
usage: run_episodes_fac.py ENV FAC OUT [--reps 4] [--limit N] [--eig_tau 0.05] [--K 6]"""
import sys, os, json, gzip, asyncio, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from baselines import episode as E
from fiovoi_policy import FIOVOI
from evalx import read


def true_fac(sc):
    return {"prior": sc["prior"], "obs": [[o["s1"], o["s0"]] for o in sc.get("observed", [])],
            "chk": {c["label"]: [c["s1"], c["s0"]] for c in sc["checks"]}}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env"); ap.add_argument("fac"); ap.add_argument("out")
    ap.add_argument("--reps", type=int, default=4); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eig_tau", type=float, default=0.05); ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--oracle", action="store_true")
    a = ap.parse_args()
    env = read(a.env)[: a.limit or None]
    fac = {r["id"]: r for r in read(a.fac)}
    for r in fac.values():  # fallback for failed elicitations (counted at eval time)
        r["prior"] = 0.5 if r.get("prior") is None else r["prior"]
        r["obs"] = [[0.5 if x is None else x for x in o] for o in r.get("obs", [])]
        r["chk"] = {k: [0.5 if x is None else x for x in v] for k, v in r.get("chk", {}).items()}
    env = [s for s in env if s["id"] in fac]
    pols = [FIOVOI(fac, "myopic", name="fiovoi"), FIOVOI(fac, "dp", name="fiovoi_dp"),
            FIOVOI(fac, "eig", eig_tau=a.eig_tau, name="fac_eig"), FIOVOI(fac, "myopic", tau=1e9, name="never")]
    if a.oracle:
        tf = {s["id"]: true_fac(s) for s in env}
        pols += [FIOVOI(tf, "myopic", name="oracle_myopic"), FIOVOI(tf, "dp", name="oracle_dp")]
    rows = []
    for sc in env:
        for rep in range(a.reps):
            for p in pols:
                r = await E.episode(p, None, sc, rep=rep)
                ev = r.get("eval", {})
                rows.append(dict(id=r["id"], method=r["method"], rep=rep, n_acq=r["n_acq"], cost=r["cost"],
                                 **{k: ev.get(k) for k in ("cau_expected", "cau_realized", "policy_regret_norm", "decision_regret", "util_range", "under_ask", "decision_correct")}))
            for k in range(0, a.K + 1):
                for nm, mode in (("fiovoi_budget", "budget"), ("eig_budget", "eig_budget")):
                    if mode == "eig_budget":
                        p = FIOVOI(fac, "eig", eig_tau=-1.0, name=nm)
                    else:
                        p = FIOVOI(fac, "budget", name=nm)
                    r = await E.episode(p, None, sc, rep=rep, budget=k)
                    ev = r.get("eval", {})
                    rows.append({"id": r["id"], "method": nm, "rep": rep, "k": k, "cau_expected": ev.get("cau_expected"),
                                 "cost": r["cost"], "util_range": ev.get("util_range"), "decision_regret": ev.get("decision_regret")})
    with gzip.open(a.out, "wt") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")
    print("wrote", len(rows))

if __name__ == "__main__":
    asyncio.run(main())
