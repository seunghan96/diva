"""LLM-free constant-strength control. Every check gets the same generic strength delta (log-odds spread; findings
assumed to raise the risk, higher bins assumed to raise the risk), the base rate is the stated one, and the identical
DIVA / DIVA-U solver is applied. delta is selected on the development split of each environment (grid below).
Compares with the LLM-specified DIVA / DIVA-U (results/know_main.json, all test rows). Writes results/const_strength.json."""
import sys, os, json, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read, score, summarize, choose_max
from eval_tab import diva_scores, diva_u
from eval_real import fio_real, fio_real_u, stated_base_rate

GRID = [0.25, 0.5, 1.0, 2.0, 3.0]
NMAX = 1000
sg = lambda x: 1 / (1 + math.exp(-x)); lg = lambda p: math.log(p / (1 - p))


def spec(kind, sc, d):
    if kind == "tab":
        rec = {"l1": [sg(d / 2)], "l0": [sg(-d / 2)]}
        return {"prior": [float(sc["prior"])], "obs": [rec for _ in sc.get("observed", [])], "chk": {c["label"]: rec for c in sc["checks"]}}
    p = min(max(stated_base_rate(sc), 0.02), 0.98)
    def bins(n):
        return [[sg(lg(p) + d * ((j - (n - 1) / 2) / max((n - 1) / 2, 1)) / 2)] for j in range(n)]
    chk = {c["label"]: bins(len(c.get("bins", [])) or 3) for c in sc["checks"]}
    obs = [bins(max(3, int(o.get("bin", 1)) + 1))[int(o.get("bin", 1))] for o in sc.get("observed", [])]
    return {"prior": [p], "obs": obs, "chk": chk}


def choices(kind, sc, d):
    fr = spec(kind, sc, d)
    if kind == "tab":
        f = diva_scores(sc, fr, "lik", "greedy", "elicit", 1.0); fu = diva_u(sc, fr, 1.0, "lik", 1.0)
    else:
        f = fio_real(sc, fr, lam=1.0); fu = fio_real_u(sc, fr, 1.0, 1.0)
    return choose_max(f["net"]), choose_max(fu["net"])


def evaluate(kind, envl, d):
    a, b, nv = [], [], []
    for sc in envl:
        c1, c2 = choices(kind, sc, d)
        a.append(score(sc, c1)); b.append(score(sc, c2)); nv.append(score(sc, "S"))
    return summarize(a), summarize(b), summarize(nv)


def main():
    KM = json.load(open(f"{IS}/results/know_main.json"))
    out = {}
    for kind, envs in [("tab", ["taiwan", "german", "cdc", "heart"]), ("sec", ["ret_1", "ret_5", "vol_5", "vol_20"])]:
        for e in envs:
            base = f"{IS}/envs_tab/data/{e}" if kind == "tab" else f"{IS}/envs_real/data/{e}"
            dev = read(f"{base}_dev.jsonl.gz")[:NMAX]; test = read(f"{base}_test.jsonl.gz")[:NMAX]
            best = min(GRID, key=lambda d: evaluate(kind, dev, d)[0]["dp_regret"])
            bestu = min(GRID, key=lambda d: evaluate(kind, dev, d)[1]["dp_regret"])
            cd, _, nv = evaluate(kind, test, best); _, cu, _ = evaluate(kind, test, bestu)
            c3, _, n3 = evaluate(kind, test[:300], best); _, u3, _ = evaluate(kind, test[:300], bestu)
            recs = [r for r in KM.get(f"{kind}:{e}", {}).values() if "full:diva" in r and "full:diva_u" in r]
            llm = {m: float(np.mean([r[f"full:{m}"][k] for r in recs])) for m in ["diva", "diva_u", "never"] for k in ["dp_regret"]}
            llm_m = {m: float(np.mean([r[f"full:{m}"]["norm_regret"] for r in recs])) for m in ["diva", "diva_u", "never"]}
            out[e] = {"delta": best, "delta_u": bestu, "const_diva": cd["dp_regret"], "const_diva_u": cu["dp_regret"], "never": nv["dp_regret"],
                      "const_diva_myo": cd["norm_regret"], "const_diva_u_myo": cu["norm_regret"], "never_myo": nv["norm_regret"],
                      "llm_diva": llm["diva"], "llm_diva_u": llm["diva_u"], "llm_never": llm["never"],
                      "llm_diva_myo": llm_m["diva"], "llm_diva_u_myo": llm_m["diva_u"], "n_llm": len(recs),
                      "s300_diva": c3["dp_regret"], "s300_diva_u": u3["dp_regret"], "s300_diva_myo": c3["norm_regret"], "s300_diva_u_myo": u3["norm_regret"]}
            print(e, {k: (round(100 * v, 2) if isinstance(v, float) and k not in ("delta", "delta_u") else v) for k, v in out[e].items()}, flush=True)
    json.dump(out, open(f"{IS}/results/const_strength.json", "w"), indent=1)


if __name__ == "__main__":
    main()
