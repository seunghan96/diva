"""Realized-utility episodes on REAL records (envs_tab): checks reveal the record's actual values (checks[].record_value),
the final action is scored with the record's actual outcome y_true. No LLM calls (DIVA's elicited specification is fixed).
Policies: DIVA (frozen knowledge config), always-stop (decide on the base belief of the same spec... and on the stated
base rate), reference-DP (gold model). Reports mean realized utility (normalized by l) minus costs with clustered
bootstrap CIs of DIVA minus always-stop. Writes results/realized_tab.json."""
import sys, os, json, glob, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read
from eval_tab import diva_scores, diva_u
from dcore import voi_exact, posterior
from extra_controls import boot_ci


def run_diva(sc, f, unc=None):
    g, l = float(sc["g"]), float(sc["l"])
    b = f["b"]; rem = {c["label"]: c for c in sc["checks"]}; cost = 0.0
    while rem:
        if unc is None:
            net = {k: voi_exact(b, f["par"][k][0], f["par"][k][1], g, l) - f["par"][k][2] for k in rem}
        else:
            import math
            GH = [(-math.sqrt(3), 1 / 6), (0.0, 2 / 3), (math.sqrt(3), 1 / 6)]
            lg = lambda x: math.log(min(max(x, .01), .99) / (1 - min(max(x, .01), .99))); sg = lambda x: 1 / (1 + math.exp(-x))
            net = {k: sum(w * voi_exact(b, min(max(sg(lg(f["par"][k][0]) + unc * z / 2), .01), .99), min(max(sg(lg(f["par"][k][1]) - unc * z / 2), .01), .99), g, l) for z, w in GH) - f["par"][k][2] for k in rem}
        k = max(net, key=net.get)
        if net[k] <= 1e-9:
            break
        c = rem.pop(k); v = c.get("record_value")
        if v is None:
            break
        cost += float(c["cost"]); s1, s0, _ = f["par"][k]; b = posterior(b, [(s1, s0, int(v))])
    act = (1 - b) * g - b * l > 0
    y = int(sc["y_true"])
    u = (g if y == 0 else -l) if act else 0.0
    return (u - cost) / l, cost


def main():
    out = {}
    for f in glob.glob(f"{IS}/outputs/test_tab/fk_*.jsonl.gz"):
        tag, ds = os.path.basename(f)[3:-9].rsplit("_", 1)
        env = {s["id"]: s for s in read(f"{IS}/envs_tab/data/{ds}_test.jsonl.gz")}
        fr = {r["id"]: r for r in read(f)}
        ids = [i for i in env if i in fr and env[i].get("y_true") is not None]
        if len(ids) < 200:
            continue
        D, S, cl, costs, DU = [], [], [], [], []
        for i in ids:
            sc = env[i]; fs = diva_scores(sc, fr[i], "cond", "greedy", "elicit", 0.75)
            u, c = run_diva(sc, fs); D.append(u); costs.append(c)
            uu, _ = run_diva(sc, fs, unc=1.0); DU.append(uu)
            g, l = float(sc["g"]), float(sc["l"]); b = fs["b"]
            act = (1 - b) * g - b * l > 0; y = int(sc["y_true"])
            S.append(((g if y == 0 else -l) if act else 0.0) / l)
            cl.append(sc.get("group_id") or i)
        diff = [a - b for a, b in zip(D, S)]
        out[f"{tag}:{ds}"] = dict(n=len(ids), diva=float(np.mean(D)), stop=float(np.mean(S)), mean_cost=float(np.mean(costs)),
                                  ci=boot_ci(diff, cl), diva_u=float(np.mean(DU)), ci_u=boot_ci([a - b for a, b in zip(DU, S)], cl))
    json.dump(out, open(f"{IS}/results/realized_tab.json", "w"), indent=1)
    for k, v in sorted(out.items()):
        print(k, v["n"], round(v["diva"], 4), round(v["stop"], 4), [round(x, 4) for x in v["ci"]])


if __name__ == "__main__":
    main()
