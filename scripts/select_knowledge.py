"""Dev-only selection of DIVA's knowledge-required configuration (robust spec-ensemble stopping quantile q, shrinkage lam,
elicitation form for binary checks). Objective: mean over (env, model) cells of DP-regret / always-stop DP-regret
(lower is better), each cell needing >= 60 scenarios. Prints the ranking; writes results/select_knowledge.json."""
import sys, os, json, glob, itertools, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read, score, summarize
from eval_tab import diva_robust as tab_robust, diva_scores
from eval_real import fio_real_robust


def cells():
    out = []
    for f in glob.glob(f"{IS}/outputs/tab_dev/fk_*_*.jsonl.gz"):
        b = os.path.basename(f)[3:-9]; tag, ds = b.rsplit("_", 1)
        out.append(("tab", ds, tag, f"{IS}/envs_tab/data/{ds}_dev.jsonl.gz", f))
    for f in glob.glob(f"{IS}/outputs/real_dev/fr_*.jsonl.gz"):
        b = os.path.basename(f)[3:-9]
        for tg in ("ret_1", "ret_5", "vol_5", "vol_20", "ret1"):
            if b.endswith("_" + tg):
                tag = b[: -len(tg) - 1]; t2 = "ret_1" if tg == "ret1" else tg
                out.append(("real", t2, tag, f"{IS}/envs_real/data/{t2}_dev.jsonl.gz", f))
    return out


def main():
    grid_q = [None, 0.5, 0.25, 0.0]; grid_l = [1.0, 0.75, 0.5]; forms = ["lik", "cond", "both"]
    table = collections.defaultdict(list)
    for kind, env_name, tag, ep, fp in cells():
        env = {s["id"]: s for s in read(ep)}; fr = {r["id"]: r for r in read(fp)}
        ids = [i for i in env if i in fr]
        if len(ids) < 60:
            continue
        base = summarize([score(env[i], "S") for i in ids])["dp_regret"]
        if base <= 0:
            continue
        for q, lam in itertools.product(grid_q, grid_l):
            for form in (forms if kind == "tab" else ["cond"]):
                rows = []
                for i in ids:
                    if kind == "tab":
                        ch, _ = tab_robust(env[i], fr[i], form, (q if q is not None else 0.5), lam, nspec=(5 if q is not None else 1))
                        if q is None:  # plain DIVA (greedy spec, tau = 0)
                            f = diva_scores(env[i], fr[i], form, "greedy", "elicit", lam)
                            k = max(f["net"], key=f["net"].get); ch = k if f["net"][k] > 1e-9 else "S"
                    else:
                        ch, _ = fio_real_robust(env[i], fr[i], (q if q is not None else 0.5), (5 if q is not None else 1), lam)
                    rows.append(score(env[i], ch))
                s = summarize(rows)
                key = (q, lam, form if kind == "tab" else "cond*")
                table[key].append(dict(cell=f"{kind}:{env_name}:{tag}", ratio=s["dp_regret"] / base, myo=s["norm_regret"],
                                       dp=s["dp_regret"], base=base, n=len(ids)))
    # forms only apply to tab; merge real cells into every tab-form key with the same (q, lam)
    merged = {}
    for (q, lam, form), rows in table.items():
        if form == "cond*":
            continue
        real = table.get((q, lam, "cond*"), [])
        allr = rows + real
        merged[str((q, lam, form))] = dict(mean_ratio=float(np.mean([r["ratio"] for r in allr])),
                                           frac_better=float(np.mean([r["ratio"] < 1 for r in allr])),
                                           n_cells=len(allr), cells=allr)
    rank = sorted(merged.items(), key=lambda kv: kv[1]["mean_ratio"])
    for k, v in rank[:12]:
        print(k, "mean ratio %.3f  better-than-stop %.2f  cells %d" % (v["mean_ratio"], v["frac_better"], v["n_cells"]))
    json.dump(merged, open(f"{IS}/results/select_knowledge.json", "w"), indent=1)


if __name__ == "__main__":
    main()
