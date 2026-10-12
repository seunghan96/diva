"""Transfer of DIVA's configuration without target-domain labels (leave-one-domain-out).
Domains: taiwan, german, cdc, heart, SEC (all four targets pooled). For each held-out domain, (lam, sigma) is selected on the
DEV splits of the OTHER domains only (mean DP-regret ratio to never-ask), then applied to the held-out TEST split.
Compared with: the global configuration (lam=0.75, sigma=1), and Fac-EIG cost-tuned on the held-out domain's own dev labels.
Writes results/transfer.json."""
import sys, os, glob, json, itertools, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read, score, summarize, choose_max
from eval_tab import diva_u
from eval_real import fio_real_u
K = "dp_regret"
GRID = [(lam, sg) for lam in (1.0, 0.75, 0.5) for sg in (0.0, 0.5, 1.0, 1.5)]
DOMS = ["taiwan", "german", "cdc", "heart", "sec"]


def cells(split):
    out = collections.defaultdict(list)
    if split == "dev":
        for f in glob.glob(f"{IS}/outputs/tab_dev/fk_*.jsonl.gz"):
            tag, ds = os.path.basename(f)[3:-9].rsplit("_", 1)
            out[ds].append(("tab", f"{IS}/envs_tab/data/{ds}_dev.jsonl.gz", f))
        for f in glob.glob(f"{IS}/outputs/real_dev/fr_*.jsonl.gz"):
            b = os.path.basename(f)[3:-9]
            tg = next((t for t in ["ret_5", "vol_5", "vol_20", "ret1"] if b.endswith(t)), None)
            if tg:
                t2 = "ret_1" if tg == "ret1" else tg
                out["sec"].append(("sec", f"{IS}/envs_real/data/{t2}_dev.jsonl.gz", f))
    else:
        for f in glob.glob(f"{IS}/outputs/test_tab/fk_*.jsonl.gz"):
            tag, ds = os.path.basename(f)[3:-9].rsplit("_", 1)
            out[ds].append(("tab", f"{IS}/envs_tab/data/{ds}_test.jsonl.gz", f))
        for f in glob.glob(f"{IS}/outputs/test_real/fr_*.jsonl.gz"):
            b = os.path.basename(f)[3:-9]
            tg = next((t for t in ["ret_1", "ret_5", "vol_5", "vol_20"] if b.endswith(t)), None)
            if tg:
                out["sec"].append(("sec", f"{IS}/envs_real/data/{tg}_test.jsonl.gz", f))
    return out


def ratios(cell_list, cfgs, maxn=400):
    res = {c: [] for c in cfgs}
    for kind, ep, fp in cell_list:
        env = {s["id"]: s for s in read(ep)}; fr = {r["id"]: r for r in read(fp)}
        ids = [i for i in env if i in fr][:maxn]
        if len(ids) < 100:
            continue
        base = summarize([score(env[i], "S") for i in ids])[K]
        if base <= 0:
            continue
        for lam, sg in cfgs:
            rows = []
            for i in ids:
                f_ = diva_u(env[i], fr[i], sg, "cond", lam) if kind == "tab" else fio_real_u(env[i], fr[i], sg, lam)
                rows.append(score(env[i], choose_max(f_["net"])))
            res[(lam, sg)].append(summarize(rows)[K] / base)
    return {c: (float(np.mean(v)) if v else np.nan) for c, v in res.items()}


def main():
    dev, test = cells("dev"), cells("test")
    devr = {d: ratios(dev[d], GRID) for d in DOMS}
    out = {}
    for d in DOMS:
        others = [o for o in DOMS if o != d]
        score_cfg = {c: np.nanmean([devr[o][c] for o in others]) for c in GRID}
        sel = min(score_cfg, key=score_cfg.get)
        t = ratios(test[d], [sel, (0.75, 1.0), (0.75, 0.0)])
        out[d] = {"selected": sel, "test_selected": t[sel], "test_global_u": t[(0.75, 1.0)], "test_point": t[(0.75, 0.0)]}
        print(d, out[d], flush=True)
    json.dump({k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()} for k, v in out.items()},
              open(f"{IS}/results/transfer.json", "w"), indent=1)


if __name__ == "__main__":
    main()
