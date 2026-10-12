"""DEV-ONLY diagnosis of the real-data weakness under the myopic gold (coordinator item 2, after R6).
For every LLM with a dev specification, on records (tab_dev) and SEC (real_dev):
  - decomposition of DIVA's / DIVA-U's MYOPIC regret into: acquiring when the myopic gold stops (over), stopping when the
    myopic gold acquires (under), acquiring a non-best check when acquiring is right (wrong check);
  - candidate principled variants (no new LLM calls): stated base rate as the prior (the task states it; the elicited
    prior can contradict it), and both priors for DIVA-U.
Writes results/dev_myopic_diag.json. TEST data are not read."""
import sys, os, glob, json, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read, score, summarize, choose_max
from eval_tab import diva_scores, diva_u
from eval_real import fio_real, fio_real_u


def tab_variants(sc, fr):
    out = {}
    for pm in ("elicit", "stated"):
        f = diva_scores(sc, fr, "cond", "greedy", pm, 0.75)
        out[f"diva_{pm}"] = choose_max(f["net"])
    out["diva_u_elicit"] = choose_max(diva_u(sc, fr, 1.0)["net"])
    fs = dict(fr); fs["prior"] = [float(sc["prior"])]
    out["diva_u_stated"] = choose_max(diva_u(sc, fs, 1.0)["net"])
    return out


def sec_variants(sc, fr):
    from eval_real import stated_base_rate
    out = {}
    for pm in ("elicit", "stated"):
        out[f"diva_{pm}"] = choose_max(fio_real(sc, fr, prior_mode=pm, lam=0.75)["net"])
    out["diva_u_elicit"] = choose_max(fio_real_u(sc, fr, 1.0)["net"])
    fs = dict(fr); fs["prior"] = [stated_base_rate(sc)]
    out["diva_u_stated"] = choose_max(fio_real_u(sc, fs, 1.0)["net"])
    return out


def decomp(rows, chs, scs):
    """myopic regret decomposition"""
    over = under = wrong = 0.0
    for r, c, sc in zip(rows, chs, scs):
        g = sc["exact"]["gold_choice"]
        if r["norm_regret"] <= 1e-12:
            continue
        if c != "S" and g == "S":
            over += r["norm_regret"]
        elif c == "S" and g != "S":
            under += r["norm_regret"]
        else:
            wrong += r["norm_regret"]
    tot = over + under + wrong
    return {"over": over / tot if tot else 0, "under": under / tot if tot else 0, "wrong": wrong / tot if tot else 0}


def main():
    out = collections.defaultdict(dict)
    cells = []
    for ds in ["taiwan", "german", "cdc", "heart"]:
        for f in glob.glob(f"{IS}/outputs/tab_dev/fk_*_{ds}.jsonl.gz"):
            cells.append(("tab", ds, f"{IS}/envs_tab/data/{ds}_dev.jsonl.gz", f))
    for tg in ["ret_1", "ret_5", "vol_5", "vol_20"]:
        dn = "ret1" if tg == "ret_1" else tg
        for f in glob.glob(f"{IS}/outputs/real_dev/fr_*_{dn}.jsonl.gz"):
            tag = os.path.basename(f)[3:-9][: -(len(dn) + 1)]
            if tag.split("_")[-1] in ("joint", "ratio", "contrast"):
                continue
            cells.append(("sec", tg, f"{IS}/envs_real/data/{tg}_dev.jsonl.gz", f))
    for kind, e, ep, fp in cells:
        tag = os.path.basename(fp)[3:-9].rsplit("_" + (e if kind == "tab" else ("ret1" if e == "ret_1" else e)), 1)[0]
        env = {s["id"]: s for s in read(ep)}; fr = {r["id"]: r for r in read(fp)}
        ids = [i for i in env if i in fr]
        if len(ids) < 100:
            continue
        V = collections.defaultdict(list); C = collections.defaultdict(list)
        for i in ids:
            ch = (tab_variants if kind == "tab" else sec_variants)(env[i], fr[i]); ch["never"] = "S"
            for m, c in ch.items():
                V[m].append(score(env[i], c)); C[m].append(c)
        scs = [env[i] for i in ids]
        rec = {}
        for m in V:
            s = summarize(V[m]); rec[m] = {"dp": s["dp_regret"], "myo": s["norm_regret"], "stop": s["stopped"], "over_acq": s["over_acq"], "under_acq": s["under_acq"]}
            if m != "never":
                rec[m]["decomp"] = decomp(V[m], C[m], scs)
        out[e][tag] = rec
        print(e, tag, {m: (round(100 * v["dp"], 2), round(100 * v["myo"], 2), round(v["stop"], 2)) for m, v in rec.items()}, flush=True)
    json.dump(out, open(f"{IS}/results/dev_myopic_diag.json", "w"), indent=1)


if __name__ == "__main__":
    main()
