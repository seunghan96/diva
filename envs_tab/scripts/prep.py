"""Raw UCI files -> clean binary findings table + metadata (split, correlations, placebo selection).
Run: python -I scripts/prep.py <dataset>   (raw data is read as data only; script dir added explicitly)"""
import sys
sys.path.append("<path>")
sys.path.insert(0, "./envs_tab/scripts")
import json
import numpy as np
import pandas as pd
from features import DATASETS, G_COLS, H_COLS

B = "./envs_tab"
SPLIT_SEED = {"taiwan": 350, "german": 144, "cdc": 891, "heart": 45}
PLACEBO_MAX_ABS_CORR = 0.02


def load(ds):
    if ds == "taiwan":
        d = pd.read_csv(f"{B}/raw/taiwan/data.csv")
        return d, d.Y.values.astype(int)
    if ds == "german":
        d = pd.read_csv(f"{B}/raw/german/german.data", sep=r"\s+", header=None, names=G_COLS)
        return d, (d.target == 2).astype(int).values
    if ds == "cdc":
        d = pd.read_csv(f"{B}/raw/cdc/data.csv")
        return d, d.Diabetes_binary.values.astype(int)
    if ds == "heart":
        parts = []
        for site in ["cleveland", "hungarian", "switzerland", "va"]:
            x = pd.read_csv(f"{B}/raw/heart/processed.{site}.data", header=None, names=H_COLS, na_values="?")
            x["site"] = site
            parts.append(x)
        d = pd.concat(parts, ignore_index=True)
        d = d.apply(lambda c: pd.to_numeric(c, errors="coerce") if c.name != "site" else c)
        # extra record-keeping attributes from the 76-attribute files (joined on site + 8 clinical values; 100% match)
        cols76 = ("id ccf age sex painloc painexer relrest pncaden cp trestbps htn chol smoke cigs years fbs dm famhist "
                  "restecg ekgmo ekgday ekgyr dig prop nitr pro diuretic proto thaldur thaltime met thalach thalrest "
                  "tpeakbps tpeakbpd dummy trestbpd exang xhypo oldpeak slope rldv5 rldv5e ca restckm exerckm restef "
                  "restwm exeref exerwm thal thalsev thalpul earlobe cmo cday cyr num").split()
        rows = []
        for site, fn in [("hungarian", "hungarian.data"), ("switzerland", "switzerland.data"), ("va", "long-beach-va.data")]:
            cur = []
            for t in open(f"{B}/raw/heart/{fn}", encoding="latin-1").read().split():
                cur.append(t)
                if t == "name":
                    rows.append([float(x) for x in cur[:58]] + [site]); cur = []
        r = pd.DataFrame(rows, columns=cols76 + ["site"]).replace(-9.0, np.nan)
        key = ["age", "sex", "cp", "trestbps", "chol", "thalach", "exang", "oldpeak"]
        kf = lambda df: df.site + "|" + df[key].apply(lambda c: c.astype(float).round(2).astype(str)).agg("|".join, axis=1)
        r["k"] = kf(r); d["k"] = kf(d)
        r = r.drop_duplicates("k")[["k", "ekgmo", "ekgday", "cday"]]
        d = d.merge(r, on="k", how="left")
        print("heart extra-attribute join: matched", float(d.ekgday.notna().mean()))
        return d, (d.num > 0).astype(int).values
    raise ValueError(ds)


def main(ds):
    spec = DATASETS[ds]
    raw, y = load(ds)
    n = len(raw)
    rng = np.random.default_rng(SPLIT_SEED[ds])
    # stratified 60/20/20 split
    split = np.empty(n, dtype=object)
    for cls in [0, 1]:
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        a, b = int(round(0.6 * len(idx))), int(round(0.8 * len(idx)))
        split[idx[:a]], split[idx[a:b]], split[idx[b:]] = "train", "cal", "test"
    out = pd.DataFrame({"rid": np.arange(n), "split": split, "y": y})
    meta = {"dataset": ds, "n": n, "split_seed": SPLIT_SEED[ds], "findings": {}, "placebo_pool": {}}
    tr = split == "train"
    meta["base_rate_train"] = float(y[tr].mean())
    meta["base_rate_all"] = float(y.mean())
    meta["n_split"] = {s: int((split == s).sum()) for s in ["train", "cal", "test"]}

    def stats(v):
        m = tr & ~np.isnan(v)
        vv, yy = v[m], y[m]
        corr = float(np.corrcoef(vv, yy)[0, 1]) if vv.std() > 0 else 0.0
        return {"prevalence_train": float(vv.mean()), "corr_train": corr,
                "emp_s1": float(vv[yy == 1].mean()), "emp_s0": float(vv[yy == 0].mean()),
                "missing_rate": float(np.isnan(v).mean())}

    for k, (name, txt, thr, fn) in spec["findings"].items():
        v = np.asarray(fn(raw), dtype=float)
        out[k] = v
        meta["findings"][k] = {"name": name, "finding_text": txt, "threshold": thr, "is_placebo": False, **stats(v)}
    pool = []
    for k, (name, txt, thr, fn) in spec["placebo_pool"].items():
        v = np.asarray(fn(raw), dtype=float)
        st = stats(v)
        meta["placebo_pool"][k] = {"name": name, "finding_text": txt, "threshold": thr, **st}
        if abs(st["corr_train"]) < PLACEBO_MAX_ABS_CORR and k not in spec["findings"]:
            pool.append((abs(st["corr_train"]), k, v))
    pool.sort(key=lambda t: t[0])
    meta["placebo_rule"] = f"|corr_train| < {PLACEBO_MAX_ABS_CORR}"
    if not pool and n < 2000:
        # small-n fallback (documented): no pool attribute passes the strict rule; take the pool attribute with the smallest
        # |corr_train| provided it is far from significant (two-sided p > 0.3 on train) and |corr_all| < 0.02
        from scipy import stats as sst
        best = None
        for k, (name, txt, thr, fn) in spec["placebo_pool"].items():
            v = np.asarray(fn(raw), dtype=float)
            m = tr & ~np.isnan(v); ma = ~np.isnan(v)
            r, pv = sst.pearsonr(v[m], y[m])
            ra = float(np.corrcoef(v[ma], y[ma])[0, 1])
            meta["placebo_pool"][k].update({"p_train": float(pv), "corr_all": ra})
            if pv > 0.3 and abs(ra) < PLACEBO_MAX_ABS_CORR and (best is None or abs(r) < best[0]):
                best = (abs(r), k, v)
        if best:
            pool = [best]
            meta["placebo_rule"] = "small-n fallback: min |corr_train| with p_train > 0.3 and |corr_all| < 0.02"
    for _, k, v in pool[:2]:
        name, txt, thr, _ = spec["placebo_pool"][k]
        out[k] = v
        meta["findings"][k] = {"name": name, "finding_text": txt, "threshold": thr, "is_placebo": True,
                               **meta["placebo_pool"][k]}
        meta["findings"][k].pop("name_", None)
    meta["placebos"] = [k for _, k, _ in pool[:2]]
    if ds == "heart":
        out["site"] = raw.site.values
    out.to_csv(f"{B}/clean/{ds}.csv.gz", index=False, compression="gzip")
    json.dump(meta, open(f"{B}/clean/{ds}_meta.json", "w"), indent=1)
    print(ds, "n", n, "base_rate_train %.4f" % meta["base_rate_train"], meta["n_split"])
    for k, f in meta["findings"].items():
        print(f"  {k:16s} plac={int(f['is_placebo'])} prev={f['prevalence_train']:.3f} corr={f['corr_train']:+.3f} "
              f"s1={f['emp_s1']:.3f} s0={f['emp_s0']:.3f} miss={f['missing_rate']:.2f}")
    print("  placebo pool:", {k: round(v["corr_train"], 4) for k, v in meta["placebo_pool"].items()})


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        main(ds)
