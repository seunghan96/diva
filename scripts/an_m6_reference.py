"""Reviewer M6: validity of the reference models and robustness of the comparison to the reference (no LLM calls).
  metrics   out-of-sample AUC, Brier score (and the base-rate Brier score) and ECE of the eight reference models on the test
            records with every finding observed (from envs_tab/models/*/metrics.json and envs_real/voi/summary_*.json).
  records   ALTERNATIVE reference of a different model class: masked logistic regression with all pairwise interactions of
            the findings (outcome model and one model per finding), Platt-calibrated on the calibration records. The joint
            of each scenario is built exactly as for the main reference (chain rule over six orders, envs_tab/scripts/gen.py),
            so it makes no conditional-independence assumption.
  sec       NON-FACTORED reference for SEC filings: the same LightGBM reference models, combined into one coherent joint over
            the offered items (chain rule of the value models, outcome model at every leaf) instead of the conditionally
            independent preposteriors that the stored DP gold uses.
  rescore   every method's first decision (results/know_choices, first 300 test scenarios, all 20 LLMs) scored with the DP
            regret under the main and the alternative reference.
usage: an_m6_reference.py [metrics|records|sec|rescore|all]   (pathcal venv: lightgbm, sklearn)"""
import sys, os, json, glob, collections, pickle, importlib.util
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
TABD = f"{IS}/envs_tab"; SECD = f"{IS}/envs_real"
TAB = ["taiwan", "german", "cdc", "heart"]; SEC = ["ret_1", "ret_5", "vol_5", "vol_20"]
N300 = 300
CACHE = f"{IS}/results/m6_cache"


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def read(p):
    import gzip
    with gzip.open(p, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


# ------------------------------------------------------------------------------------------------------------------ metrics
def metrics():
    out = {}
    for ds in TAB:
        m = json.load(open(f"{TABD}/models/{ds}/metrics.json"))["y_full"]
        out[ds] = {"auc": m["auc"], "brier": m["brier"], "brier0": m["brier_baserate"], "ece": m["ece"]}
    for t in SEC:
        d = json.load(open(f"{SECD}/voi/summary_{t}.json"))[t]["diag"]["test"]
        b = d["base_rate"]
        out[t] = {"auc": d["auc_full"], "brier": d["brier_full"], "brier0": b * (1 - b), "ece": d["ece_full"]}
    return out


# ------------------------------------------------------------------------------------------------------------------ records
def _enc(X, drop=None):
    """masked binary findings -> sparse indicators (x=1, x=0 per finding; NaN = neither) and all pairwise products"""
    import scipy.sparse as sp
    X = np.array(X, dtype=float)
    if drop is not None:
        X[:, drop] = np.nan
    A = np.concatenate([(X == 1), (X == 0)], 1).astype(np.float32)
    F2 = A.shape[1]
    iu = np.triu_indices(F2, 1)
    P = A[:, iu[0]] * A[:, iu[1]]
    return sp.csr_matrix(np.concatenate([A, P], 1))


def _fit_lr(args):
    from sklearn.linear_model import LogisticRegression
    Xr, yr, Xc, yc, drop, C, platt = args
    lr = LogisticRegression(C=C, max_iter=3000, solver="lbfgs").fit(_enc(Xr, drop), yr)
    ab = (1.0, 0.0)
    if platt:
        z = lr.decision_function(_enc(Xc, drop))
        cal = LogisticRegression(C=1e6, max_iter=1000).fit(z[:, None], yc)
        ab = (float(cal.coef_[0, 0]), float(cal.intercept_[0]))
    return lr, ab


class LRRef:
    """py / pf interface of envs_tab/scripts/gen.py Ref, backed by the logistic-regression reference"""
    def __init__(self, y, ab, f):
        self.ylr, self.ab, self.f = y, ab, f

    def py(self, X):
        z = self.ylr.decision_function(_enc(X))
        return np.clip(1 / (1 + np.exp(-(self.ab[0] * z + self.ab[1]))), 1e-3, 1 - 1e-3)

    def pf(self, j, X):
        return np.clip(self.f[j].predict_proba(_enc(X, j))[:, 1], 1e-3, 1 - 1e-3)


def fit_records(ds):
    import pandas as pd
    from multiprocessing import Pool
    fr = _mod("fit_ref", f"{TABD}/scripts/fit_ref.py")
    meta = json.load(open(f"{TABD}/clean/{ds}_meta.json")); feats = list(meta["findings"])
    df = pd.read_csv(f"{TABD}/clean/{ds}.csv.gz")
    X = {s: df.loc[df.split == s, feats].values.astype(float) for s in ["train", "cal", "test"]}
    Y = {s: df.loc[df.split == s, "y"].values.astype(int) for s in ["train", "cal", "test"]}
    rng = np.random.default_rng(11)
    if ds == "cdc":  # 152k training records: a 40k subsample keeps the fit tractable
        k = rng.choice(len(Y["train"]), 40000, replace=False); X["train"], Y["train"] = X["train"][k], Y["train"][k]
    K = {"taiwan": 6, "cdc": 4, "german": 40, "heart": 40}[ds]
    Kc = {"taiwan": 3, "cdc": 2, "german": 20, "heart": 20}[ds]
    C = 0.3 if ds in ("german", "heart") else 1.0
    Xr, _ = fr.mask_rows(X["train"], K, rng); yr = np.repeat(Y["train"], K)
    Xc, _ = fr.mask_rows(X["cal"], Kc, rng); yc = np.repeat(Y["cal"], Kc)
    jobs = [(Xr, yr, Xc, yc, None, C, True)]
    for j in range(len(feats)):  # f-model j: P(f_j = 1 | other findings), rows where f_j is observed in the record
        ok = ~np.isnan(np.repeat(X["train"][:, j], K))
        yj = np.repeat(X["train"][:, j], K)[ok].astype(int)
        jobs.append((Xr[ok], yj, None, None, j, C, False))
    with Pool(min(len(jobs), 16)) as p:
        fits = p.map(_fit_lr, jobs)
    ref = LRRef(fits[0][0], fits[0][1], {j: fits[j + 1][0] for j in range(len(feats))})
    # out-of-sample metrics of the alternative reference (all findings observed), as for the main reference
    pt = ref.py(X["test"]); yt = Y["test"]
    from sklearn.metrics import roc_auc_score, brier_score_loss
    met = {"auc": float(roc_auc_score(yt, pt)), "brier": float(brier_score_loss(yt, pt)), "ece": fr.ece(pt, yt)}
    os.makedirs(CACHE, exist_ok=True)
    pickle.dump(ref, open(f"{CACHE}/lrref_{ds}.pkl", "wb"))
    return ref, met, feats, df


def alt_records(ds):
    gen = _mod("gen", f"{TABD}/scripts/gen.py")
    ref, met, feats, df = fit_records(ds)
    envl = read(f"{TABD}/data/{ds}_test.jsonl.gz")[:N300]
    fi = {f: j for j, f in enumerate(feats)}
    Xall = df.set_index("rid")[feats]
    recs, drws = [], []
    rng = np.random.default_rng(5)
    for sc in envl:
        x = Xall.loc[sc["record_id"]].values.astype(float)
        drws.append({"obs": [fi[o["key"]] for o in sc["observed"]], "cands": [fi[c["key"]] for c in sorted(sc["checks"], key=lambda c: c["label"])],
                     "orders": [list(map(int, rng.permutation(gen.M))) for _ in range(gen.R_ORDERS)]})
        for o in sc["observed"]:
            x[fi[o["key"]]] = o["value"]
        recs.append(x)
    Js, _ = gen.build_joints(ref, len(feats), np.array(recs), drws)
    alt = {}
    for sc, J in zip(envl, Js):
        cs = [sc["costs"][L] for L in gen.LABELS]
        ex = gen.solve(J, cs, float(sc["g"]), float(sc["l"]))
        alt[sc["id"]] = {"q": ex["q"], "vs": ex["value_of_stopping"], "v": ex["dp_value"], "l": float(sc["l"])}
    # check: the main reference rebuilt with the same code reproduces the stored gold (sanity of the scenario mapping)
    main_ref = gen.Ref(ds, feats)
    Jm, _ = gen.build_joints(main_ref, len(feats), np.array(recs[:30]), drws[:30])
    dev = [abs(gen.solve(J, [sc["costs"][L] for L in gen.LABELS], float(sc["g"]), float(sc["l"]))["dp_value"] - sc["exact"]["dp_value"]) / float(sc["l"])
           for sc, J in zip(envl[:30], Jm)]
    return {"metrics": met, "gold": alt, "sanity_max_abs_dv": float(max(dev))}


# ------------------------------------------------------------------------------------------------------------------ SEC
def sec_joint_solver(J, nv, costs, g, l, tol=1e-9):
    """exact DP on a joint J[x_1..x_m, y] of multi-valued items (any dependence); returns q, value of stopping, value"""
    import itertools
    m = len(nv)
    U = lambda b: max(g * (1 - b) - l * b, 0.0)
    memo = {}

    def marg(state):
        idx = tuple(slice(None) if s < 0 else s for s in state)
        sub = J[idx]
        ax = tuple(range(sub.ndim - 1))
        return sub.sum(axis=ax) if ax else sub

    def value(state):
        if state in memo:
            return memo[state]
        my = marg(state); tot = my.sum(); b = my[1] / tot
        best = U(b)
        for i in range(m):
            if state[i] >= 0:
                continue
            best = max(best, qval(state, i, tot))
        memo[state] = best
        return best

    def qval(state, i, tot):
        v = -costs[i]
        for x in range(nv[i]):
            s2 = list(state); s2[i] = x; s2 = tuple(s2)
            p = marg(s2).sum() / tot
            if p > 1e-12:
                v += p * value(s2)
        return v
    s0 = tuple([-1] * m)
    my = marg(s0); tot = my.sum()
    q = {i: qval(s0, i, tot) for i in range(m)}
    return q, U(my[1] / tot), value(s0)


def alt_sec(arg):
    t, lo, hi = arg
    import pandas as pd, lightgbm as lgb
    be = _mod("be", f"{SECD}/scripts/04_build_env.py")
    md = f"{SECD}/data/ref_models"
    yb = lgb.Booster(model_file=f"{md}/{t}.txt"); ab = json.load(open(f"{md}/{t}.platt.json"))
    be.FEATS = yb.feature_name(); be.ITEM_KEYS = [f[2:] for f in be.FEATS if f.startswith("b_")]
    it = pd.read_parquet(f"{SECD}/data/items.parquet")
    tr = it.split == "train"
    marg = {k: (np.bincount(it.loc[tr, f"b_{k}"].dropna().astype(int), minlength=be.nvals(k)) / it.loc[tr, f"b_{k}"].notna().sum())
            for k in be.ITEM_KEYS if be.ITEMS[k].get("placebo")}
    vb = {k: lgb.Booster(model_file=f"{md}/v_{k}.txt") for k in be.ITEM_KEYS if not be.ITEMS[k].get("placebo")}
    ref = be.Ref(yb, (ab["a"], ab["b"]), vb, marg)
    envl = read(f"{SECD}/data/{t}_test.jsonl.gz")[:N300][lo:hi]
    byrid = it.set_index("rid")
    rng = np.random.default_rng(9 + lo)
    alt, sanity = {}, []
    for sc in envl:
        base = byrid.loc[[sc["_meta_do_not_show"]["rid"]], be.FEATS].copy()
        for k in be.ITEM_KEYS:
            base[f"b_{k}"] = np.nan
        for o in sc["observed"]:
            base[f"b_{o['item']}"] = float(o["bin"])
        chk = sorted(sc["checks"], key=lambda c: c["label"])
        items = [c["item"] for c in chk]; nv = [be.nvals(k) for k in items]; m = len(items)
        grid = np.array(list(np.ndindex(*nv)))  # all leaves
        # p(x_C | E): chain rule over six random orders of the value models (placebos: training marginal)
        px = np.zeros(len(grid))
        for _ in range(6):
            perm = list(rng.permutation(m)); pr = np.ones(len(grid))
            for j, i in enumerate(perm):
                prev = perm[:j]
                keys, inv = np.unique(grid[:, prev], axis=0, return_inverse=True) if prev else (np.zeros((1, 0), int), np.zeros(len(grid), int))
                inv = np.asarray(inv).reshape(-1)
                Xq = pd.concat([base] * len(keys), ignore_index=True)
                for a, pi in enumerate(prev):
                    Xq[f"b_{items[pi]}"] = keys[:, a].astype(float)
                P = ref.pv(items[i], Xq)  # len(keys) x nv_i
                pr *= P[inv, grid[:, i]]
            px += pr / 6
        Xl = pd.concat([base] * len(grid), ignore_index=True)
        for i, k in enumerate(items):
            Xl[f"b_{k}"] = grid[:, i].astype(float)
        py = ref.py(Xl)
        J = np.zeros(tuple(nv) + (2,))
        for r, cfg in enumerate(grid):
            J[tuple(cfg)] = (px[r] * (1 - py[r]), px[r] * py[r])
        q, vs, v = sec_joint_solver(J, nv, [float(c["cost"]) for c in chk], float(sc["g"]), float(sc["l"]))
        alt[sc["id"]] = {"q": {chk[i]["label"]: q[i] for i in range(m)}, "vs": vs, "v": v, "l": float(sc["l"])}
        sanity.append(abs(J[..., 1].sum() / J.sum() - sc["exact"]["posterior"]))
    return {"gold": alt, "gaps": sanity}


# ------------------------------------------------------------------------------------------------------------------ rescore
METHS = ["direct", "cot_need", "sc", "react", "ask_p", "voi_prompt", "voi_ask", "bed_llm", "pal", "never", "always_top", "diva", "diva_u", "diva_dp"]


def rescore():
    sys.path.insert(0, HERE)
    from complete import KNOW_LLMS
    gold = {}
    for e in TAB + SEC:
        p = f"{CACHE}/alt_{e}.json"
        if os.path.exists(p):
            gold[e] = json.load(open(p))["gold"]
    envs = {e: {s["id"]: s for s in (read(f"{TABD}/data/{e}_test.jsonl.gz") if e in TAB else read(f"{SECD}/data/{e}_test.jsonl.gz"))[:N300]} for e in gold}
    res = {}
    for e in gold:
        per = collections.defaultdict(lambda: collections.defaultdict(list))
        ok = True
        for tag in KNOW_LLMS:
            p = f"{IS}/results/know_choices/{e}_{tag}.json"
            if not os.path.exists(p):
                ok = False; continue
            C = json.load(open(p))
            for m in METHS:
                ref_r, alt_r = [], []
                for i, ch in C.items():
                    if m not in ch or i not in gold[e]:
                        continue
                    sc = envs[e][i]; ex = sc["exact"]; L = float(sc["l"])
                    qr = float(ex["value_of_stopping"]) if ch[m] == "S" else float(ex["q"][ch[m]])
                    ref_r.append((float(ex["dp_value"]) - qr) / L)
                    a = gold[e][i]; qa = a["vs"] if ch[m] == "S" else a["q"][ch[m]]
                    alt_r.append((a["v"] - qa) / L)
                if ref_r:
                    per[m]["ref"].append(float(np.mean(ref_r))); per[m]["alt"].append(float(np.mean(alt_r)))
        res[e] = {m: {"ref": float(np.mean(v["ref"])), "alt": float(np.mean(v["alt"])), "n_llm": len(v["ref"])} for m, v in per.items()} if ok else None
    json.dump(res, open(f"{IS}/results/m6_rescore.json", "w"), indent=1)
    return res


def main(what):
    os.makedirs(CACHE, exist_ok=True)
    if what in ("metrics", "all"):
        json.dump(metrics(), open(f"{IS}/results/m6_metrics.json", "w"), indent=1)
    if what in ("records", "all"):
        for ds in TAB:
            r = alt_records(ds)
            json.dump(r, open(f"{CACHE}/alt_{ds}.json", "w"))
            print(ds, r["metrics"], "sanity", r["sanity_max_abs_dv"], flush=True)
    if what in ("sec", "all"):
        from multiprocessing import Pool
        chunks = [(t, lo, lo + 10) for t in SEC for lo in range(0, N300, 10)]
        with Pool(int(os.environ.get("NPROC", "60"))) as p:
            rs = p.map(alt_sec, chunks)
        for t in SEC:
            g, gaps = {}, []
            for (tt, lo, hi), r in zip(chunks, rs):
                if tt == t:
                    g.update(r["gold"]); gaps += r["gaps"]
            json.dump({"gold": g, "root_belief_gap_mean": float(np.mean(gaps)), "root_belief_gap_max": float(np.max(gaps))}, open(f"{CACHE}/alt_{t}.json", "w"))
            print(t, len(g), "root belief gap", float(np.mean(gaps)), float(np.max(gaps)), flush=True)
    if what in ("rescore", "all"):
        res = rescore()
        for e, r in res.items():
            if r:
                print(e, {m: (round(100 * v["ref"], 2), round(100 * v["alt"], 2)) for m, v in r.items()})


if __name__ == "__main__" and (len(sys.argv) < 2 or sys.argv[1] != "tables"):
    main(sys.argv[1] if len(sys.argv) > 1 else "all")


# ------------------------------------------------------------------------------------------------------------------ tables
def tables():
    from complete import mname, MCITE, group_rows, ENVN, fmt_best, PH
    from scipy.stats import spearmanr
    M = json.load(open(f"{IS}/results/m6_metrics.json"))
    alt_m = {ds: json.load(open(f"{CACHE}/alt_{ds}.json"))["metrics"] for ds in TAB if os.path.exists(f"{CACHE}/alt_{ds}.json")}
    lines = [f"{ENVN[e]} & {M[e]['auc']:.3f} & {M[e]['brier']:.3f} & {M[e]['brier0']:.3f} & {M[e]['ece']:.3f} \\\\" for e in TAB + SEC]
    lines.insert(len(TAB), "\\midrule")
    t1 = ("\\begin{table}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{l|cccc}\n\\toprule\n"
          "Environment & AUC & Brier & \\begin{tabular}[c]{@{}c@{}}Brier\\\\(base rate)\\end{tabular} & ECE \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
          "\\caption{\\textbf{Validity of the reference models} on the held-out test records with every finding observed. Brier (base rate): Brier score of a constant prediction at the training (records) or test (SEC) base rate.}\n"
          "\\label{tab:ref_metrics}\n\\end{table}\n")
    open(f"{IS}/paper/tables/ref_metrics.tex", "w").write(t1)
    R = json.load(open(f"{IS}/results/m6_rescore.json"))
    ok = all(R.get(e) for e in TAB + SEC)
    meths = [m for m in METHS]
    groups = [("records", TAB), ("sec", SEC)]
    val = {}
    for m in meths:
        for g, es in groups:
            for k in ("ref", "alt"):
                val[(m, g, k)] = float(np.mean([R[e][m][k] for e in es])) if ok else None
    rho = {g: float(np.mean([spearmanr([R[e][m]["ref"] for m in meths], [R[e][m]["alt"] for m in meths]).correlation for e in es])) if ok else None for g, es in groups}
    cols = [(g, k) for g, _ in groups for k in ("ref", "alt")]
    fm = {c: fmt_best([val[(m,) + c] for m in meths], scale=100) for c in cols}
    names, rows = [], []
    for j, m in enumerate(meths):
        n = mname(m); cite = MCITE.get(n)
        cell = ("\\begin{tabular}[c]{@{}l@{}}" + n + "\\\\\\citep{" + cite + "}\\end{tabular}") if cite else n
        ours = n in ("\\diva{}", "\\diva{}-U", "\\diva{}-DP")
        names.append(n)
        rows.append(("\\rowcolor{oursrow} " if ours else "") + cell + " & " + " & ".join(fm[c][j] for c in cols) + " \\\\")
    body = group_rows(names, rows, 5)
    body += ["\\midrule", "Rank correlation & \\multicolumn{2}{c|}{" + (PH if rho["records"] is None else f"{rho['records']:.3f}") + "} & \\multicolumn{2}{c}{"
             + (PH if rho["sec"] is None else f"{rho['sec']:.3f}") + "} \\\\"]
    t2 = ("\\begin{table}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{l|cc|cc}\n\\toprule\n"
          "\\multirow{2}{*}[-1.0ex]{Method} & \\multicolumn{2}{c|}{Records} & \\multicolumn{2}{c}{SEC filings} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
          " & Main & Alt. & Main & Alt. \\\\\n\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n"
          "\\caption{\\textbf{Robustness to the reference model.} DP regret ($\\times$100) under the main and an alternative reference (Alt.), averaged over the 20 LLMs. Alt.: Logistic regression with pairwise interactions (records), non-factored joint (SEC filings).}\n"
          "\\label{tab:alt_ref}\n\\end{table}\n")
    open(f"{IS}/paper/tables/alt_ref.tex", "w").write(t2)
    print("rho", rho, "alt metrics", alt_m)
    return rho


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "tables":
    tables()
