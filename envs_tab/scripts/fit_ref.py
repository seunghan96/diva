"""Fit the masked reference models on TRAIN records (CPU only).
  Y-model:  P(Y=1 | any observed subset of findings), LightGBM on binary features with random masking (NaN = unobserved),
            early stopping on masked CAL replicates, isotonic calibration on CAL (Platt for the two ~1k-record sets).
  f-models: for each finding f, P(f=1 | any observed subset of the other findings), same masking scheme (uncalibrated).
Saves models/{ds}/*.txt + calib, and models/{ds}/metrics.json (out-of-sample AUC / Brier / ECE on masked TEST replicates)."""
import sys
sys.path.append("<path>")
import json, os, pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

B = "./envs_tab"
CLIP = 1e-3
SMALL = {"german", "heart"}


def mask_rows(X, K, rng):
    """Replicate rows K times; in each replicate keep a uniformly random number k in {0..F} of features observed."""
    n, F = X.shape
    Xr = np.repeat(X, K, axis=0)
    ks = rng.integers(0, F + 1, size=len(Xr))
    keep = rng.random((len(Xr), F)).argsort(1) < ks[:, None]
    Xr = np.where(keep, Xr, np.nan)
    return Xr, keep


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def lgb_params(ds):
    if ds in SMALL:
        return dict(objective="binary", learning_rate=0.03, num_leaves=7, min_child_samples=300, lambda_l2=10.0,
                    feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=32, seed=0)
    return dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=100, lambda_l2=1.0,
                verbose=-1, num_threads=32, seed=0)


def fit_masked(X, y, Xc, yc, ds, rng, K, Kc):
    Xr, _ = mask_rows(X, K, rng)
    yr = np.repeat(y, K)
    Xcr, _ = mask_rows(Xc, Kc, rng)
    ycr = np.repeat(yc, Kc)
    dtr = lgb.Dataset(Xr, yr, free_raw_data=True)
    dva = lgb.Dataset(Xcr, ycr, reference=dtr)
    bst = lgb.train(lgb_params(ds), dtr, num_boost_round=3000, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(100, verbose=False)])
    return bst, Xcr, ycr


def main(ds):
    meta = json.load(open(f"{B}/clean/{ds}_meta.json"))
    df = pd.read_csv(f"{B}/clean/{ds}.csv.gz")
    feats = list(meta["findings"].keys())
    X = {s: df.loc[df.split == s, feats].values.astype(float) for s in ["train", "cal", "test"]}
    Y = {s: df.loc[df.split == s, "y"].values.astype(int) for s in ["train", "cal", "test"]}
    out = f"{B}/models/{ds}"
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(7)
    K = {"taiwan": 8, "cdc": 3, "german": 60, "heart": 60}[ds]
    Kc = {"taiwan": 4, "cdc": 2, "german": 30, "heart": 30}[ds]
    metrics = {"dataset": ds, "features": feats, "K_train_replicates": K, "K_cal_replicates": Kc}

    # ---------------- Y-model
    bst, Xcr, ycr = fit_masked(X["train"], Y["train"], X["cal"], Y["cal"], ds, rng, K, Kc)
    bst.save_model(f"{out}/ymodel.txt")
    raw_c = bst.predict(Xcr)
    if ds in SMALL:
        lr = LogisticRegression(C=1e6).fit(np.log(raw_c / (1 - raw_c))[:, None], ycr)
        calib = ("platt", float(lr.coef_[0, 0]), float(lr.intercept_[0]))
    else:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw_c, ycr)
        calib = ("isotonic", iso.X_thresholds_.tolist(), iso.y_thresholds_.tolist())
    pickle.dump(calib, open(f"{out}/ycalib.pkl", "wb"))
    metrics["y_best_iter"] = bst.best_iteration

    def ypred(Xm):
        p = bst.predict(Xm)
        if calib[0] == "platt":
            p = 1 / (1 + np.exp(-(calib[1] * np.log(p / (1 - p)) + calib[2])))
        else:
            p = np.interp(p, calib[1], calib[2])
        return np.clip(p, CLIP, 1 - CLIP)

    # out-of-sample evaluation on TEST: full observation + masked replicates by #observed
    yt = Y["test"]
    pf = ypred(X["test"])
    metrics["y_full"] = {"auc": roc_auc_score(yt, pf), "brier": brier_score_loss(yt, pf), "ece": ece(pf, yt),
                         "brier_baserate": brier_score_loss(yt, np.full(len(yt), Y["train"].mean()))}
    Kt = 10 if ds in SMALL else 3
    Xtr, keep = mask_rows(X["test"], Kt, np.random.default_rng(11))
    ytr = np.repeat(yt, Kt)
    nobs = (~np.isnan(Xtr)).sum(1)
    pt = ypred(Xtr)
    metrics["y_masked_all"] = {"auc": roc_auc_score(ytr, pt), "brier": brier_score_loss(ytr, pt), "ece": ece(pt, ytr)}
    byk = {}
    for lo, hi in [(0, 0), (1, 2), (3, 5), (6, 8), (9, 99)]:
        m = (nobs >= lo) & (nobs <= hi)
        if m.sum() > 50 and len(np.unique(ytr[m])) == 2:
            byk[f"{lo}-{hi}"] = {"n": int(m.sum()), "auc": roc_auc_score(ytr[m], pt[m]) if lo > 0 else None,
                                 "brier": brier_score_loss(ytr[m], pt[m]), "ece": ece(pt[m], ytr[m])}
    metrics["y_masked_by_nobs"] = byk
    # calibration curve (masked test)
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(pt, edges) - 1, 0, 9)
    metrics["y_calibration_curve"] = [{"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}", "n": int((idx == b).sum()),
                                       "mean_pred": float(pt[idx == b].mean()), "obs_rate": float(ytr[idx == b].mean())}
                                      for b in range(10) if (idx == b).sum() > 0]

    # ---------------- f-models
    fm = {}
    for j, f in enumerate(feats):
        others = [i for i in range(len(feats)) if i != j]
        def sub(s):
            m = ~np.isnan(X[s][:, j])
            Xs = X[s][m].copy()
            Xs[:, j] = np.nan  # keep column layout; target column is always unobserved
            return Xs, X[s][m, j].astype(int)
        Xa, ya = sub("train")
        Xb, yb = sub("cal")
        Kf = max(2, K // 2) if ds not in SMALL else K // 2
        fb, _, _ = fit_masked(Xa, ya, Xb, yb, ds, rng, Kf, max(1, Kc // 2))
        fb.save_model(f"{out}/fmodel_{f}.txt")
        Xs, ys = sub("test")
        Xsr, _ = mask_rows(Xs, Kt, np.random.default_rng(13))
        ysr = np.repeat(ys, Kt)
        ps = np.clip(fb.predict(Xsr), CLIP, 1 - CLIP)
        fm[f] = {"auc_masked": roc_auc_score(ysr, ps) if len(np.unique(ysr)) == 2 else None,
                 "brier_masked": brier_score_loss(ysr, ps), "ece_masked": ece(ps, ysr), "best_iter": fb.best_iteration}
        print(f"  f-model {f}: auc={fm[f]['auc_masked']:.3f} ece={fm[f]['ece_masked']:.4f}", flush=True)
    metrics["f_models"] = fm
    json.dump(metrics, open(f"{out}/metrics.json", "w"), indent=1, default=float)
    print(ds, json.dumps({k: metrics[k] for k in ["y_full", "y_masked_all", "y_masked_by_nobs"]}, default=float, indent=0))


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        main(ds)
