"""Reference decision model + empirical VOI + scenario rows for the real market-grounded slice.

usage: python 04_build_env.py [--targets ret_1,...] [--seed 0]
inputs : data/records_base.parquet (02), data/llm/{PRIMARY}/*.jsonl.gz (03, optional until extraction finishes)
outputs: data/items.parquet               one row per firm-quarter: actual values, bins, verbalized texts
         data/ref_models/{target}.txt     LightGBM reference models (+ platt.json), vmodels/{item}.txt
         data/{target}_{split}.jsonl.gz   scenario rows (schema = envs_synth binary track + real-specific fields)
         voi/voi_table_{target}.csv       per-item VOI tables (model preposterior + held-out realized)
         voi/summary.json                 calibration / coverage / coherence diagnostics
"""
import os, sys, json, math, argparse, glob, gzip, time
import numpy as np, pandas as pd
import lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, VOI, RAW, read_jsonl_gz
SYN = './envs_synth'
sys.path.insert(0, SYN)
from core import Model, summarize_root

PRIMARY = os.environ.get('LLM_PRIMARY', 'qwen3_32b')
HS = [1, 5, 20, 60]
TARGETS = [f'ret_{h}' for h in HS] + [f'vol_{h}' for h in HS]
SECTORS = None

# ------------------------------------------------------------------ item catalogue
# kind: 'q' = tercile-binned continuous, 'c' = categorical (fixed values)
ITEMS = {
 'rev_growth':     dict(kind='q', name='Revenue growth', ask='the company\'s revenue growth versus the same quarter a year earlier',
                        fmt=lambda x: f'revenue changed by {100*x:+.1f}% versus the same quarter a year earlier'),
 'gm_change':      dict(kind='q', name='Gross-margin change', ask='the year-over-year change in gross margin',
                        fmt=lambda x: f'gross margin changed by {100*x:+.1f} percentage points year over year'),
 'om_change':      dict(kind='q', name='Operating-margin change', ask='the year-over-year change in operating margin',
                        fmt=lambda x: f'operating margin changed by {100*x:+.1f} percentage points year over year'),
 'eps_sue':        dict(kind='q', name='EPS surprise (time-series)', ask='how far diluted EPS moved versus the same quarter last year, scaled by its usual variability',
                        fmt=lambda x: f'diluted EPS changed year over year by {x:+.1f} times its typical seasonal variation (standardized unexpected earnings)'),
 'analyst_surprise': dict(kind='q', name='EPS vs. analyst consensus', ask='reported EPS relative to the analyst consensus estimate',
                        fmt=lambda x: f'reported EPS came in {x:+.1f}% versus the analyst consensus estimate'),
 'guidance':       dict(kind='c', name='Guidance update', ask='whether the company raised, kept or lowered its financial guidance',
                        values=['raised', 'maintained or newly issued', 'lowered or withdrawn', 'no guidance given'],
                        fmt=lambda v: {'raised': 'the company raised its financial guidance',
                                       'maintained or newly issued': 'the company maintained (or newly issued) its financial guidance',
                                       'lowered or withdrawn': 'the company lowered or withdrew its financial guidance',
                                       'no guidance given': 'the release gives no financial guidance'}[v]),
 'segments':       dict(kind='c', name='Segment results', ask='how the company\'s business segments performed',
                        values=['mostly growing', 'mixed', 'mostly declining', 'not reported'],
                        fmt=lambda v: {'mostly growing': 'all or most business segments grew year over year',
                                       'mixed': 'segment results were mixed (some grew, some declined)',
                                       'mostly declining': 'all or most business segments declined year over year',
                                       'not reported': 'no segment breakdown is reported'}[v]),
 'tone':           dict(kind='c', name='Management tone', ask='the tone of management\'s commentary in the release',
                        values=['positive', 'neutral', 'negative'],
                        fmt=lambda v: f'management\'s tone in the release is {v}'),
 'fcf_margin':     dict(kind='q', name='Free-cash-flow margin', ask='free cash flow (operating cash flow minus capital expenditures) as a share of revenue',
                        fmt=lambda x: f'free cash flow was {100*x:+.1f}% of revenue in the quarter'),
 'accruals':       dict(kind='q', name='Accruals', ask='net income minus operating cash flow, relative to total assets',
                        fmt=lambda x: f'net income minus operating cash flow equalled {100*x:+.2f}% of total assets'),
 'lev_change':     dict(kind='q', name='Leverage change', ask='the year-over-year change in debt-to-assets',
                        fmt=lambda x: f'debt-to-assets changed by {100*x:+.1f} percentage points year over year'),
 'capex_growth':   dict(kind='q', name='Capex growth', ask='the year-over-year growth of capital expenditures',
                        fmt=lambda x: f'capital expenditures changed by {100*x:+.0f}% year over year'),
 'buyback_yield':  dict(kind='q', name='Share buybacks', ask='share repurchases in the quarter relative to total assets',
                        fmt=lambda x: f'share repurchases in the quarter equalled {100*x:.2f}% of total assets'),
 'rnd_intensity':  dict(kind='q', name='R&D intensity', ask='R&D expense as a share of revenue',
                        fmt=lambda x: f'R&D expense was {100*x:.1f}% of revenue'),
 'mom60':          dict(kind='q', name='Pre-announcement momentum', ask='the stock\'s market-adjusted return over the 60 trading days before the announcement',
                        fmt=lambda x: f'the stock\'s market-adjusted return over the prior 60 trading days was {100*(math.exp(x)-1):+.1f}%'),
 'rv_pre':         dict(kind='q', name='Pre-announcement volatility', ask='the stock\'s daily abnormal-return volatility over the prior 60 trading days',
                        fmt=lambda x: f'the stock\'s daily abnormal-return volatility over the prior 60 trading days was {100*x:.2f}% (about {100*x*math.sqrt(252):.0f}% annualized)'),
 'size_dvol':      dict(kind='q', name='Trading size', ask='the stock\'s average daily dollar trading volume (a size proxy)',
                        fmt=lambda x: f'average daily dollar trading volume over the prior 60 days was about ${math.exp(x)/1e6:,.0f} million'),
 'placebo_weekday': dict(kind='c', name='Announcement weekday', ask='the weekday on which the results are announced', placebo=True,
                        values=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                        fmt=lambda v: f'the announcement falls on a {v}'),
 'placebo_code':   dict(kind='c', name='Record code', ask='the first letter of the internal record code of this filing', placebo=True,
                        values=['A', 'B', 'C'], fmt=lambda v: f'the internal record code of this filing begins with the letter {v}'),
}
ITEM_KEYS = list(ITEMS)
GMAP = {'raised': 'raised', 'maintained': 'maintained or newly issued', 'initiated': 'maintained or newly issued',
        'lowered': 'lowered or withdrawn', 'withdrawn': 'lowered or withdrawn', 'mixed': None, 'none': 'no guidance given'}
SMAP = {'all_up': 'mostly growing', 'mostly_up': 'mostly growing', 'mixed': 'mixed', 'mostly_down': 'mostly declining',
        'all_down': 'mostly declining', 'not_reported': 'not reported'}


def load_llm(tag):
    out = {}
    for f in glob.glob(f'{DATA}/llm/{tag}/*.jsonl.gz'):
        for r in read_jsonl_gz(f):
            if r.get('out'):
                out[r['accn']] = r['out']
    return out


def build_items(rec):
    llm = load_llm(PRIMARY)
    g, s, t = [], [], []
    for a in rec.accn_8k:
        o = llm.get(a)
        g.append(GMAP.get(o['guidance']) if o else None)
        s.append(SMAP.get(o['segments']) if o else None)
        t.append(o['tone'] if o else None)
    rec = rec.copy()
    rec['guidance'], rec['segments'], rec['tone'] = g, s, t
    rec['placebo_weekday'] = rec.placebo_weekday.map(lambda i: ITEMS['placebo_weekday']['values'][int(i)] if i < 5 else None)
    rec['placebo_code'] = rec.placebo_code.map(lambda i: 'ABC'[int(i)])
    tr = rec[rec.split == 'train']
    cuts, reps = {}, {}
    for k, it in ITEMS.items():
        if it['kind'] == 'q':
            x = tr[k].astype(float).dropna()
            if len(x) < 100:
                cuts[k] = None; continue
            # winsorize extreme tails for bin representatives only
            q = x.quantile([1 / 3, 2 / 3]).values
            cuts[k] = [float(q[0]), float(q[1])]
            reps[k] = [float(x.quantile(p)) for p in (1 / 6, 1 / 2, 5 / 6)]
    B = {}
    for k, it in ITEMS.items():
        if it['kind'] == 'q':
            v = rec[k].astype(float)
            if cuts[k] is None:
                B[k] = pd.Series(np.nan, index=rec.index)
            else:
                b = np.where(v.isna(), np.nan, np.digitize(v.fillna(0), cuts[k]))
                B[k] = pd.Series(b, index=rec.index)
        else:
            m = {vv: i for i, vv in enumerate(it['values'])}
            B[k] = rec[k].map(lambda z: m.get(z, np.nan) if z is not None else np.nan).astype(float)
    bins = pd.DataFrame({f'b_{k}': B[k] for k in ITEMS})
    rec = pd.concat([rec.reset_index(drop=True), bins.reset_index(drop=True)], axis=1)
    return rec, cuts, reps


def item_text(k, raw):
    it = ITEMS[k]
    return it['fmt'](raw)


def bin_texts(k, reps):
    it = ITEMS[k]
    if it['kind'] == 'q':
        lab = ['low (bottom third)', 'typical (middle third)', 'high (top third)']
        return [{'bin': i, 'value': reps[k][i], 'level': lab[i], 'text': it['fmt'](reps[k][i])} for i in range(3)]
    return [{'bin': i, 'value': v, 'level': v, 'text': it['fmt'](v)} for i, v in enumerate(it['values'])]


def nvals(k):
    return 3 if ITEMS[k]['kind'] == 'q' else len(ITEMS[k]['values'])


# ------------------------------------------------------------------ targets
def make_targets(rec):
    tr = rec[rec.split == 'train']
    med = {}
    Y = {}
    for h in HS:
        Y[f'ret_{h}'] = np.where(rec[f'car_{h}'].isna(), np.nan, (rec[f'car_{h}'] < 0).astype(float))
        med[h] = float(tr[f'lvr_{h}'].median())
        Y[f'vol_{h}'] = np.where(rec[f'lvr_{h}'].isna(), np.nan, (rec[f'lvr_{h}'] > med[h]).astype(float))
    return pd.DataFrame(Y), med


# ------------------------------------------------------------------ reference models
FEATS = [f'b_{k}' for k in ITEM_KEYS] + ['sector_code']


def masked_copies(X, rng, R, kmax=5, full_frac=0.15):
    """R copies of X; in each copy keep a random subset (size 0..kmax, or all with prob full_frac) of available items."""
    out = []
    items = [f'b_{k}' for k in ITEM_KEYS]
    A = X[items].values
    for r in range(R):
        M = np.full(A.shape, np.nan)
        for i in range(len(A)):
            av = np.where(~np.isnan(A[i]))[0]
            if rng.random() < full_frac:
                keep = av
            else:
                k = rng.integers(0, min(kmax, len(av)) + 1)
                keep = rng.choice(av, size=k, replace=False) if k else []
            M[i, keep] = A[i, keep]
        C = X.copy()
        C[items] = M
        out.append(C)
    return pd.concat(out, ignore_index=True)


LGB_Y = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=300, feature_fraction=0.9,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=8, seed=0)
LGB_V = dict(objective='multiclass', learning_rate=0.05, num_leaves=15, min_data_in_leaf=300, feature_fraction=0.9,
             lambda_l2=10.0, verbose=-1, num_threads=8, seed=0)


def fit_y(rec, y, rng, mdir, target):
    tr = rec.split.values == 'train'; ca = rec.split.values == 'calib'
    okt = tr & ~np.isnan(y); okc = ca & ~np.isnan(y)
    Xt = masked_copies(rec.loc[okt, FEATS], rng, 6); yt = np.tile(y[okt], 6)
    Xc = masked_copies(rec.loc[okc, FEATS], rng, 4); yc = np.tile(y[okc], 4)
    for pcol in PLACEBO_COLS:   # placebos are irrelevant by construction: the outcome model never sees them (VOI = EIG = 0)
        Xt[pcol] = np.nan; Xc[pcol] = np.nan
    dt = lgb.Dataset(Xt, yt, categorical_feature=['sector_code'], free_raw_data=False)
    dc = lgb.Dataset(Xc, yc, reference=dt, categorical_feature=['sector_code'])
    bst = lgb.train(LGB_Y, dt, num_boost_round=1500, valid_sets=[dc], callbacks=[lgb.early_stopping(100, verbose=False)])
    # Platt recalibration on the (masked) calibration year
    z = logit(bst.predict(Xc, num_iteration=bst.best_iteration))
    a, b = platt(z, yc)
    bst.save_model(f'{mdir}/{target}.txt', num_iteration=bst.best_iteration)
    json.dump({'a': a, 'b': b, 'best_iter': bst.best_iteration}, open(f'{mdir}/{target}.platt.json', 'w'))
    return bst, (a, b)


def fit_v(rec, k, rng, mdir):
    col = f'b_{k}'
    tr = (rec.split.values == 'train') & rec[col].notna().values
    ca = (rec.split.values == 'calib') & rec[col].notna().values
    nv = nvals(k)
    Xt = masked_copies(rec.loc[tr, FEATS], rng, 4); yt = np.tile(rec.loc[tr, col].values.astype(int), 4)
    Xt[col] = np.nan
    Xc = masked_copies(rec.loc[ca, FEATS], rng, 2); yc = np.tile(rec.loc[ca, col].values.astype(int), 2)
    Xc[col] = np.nan
    for pcol in PLACEBO_COLS:
        Xt[pcol] = np.nan; Xc[pcol] = np.nan
    p = dict(LGB_V, num_class=nv)
    dt = lgb.Dataset(Xt, yt, categorical_feature=['sector_code'], free_raw_data=False)
    dc = lgb.Dataset(Xc, yc, reference=dt, categorical_feature=['sector_code'])
    bst = lgb.train(p, dt, num_boost_round=800, valid_sets=[dc], callbacks=[lgb.early_stopping(50, verbose=False)])
    bst.save_model(f'{mdir}/v_{k}.txt', num_iteration=bst.best_iteration)
    P = bst.predict(Xc, num_iteration=bst.best_iteration)
    ll = float(-np.mean(np.log(P[np.arange(len(yc)), yc] + 1e-12)))
    marg = np.bincount(yt, minlength=nv) / len(yt)
    ll0 = float(-np.mean(np.log(marg[yc] + 1e-12)))
    return bst, {'calib_logloss': ll, 'calib_logloss_marginal': ll0}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def platt(z, y, iters=200):
    a, b = 1.0, 0.0
    for _ in range(iters):  # Newton on logistic loss
        p = sigmoid(a * z + b)
        g = np.array([np.sum((p - y) * z), np.sum(p - y)])
        w = p * (1 - p)
        H = np.array([[np.sum(w * z * z), np.sum(w * z)], [np.sum(w * z), np.sum(w)]]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-8:
            break
    return float(a), float(b)


PLACEBO_COLS = [f'b_{k}' for k, it in ITEMS.items() if it.get('placebo')]


class Ref:
    def __init__(self, ybst, ab, vb, marg=None):
        self.y, self.ab, self.v, self.marg = ybst, ab, vb, marg or {}

    def py(self, X):
        X = X.copy()
        for pcol in PLACEBO_COLS:
            if pcol in X:
                X[pcol] = np.nan
        z = logit(self.y.predict(X))
        return sigmoid(self.ab[0] * z + self.ab[1])

    def pv(self, k, X):
        if ITEMS[k].get('placebo'):   # independent of everything by construction -> training marginal
            return np.tile(self.marg[k], (len(X), 1))
        X = X.copy(); X[f'b_{k}'] = np.nan
        for pcol in PLACEBO_COLS:
            X[pcol] = np.nan
        return self.v[k].predict(X)


def recenter(pv, py_v, p0, iters=60):
    """log-odds shift delta s.t. sum_v pv * sigmoid(logit(py_v)+delta) = p0 (coherent preposterior)."""
    z = logit(py_v)
    lo, hi = -5.0, 5.0
    for _ in range(iters):
        m = (lo + hi) / 2
        if np.dot(pv, sigmoid(z + m)) < p0:
            lo = m
        else:
            hi = m
    d = (lo + hi) / 2
    return sigmoid(z + d), d


def voi_eig_binary(pv, pyv, p0, g, l):
    """myopic VOI (act/pass, U(act,0)=g, U(act,1)=-l, pass=0) and EIG(Y;e) in bits."""
    eu0 = max(g * (1 - p0) - l * p0, 0.0)
    euv = np.maximum(g * (1 - pyv) - l * pyv, 0.0)
    voi = float(np.dot(pv, euv) - eu0)
    def H(p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    eig = float(H(p0) - np.dot(pv, H(pyv)))
    return max(voi, 0.0), max(eig, 0.0)


# ------------------------------------------------------------------ scenarios
PAYOFFS = [(g, l) for g in (1, 2, 3) for l in (1, 2, 3, 5, 8)]
COST_REGIMES = [0.0, 0.005, 0.01, 0.02, 0.04]   # x min(g,l); real VOIs are small, so the grid is finer than synth's
LABELS = 'ABCDEF'


def target_text(target, sector):
    kind, h = target.split('_'); h = int(h)
    hd = f'{h} trading day' + ('s' if h > 1 else '')
    ctx_head = (f'You are a portfolio manager. A company in the {sector} sector has just released its quarterly earnings '
                f'(the company\'s identity and the calendar dates are hidden).')
    if kind == 'ret':
        return dict(act='BUY', pass_='PASS', adverse=f'the stock underperforms the market over the next {hd} '
                    f'(negative market-adjusted return, counting from the first trading session after the release)',
                    adverse_short='underperforms', benign_short='does not underperform',
                    context=ctx_head + f' You must decide whether to BUY the stock now and hold it for {hd}, or PASS.')
    return dict(act='SELL VOLATILITY', pass_='PASS', adverse=f'the stock\'s abnormal volatility over the next {hd}, '
                f'relative to its own pre-announcement volatility, is in the top half of all earnings announcements',
                adverse_short='volatility spike', benign_short='calm',
                context=ctx_head + f' You must decide whether to SELL VOLATILITY (write an option position that profits '
                f'if the stock stays calm over the next {hd}) or PASS.')


def stratum_real(ex, eig_thr=0.005, tol=1e-9):
    voi, eig, net = ex['voi'], ex['eig'], ex['net']
    if not voi:
        return 'other'
    mv, me = max(voi.values()), max(eig.values())
    a_voi = {l for l, v in voi.items() if v >= mv - tol}
    a_eig = {l for l, v in eig.items() if v >= me - tol}
    maxnet = max(net.values())
    if maxnet <= tol and ex['dp_first_action'] == 'S' and me > eig_thr:
        return 'S2'
    if mv > tol and not (a_voi & a_eig) and max(voi[l] for l in a_eig) <= 0.5 * mv:
        return 'S1'
    if mv > tol and (a_voi & a_eig) and maxnet > tol:
        return 'S3'
    return 'other'


def run_target(target, rec, Y, ref, reps, sector_rate, splits=('calib', 'dev', 'test'), seed=0):
    y_all = Y[target].values
    out_rows = {s: [] for s in splits}
    item_rows = []
    sel = np.where(np.isin(rec.split.values, splits) & ~np.isnan(y_all))[0]
    R = rec.iloc[sel].reset_index(drop=True); yv = y_all[sel]
    n = len(R)
    rng = np.random.default_rng(abs(hash((target, seed))) % 2 ** 32)
    nonpl = [k for k in ITEM_KEYS if not ITEMS[k].get('placebo')]
    pls = [k for k in ITEM_KEYS if ITEMS[k].get('placebo')]
    # observed sets
    obs_sets = []
    for i in range(n):
        av = [k for k in nonpl if not np.isnan(R.at[i, f'b_{k}'])]
        ko = rng.choice([0, 1, 2], p=[0.4, 0.4, 0.2]) if av else 0
        obs_sets.append(list(rng.choice(av, size=min(ko, len(av)), replace=False)) if ko else [])
    # base rows: x_E (scenario) and x_0 (sector only)
    X0 = R[FEATS].copy()
    for k in ITEM_KEYS:
        X0[f'b_{k}'] = np.nan
    XE = X0.copy()
    for i, E in enumerate(obs_sets):
        for k in E:
            XE.at[i, f'b_{k}'] = R.at[i, f'b_{k}']
    p0E = ref.py(XE); p00 = ref.py(X0)
    # per item: p(v|E), p(Y|E,v) for both bases
    res = {}
    for base, XB, pb in (('E', XE, p0E), ('0', X0, p00)):
        for k in ITEM_KEYS:
            nv = nvals(k)
            PV = ref.pv(k, XB)                            # n x nv
            PY = np.zeros((n, nv))
            for v in range(nv):
                Xv = XB.copy(); Xv[f'b_{k}'] = float(v)
                PY[:, v] = ref.py(Xv)
            res[(base, k)] = (PV, PY)
    for i in range(n):
        r = R.iloc[i]
        sector = r.sector if isinstance(r.sector, str) else 'unclassified'
        T = target_text(target, sector)
        E = obs_sets[i]
        av = [k for k in ITEM_KEYS if not np.isnan(r[f'b_{k}']) and k not in E]
        p0 = float(p0E[i])
        # payoff near the decision boundary of the reference belief (otherwise nothing can flip the decision)
        cand_pay = [pq for pq in PAYOFFS if abs(pq[0] / (pq[0] + pq[1]) - p0) <= 0.08]
        if not cand_pay:
            cand_pay = sorted(PAYOFFS, key=lambda pq: abs(pq[0] / (pq[0] + pq[1]) - p0))[:2]
        g, l = cand_pay[rng.integers(len(cand_pay))]
        regime = float(rng.choice(COST_REGIMES))
        # all-item stats at E (scenario payoff) and at E=0 (symmetric payoff g=l=1)
        allstats = {}
        for k in av + [k for k in E]:
            pass
        for k in [k for k in ITEM_KEYS if not np.isnan(r[f'b_{k}'])]:
            st = {}
            for base, pb, gg, ll in (('E', p0, g, l), ('0', float(p00[i]), 1, 1)):
                if base == 'E' and k in E:
                    continue
                PV, PY = res[(base, k)]
                pv = PV[i]; pyv, d = recenter(pv, PY[i], pb)
                voi, eig = voi_eig_binary(pv, pyv, pb, gg, ll)
                av_bin = int(r[f'b_{k}'])
                yy = yv[i]
                a0 = 1 if gg * (1 - pb) - ll * pb > 0 else 0
                a1 = 1 if gg * (1 - pyv[av_bin]) - ll * pyv[av_bin] > 0 else 0
                u = lambda a: (gg if yy == 0 else -ll) if a else 0.0
                st[base] = {'voi': voi, 'eig': eig, 'delta': float(d), 'pv': pv.tolist(), 'pyv': pyv.tolist(), 'p0': pb,
                            'real_util_gain': u(a1) - u(a0),
                            'real_logscore_gain': float(np.log(pyv[av_bin] if yy == 1 else 1 - pyv[av_bin]) -
                                                        np.log(pb if yy == 1 else 1 - pb))}
            allstats[k] = st
            item_rows.append({'target': target, 'split': r.split, 'rid': r.rid, 'item': k, 'n_obs': len(E),
                              'g': g, 'l': l, **{f'{b}_{m}': st[b][m] for b in st for m in
                                                 ('voi', 'eig', 'real_util_gain', 'real_logscore_gain', 'delta')}})
        # candidate set: 1 placebo + up to 5 real items
        avp = [k for k in av if k in pls]; avn = [k for k in av if k not in pls]
        cands = list(rng.choice(avn, size=min(5, len(avn)), replace=False)) + ([str(rng.choice(avp))] if avp else [])
        rng.shuffle(cands)
        labs = list(LABELS[:len(cands)])
        mins = min(g, l)
        checks, items, outcomes_ep = [], [], {}
        for lab, k in zip(labs, cands):
            st = allstats[k]['E']
            pv = np.array(st['pv']); pyv = np.array(st['pyv'])
            lik = [(pv * (1 - pyv) / (1 - p0)).tolist(), (pv * pyv / p0).tolist()]
            cost = 0.0 if regime == 0 else round(regime * mins * float(rng.choice([0.5, 1.0, 2.0])), 5)
            bt = bin_texts(k, reps)
            items.append({'label': lab, 'lik': lik, 'cost': cost, 'outcomes': [b['text'] for b in bt]})
            actual = item_text(k, r[k])
            checks.append({'label': lab, 'item': k, 'name': ITEMS[k]['name'], 'ask': ITEMS[k]['ask'], 'cost': cost,
                           'is_placebo': bool(ITEMS[k].get('placebo')), 'kind': ITEMS[k]['kind'],
                           'bins': bt, 'p_bins': pv.tolist(), 'p_y_given_bin': pyv.tolist(),
                           'actual_bin': int(r[f'b_{k}']), 'actual_text': actual,
                           'finding_text': actual})
            outcomes_ep[lab] = int(r[f'b_{k}'])
        model = {'prior': [1 - p0, p0], 'utility': [[float(g), float(-l)], [0.0, 0.0]], 'items': items, 'observed': [],
                 'budget': len(items)}
        M = Model(model)
        ex = summarize_root(M)
        exact = {'posterior': p0, 'posterior_vec': ex['posterior'], 'eu_act': ex['eu'][0], 'eu_pass': ex['eu'][1],
                 'best_action': T['act'] if ex['best_action'] == 0 else T['pass_'], 'threshold': g / (g + l)}
        for kk in ['voi', 'eig', 'eig_dec', 'net', 'q']:
            exact[kk] = ex[kk]
        for kk in ['gold_choice', 'gold_ties', 'dp_value', 'dp_first_action', 'dp_first_ties', 'myopic_value', 'value_of_stopping']:
            exact[kk] = ex[kk]
        exact['stratum'] = stratum_real(exact)
        exact['all_items'] = {k: {'voi': allstats[k]['E']['voi'], 'eig': allstats[k]['E']['eig']} for k in av}
        pay = (f'Payoffs: if you {T["act"]} and the adverse case does not occur, you gain {g}; if it occurs '
               f'({T["adverse"]}), you lose {l}. {T["pass_"]} pays 0.')
        br = sector_rate.get((target, sector), sector_rate[(target, '_all')])
        base_rate = (f'Base rate: in past years, about {round(100 * br)}% of earnings announcements by {sector} companies '
                     f'were followed by the adverse case.')
        known = [f'- {ITEMS[k]["name"]}: {item_text(k, r[k])}.' for k in E]
        known_txt = ('Already known:\n' + '\n'.join(known)) if known else 'Already known: nothing beyond the sector and the base rate.'
        opts = {}
        for c in checks:
            cost = 'free' if c['cost'] == 0 else f'costs {c["cost"]:g}'
            opts[c['label']] = f'({c["label"]}) {c["name"]} ({cost}): reports {c["ask"]}.'
        stop = f'(S) Stop and decide now ({T["act"]} or {T["pass_"]}).'
        options = ('You may request any of the following items (each at most once) before deciding.\n' +
                   '\n'.join(opts[l_] for l_ in labs) + '\n' + stop)
        parts = {'context': T['context'], 'payoff': pay, 'base_rate': base_rate, 'known': known_txt, 'options': options,
                 'option_lines': opts, 'stop_line': stop, 'budget': ''}
        text = '\n\n'.join([T['context'], pay, base_rate, known_txt, options])
        row = {'id': f'real_{target}_{r.split}_{i:05d}', 'track': 'real', 'domain': 'finance_real', 'family': 'real',
               'target': target, 'split': r.split, 'stratum': exact['stratum'], 'prior': p0, 'g': g, 'l': l,
               'cost_regime': regime, 'budget': len(items), 'costs': {c['label']: c['cost'] for c in checks},
               'checks': checks, 'observed': [{'item': k, 'name': ITEMS[k]['name'], 'text': item_text(k, r[k]),
                                               'bin': int(r[f'b_{k}'])} for k in E],
               'label_order': labs, 'actions': [T['act'], T['pass_']], 'states': [T['benign_short'], T['adverse_short']],
               'text_parts': parts, 'text': text, 'model': model, 'exact': exact,
               'episodes': [{'state': int(yv[i]), 'y': int(yv[i]), 'outcomes': outcomes_ep, 'source': 'actual'}],
               'realized': {'y': int(yv[i]), 'car': {h: r[f'car_{h}'] for h in HS}, 'lvr': {h: r[f'lvr_{h}'] for h in HS}},
               'sector': sector, 'day0': r.day0,
               '_meta_do_not_show': {'rid': r.rid, 'cik': int(r.cik), 'ticker': r.ticker, 'period_end': r.period_end,
                                     'accn_8k': r.accn_8k}}
        out_rows[r.split].append(row)
    return out_rows, pd.DataFrame(item_rows)


def voi_table(df, split='test'):
    d = df[df.split == split]
    rows = []
    for k, g in d.groupby('item'):
        def ms(c):
            x = g[c].dropna().values
            return (float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float('nan'))
        r = {'item': k, 'name': ITEMS[k]['name'], 'placebo': bool(ITEMS[k].get('placebo')), 'n': len(g)}
        for c in ('0_voi', '0_eig', '0_real_util_gain', '0_real_logscore_gain', 'E_voi', 'E_eig', 'E_real_util_gain',
                  'E_real_logscore_gain'):
            if c in g:
                r[c], r[c + '_se'] = ms(c)
        rows.append(r)
    t = pd.DataFrame(rows).sort_values('0_voi', ascending=False)
    # argmax share at E=0
    piv = d.pivot_table(index='rid', columns='item', values='0_voi')
    am = piv.idxmax(axis=1).value_counts(normalize=True)
    t['share_argmax_voi0'] = t.item.map(am).fillna(0.0)
    piv2 = d.pivot_table(index='rid', columns='item', values='0_eig')
    am2 = piv2.idxmax(axis=1).value_counts(normalize=True)
    t['share_argmax_eig0'] = t.item.map(am2).fillna(0.0)
    return t


def write_gz(path, rows):
    with gzip.open(path + '.tmp', 'wt') as f:
        for r in rows:
            f.write(json.dumps(r, default=lambda o: o.item() if hasattr(o, 'item') else str(o)) + '\n')
    os.replace(path + '.tmp', path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', default=','.join(TARGETS))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--records', default=f'{DATA}/records_base.parquet')
    a = ap.parse_args()
    t0 = time.time()
    rec = pd.read_parquet(a.records)
    rec, cuts, reps = build_items(rec)
    global ITEM_KEYS, FEATS
    trm = rec.split == 'train'
    dropped = [k for k in ITEM_KEYS if rec.loc[trm, f'b_{k}'].notna().sum() < 300]
    ITEM_KEYS = [k for k in ITEM_KEYS if k not in dropped]
    FEATS = [f'b_{k}' for k in ITEM_KEYS] + ['sector_code']
    print('items', ITEM_KEYS, 'dropped (coverage)', dropped, flush=True)
    sec = sorted(rec.sector.fillna('unclassified').unique())
    rec['sector'] = rec.sector.fillna('unclassified')
    rec['sector_code'] = rec.sector.map({s: i for i, s in enumerate(sec)}).astype(int)
    Y, med = make_targets(rec)
    rec.to_parquet(f'{DATA}/items.parquet')
    json.dump({'cuts': cuts, 'reps': reps, 'vol_median_train': med, 'sectors': sec,
               'coverage': {k: float(rec[f'b_{k}'].notna().mean()) for k in ITEM_KEYS},
               'n_by_split': rec.split.value_counts().to_dict()}, open(f'{DATA}/bins.json', 'w'), indent=1)
    mdir = f'{DATA}/ref_models'; os.makedirs(mdir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    vb, vdiag = {}, {}
    for k in ITEM_KEYS:
        if ITEMS[k].get('placebo'):
            continue
        vb[k], vdiag[k] = fit_v(rec, k, rng, mdir)
        print(f'vmodel {k} {vdiag[k]} {time.time()-t0:.0f}s', flush=True)
    summ = {'vmodels': vdiag, 'targets': {}}
    sector_rate = {}
    tr = rec.split == 'train'
    for t in a.targets.split(','):
        yv = Y[t]
        sector_rate[(t, '_all')] = float(np.nanmean(yv[tr]))
        for s in sec:
            m = tr & (rec.sector == s)
            if m.sum() > 50:
                sector_rate[(t, s)] = float(np.nanmean(yv[m]))
        ybst, ab = fit_y(rec, yv.values, rng, mdir, t)
        marg = {k: (np.bincount(rec.loc[tr, f'b_{k}'].dropna().astype(int), minlength=nvals(k)) /
                    rec.loc[tr, f'b_{k}'].notna().sum()) for k in ITEM_KEYS if ITEMS[k].get('placebo')}
        ref = Ref(ybst, ab, vb, marg)
        # diagnostics: full-information and sector-only discrimination / calibration on dev and test
        dg = {}
        for sp in ('calib', 'dev', 'test'):
            m = (rec.split == sp).values & ~np.isnan(yv.values)
            X = rec.loc[m, FEATS]
            p = ref.py(X); yy = yv.values[m]
            X0 = X.copy()
            for k in ITEM_KEYS:
                X0[f'b_{k}'] = np.nan
            p0 = ref.py(X0)
            dg[sp] = {'n': int(m.sum()), 'base_rate': float(yy.mean()), 'auc_full': auc(yy, p), 'auc_sector_only': auc(yy, p0),
                      'logloss_full': ll(yy, p), 'logloss_sector_only': ll(yy, p0), 'brier_full': float(np.mean((p - yy) ** 2)),
                      'ece_full': ece(yy, p)}
        rows, items_df = run_target(t, rec, Y, ref, reps, sector_rate, seed=a.seed)
        for sp, rr in rows.items():
            write_gz(f'{DATA}/{t}_{sp}.jsonl.gz', rr)
        items_df.to_parquet(f'{VOI}/item_stats_{t}.parquet')
        for sp in ('dev', 'test'):
            voi_table(items_df, sp).to_csv(f'{VOI}/voi_table_{t}_{sp}.csv', index=False)
        strata = {sp: pd.Series([r['stratum'] for r in rr]).value_counts().to_dict() for sp, rr in rows.items()}
        gold_stop = {sp: float(np.mean([r['exact']['gold_choice'] == 'S' for r in rr])) if rr else None for sp, rr in rows.items()}
        summ['targets'][t] = {'diag': dg, 'strata': strata, 'gold_stop_rate': gold_stop,
                              'coherence_abs_delta_logodds_mean': float(items_df['E_delta'].abs().mean())}
        print(t, json.dumps(summ['targets'][t]), f'{time.time()-t0:.0f}s', flush=True)
        json.dump({'vmodels': summ['vmodels'], t: summ['targets'][t]}, open(f'{VOI}/summary_{t}.json', 'w'), indent=1)
    matrix()
    print('DONE', f'{time.time()-t0:.0f}s')


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(set(y)) > 1 else float('nan')


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y, p, nb=10):
    b = np.minimum((p * nb).astype(int), nb - 1)
    return float(sum(abs(y[b == i].mean() - p[b == i].mean()) * (b == i).mean() for i in range(nb) if (b == i).any()))



def matrix():
    """voi/voi_matrix_{split}.csv: item x target summary (model VOI at E=0 with g=l=1, realized utility gain, EIG)."""
    for sp in ('dev', 'test'):
        out = []
        for t in TARGETS:
            f = f'{VOI}/voi_table_{t}_{sp}.csv'
            if os.path.exists(f):
                d = pd.read_csv(f)
                d['target'] = t
                out.append(d)
        if not out:
            continue
        d = pd.concat(out)
        rows = []
        for m in ('0_voi', '0_real_util_gain', '0_real_util_gain_se', '0_eig', '0_real_logscore_gain', 'share_argmax_voi0'):
            p = d.pivot_table(index='item', columns='target', values=m)
            p.columns = [f'{m}|{c}' for c in p.columns]
            rows.append(p)
        pd.concat(rows, axis=1).to_csv(f'{VOI}/voi_matrix_{sp}.csv')


if __name__ == '__main__':
    main()
