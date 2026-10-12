"""Build point-in-time firm-quarter records anchored at the earnings announcement (first 8-K Item 2.02 after the
fiscal-quarter end and no later than the 10-Q/10-K for that quarter).

Output: data/records.parquet (one row per cik x fiscal quarter) with
  ids/time   : rid, cik, ticker, sector, period_end, form, ann_time_et, day0, split
  items      : continuous fundamentals/market items (see ITEMS below), *_avail flags
  outcomes   : car_{1,5,20,60}, lvr_{1,5,20,60} (log realized-vol ratio post/pre), n_post_{h}
Point-in-time rules:
  * every XBRL value is the FIRST-filed vintage of that (concept, start, end) fact (as originally reported, no restatements)
  * prior-period values must have been filed <= announcement date; current-quarter values come from the 10-Q/10-K of
    that quarter (filed after the release; the release discloses the same headline numbers -> 'release-time proxy',
    validated in 04_validate_proxy.py by matching the XBRL revenue/EPS inside the release text)
  * market items use sessions <= day0-2 only
"""
import os, sys, json, glob, math, hashlib
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, DATA, EQ, read_jsonl_gz

REV = ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'RevenueFromContractWithCustomerIncludingAssessedTax',
       'SalesRevenueNet', 'RevenuesNetOfInterestExpense', 'RegulatedAndUnregulatedOperatingRevenue', 'ElectricUtilityRevenue',
       'SalesRevenueGoodsNet', 'Revenue']
COGS = ['CostOfRevenue', 'CostOfGoodsAndServicesSold', 'CostOfGoodsSold', 'CostOfServices']
FLOW = {'rev': REV, 'gp': ['GrossProfit'], 'cogs': COGS, 'oi': ['OperatingIncomeLoss'],
        'ni': ['NetIncomeLoss', 'ProfitLoss'], 'eps': ['EarningsPerShareDiluted', 'EarningsPerShareBasicAndDiluted', 'EarningsPerShareBasic',
                                                       'DilutedEarningsLossPerShare', 'BasicEarningsLossPerShare'],
        'cfo': ['NetCashProvidedByUsedInOperatingActivities', 'NetCashProvidedByUsedInOperatingActivitiesContinuingOperations'],
        'capex': ['PaymentsToAcquirePropertyPlantAndEquipment', 'PaymentsToAcquireProductiveAssets', 'PaymentsForCapitalImprovements'],
        'buyback': ['PaymentsForRepurchaseOfCommonStock', 'PaymentsForRepurchaseOfEquity'],
        'div': ['PaymentsOfDividends', 'PaymentsOfDividendsCommonStock'],
        'rnd': ['ResearchAndDevelopmentExpense']}
STOCK = {'assets': ['Assets'], 'ltd_nc': ['LongTermDebtNoncurrent', 'LongTermDebtAndCapitalLeaseObligations'],
         'ltd_c': ['LongTermDebtCurrent', 'LongTermDebtAndCapitalLeaseObligationsCurrent', 'DebtCurrent'],
         'ltd': ['LongTermDebt', 'DebtLongtermAndShorttermCombinedAmount'], 'stb': ['ShortTermBorrowings', 'CommercialPaper'],
         'cash': ['CashAndCashEquivalentsAtCarryingValue', 'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents'],
         'equity': ['StockholdersEquity', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest']}
SPLITS = [('train', '2015-01-01', '2022-12-31'), ('calib', '2023-01-01', '2023-12-31'), ('dev', '2024-01-01', '2024-12-31'),
          ('test', '2025-01-01', '2026-12-31')]
HS = [1, 5, 20, 60]


def load_facts(cik):
    f = f'{RAW}/facts/{cik}.jsonl.gz'
    if not os.path.exists(f):
        return None
    d = pd.DataFrame(list(read_jsonl_gz(f)))
    if d.empty:
        return None
    d = d[d.u.isin(['USD', 'USD/shares'])]
    d['e'] = pd.to_datetime(d.e); d['s'] = pd.to_datetime(d.s); d['filed'] = pd.to_datetime(d.filed)
    # first-filed vintage per fact
    d = d.sort_values('filed').drop_duplicates(['c', 's', 'e'], keep='first')
    return d


def quarterly_flow(d, concepts):
    """{period_end: (value, filed)} of 3-month values, first concept with data wins per end date."""
    out = {}
    for c in concepts:
        x = d[(d.c == c) & d.s.notna()]
        if x.empty:
            continue
        x = x.assign(dur=(x.e - x.s).dt.days)
        q = x[(x.dur >= 80) & (x.dur <= 100)]
        for _, r in q.iterrows():
            out.setdefault(r.e, (float(r.v), r.filed))
        # YTD differencing for missing quarters
        ytd = x[(x.dur > 100) & (x.dur <= 380)]
        allx = x[(x.dur >= 80) & (x.dur <= 380)]
        for _, r in ytd.iterrows():
            if r.e in out:
                continue
            prev = allx[(allx.s == r.s) & (allx.e < r.e) & ((r.e - allx.e).dt.days.between(80, 100))]
            if len(prev):
                p = prev.iloc[0]
                out[r.e] = (float(r.v) - float(p.v), max(r.filed, p.filed))
    return out


def stock_val(d, concepts):
    out = {}
    for c in concepts:
        x = d[(d.c == c)]
        for _, r in x.iterrows():
            out.setdefault(r.e, (float(r.v), r.filed))
    return out


def near(dct, e, tol=20):
    """value at the period end closest to e within tol days."""
    best = None
    for k in dct:
        dd = abs((k - e).days)
        if dd <= tol and (best is None or dd < best[0]):
            best = (dd, k)
    return dct[best[1]] if best else None


def pit(v, t):
    """value only if filed <= t."""
    return v[0] if (v is not None and v[1] <= t) else None


class Market:
    def __init__(self):
        self.R = pd.read_parquet(f'{EQ}/processed/panel_D_ret.parquet')
        self.DV = pd.read_parquet(f'{EQ}/processed/panel_D_dvol.parquet')
        X = pd.read_parquet(f'{EQ}/processed/extra_D_ret.parquet')
        self.M = X['SPY'].reindex(self.R.index)
        self.sess = self.R.index


def ann_time(acc_utc, yahoo):
    """Announcement time in ET: earliest of the 8-K acceptance and a Yahoo earnings timestamp in [t-3d, t]."""
    t = pd.Timestamp(acc_utc)
    t = t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')
    src = '8k'
    if yahoo is not None and len(yahoo):
        c = yahoo[(yahoo >= t - pd.Timedelta(days=3)) & (yahoo <= t + pd.Timedelta(hours=1))]
        if len(c) and c.min() < t:
            t = c.min(); src = 'yahoo'
    return t.tz_convert('America/New_York'), src


def day0_index(sess, t_et):
    """First session whose close-to-close return can reflect the news: same day if released before 16:00 ET on a session,
    else the next session."""
    d = pd.Timestamp(t_et.date())
    i = sess.searchsorted(d)
    if i < len(sess) and sess[i] == d and (t_et.hour < 16):
        return i
    if i < len(sess) and sess[i] == d:
        return i + 1
    return i


def outcomes(mk, tk, i0):
    r = mk.R[tk].values; m = mk.M.values
    if i0 - 250 < 0 or i0 >= len(r):
        return None
    est = slice(i0 - 250, i0 - 20)
    ri, mi = r[est], m[est]
    ok = ~np.isnan(ri) & ~np.isnan(mi)
    if ok.sum() < 100:
        return None
    beta = np.cov(ri[ok], mi[ok])[0, 1] / np.var(mi[ok], ddof=1)
    beta = 0.67 * beta + 0.33
    ar = r - beta * m
    pre = ar[i0 - 61:i0 - 1]
    pre = pre[~np.isnan(pre)]
    if len(pre) < 40:
        return None
    rvpre = float(np.sqrt(np.mean((pre - pre.mean()) ** 2)))
    out = {'beta': float(beta), 'mom60': float(np.nansum(ar[i0 - 61:i0 - 1])), 'rv_pre': rvpre,
           'size_dvol': float(np.nanmean(mk.DV[tk].values[i0 - 61:i0 - 1])),
           'ret_pre1': float(r[i0 - 1]) if not np.isnan(r[i0 - 1]) else None}
    for h in HS:
        w = ar[i0:i0 + h]
        if i0 + h > len(ar) or np.isnan(w).sum() > max(0, h // 10):
            out[f'car_{h}'] = None; out[f'lvr_{h}'] = None; continue
        w = w[~np.isnan(w)]
        out[f'car_{h}'] = float(w.sum())
        out[f'lvr_{h}'] = float(0.5 * np.log(np.mean(w ** 2) / rvpre ** 2 + 1e-12))
    return out


def safe_div(a, b):
    if a is None or b is None or b == 0 or (isinstance(b, float) and math.isnan(b)):
        return None
    return a / b


def build_cik(args):
    cik, tickers, sector, yahoo = args
    d = load_facts(cik)
    fp = f'{RAW}/filings/{cik}.json'
    if d is None or not os.path.exists(fp):
        return []
    fl = json.load(open(fp))['filings']
    per = [f for f in fl if f['form'] in ('10-Q', '10-K') and f['reportDate']]
    k8 = sorted([f for f in fl if f['form'] == '8-K' and '2.02' in (f['items'] or '')], key=lambda f: f['acceptance'])
    F = {k: quarterly_flow(d, v) for k, v in FLOW.items()}
    S = {k: stock_val(d, v) for k, v in STOCK.items()}
    mk = MK
    tk = [t for t in tickers if t in mk.R.columns]
    if not tk:
        return []
    tk = max(tk, key=lambda t: mk.R[t].notna().sum())
    rows = []
    for f in sorted(per, key=lambda f: f['reportDate']):
        e = pd.Timestamp(f['reportDate']); fd = pd.Timestamp(f['filingDate'])
        if e < pd.Timestamp('2014-09-01'):
            continue
        cands = [k for k in k8 if e < pd.Timestamp(k['filingDate']) <= fd]
        if not cands:
            continue
        k = cands[0]
        t_et, src = ann_time(k['acceptance'], yahoo)
        t = pd.Timestamp(t_et.date())
        def cur(key):
            v = near(F[key], e, 10)
            return v[0] if v is not None else None
        def lag(key, days):
            return pit(near(F[key], e - pd.Timedelta(days=days), 20), t)
        def cs(key, days=0):
            v = near(S[key], e - pd.Timedelta(days=days), 20)
            return (v[0] if v is not None else None) if days == 0 else pit(v, t)
        rev, rev4 = cur('rev'), lag('rev', 364)
        gp = cur('gp'); cogs = cur('cogs')
        if gp is None and rev is not None and cogs is not None:
            gp = rev - cogs
        gp4 = lag('gp', 364)
        if gp4 is None:
            c4 = lag('cogs', 364)
            gp4 = rev4 - c4 if (rev4 is not None and c4 is not None) else None
        oi, oi4 = cur('oi'), lag('oi', 364)
        eps, eps4 = cur('eps'), lag('eps', 364)
        # SUE: seasonal random walk, scaled by std of past seasonal EPS changes (8 quarters, PIT)
        dif = []
        for j in range(1, 9):
            a, b = lag('eps', 91 * j), lag('eps', 91 * j + 364)
            if a is not None and b is not None:
                dif.append(a - b)
        sue = None
        if eps is not None and eps4 is not None and len(dif) >= 4:
            sd = float(np.std(dif, ddof=1))
            sue = (eps - eps4) / max(sd, 0.01)
        cfo, capex, ni = cur('cfo'), cur('capex'), cur('ni')
        bb, rnd = cur('buyback'), cur('rnd')
        capex4 = lag('capex', 364)
        assets_prev = cs('assets', 91) or cs('assets', 0)
        assets = cs('assets', 0)
        def debt(days):
            g = (lambda key: cs(key, days))
            nc, c, l, sb = g('ltd_nc'), g('ltd_c'), g('ltd'), g('stb')
            if nc is None and l is None:
                return None
            base = nc if nc is not None else l
            return base + (c or 0) + (sb or 0) if nc is not None else l + (sb or 0)
        a4 = cs('assets', 364)
        lev, lev4 = safe_div(debt(0), assets), safe_div(debt(364), a4)
        i0 = day0_index(mk.sess, t_et)
        oc = outcomes(mk, tk, i0) if i0 < len(mk.sess) else None
        if oc is None:
            continue
        rid = f'{cik}_{e.date()}'
        row = {'rid': rid, 'cik': cik, 'ticker': tk, 'sector': sector, 'period_end': e.date().isoformat(),
               'form': f['form'], 'accn_8k': k['accn'], 'accn_10q': f['accn'], 'filed_10q': f['filingDate'],
               'ann_time_et': t_et.isoformat(), 'ann_src': src, 'day0': mk.sess[i0].date().isoformat(),
               'lag_days': (t - e).days,
               'rev_growth': safe_div(rev, rev4) - 1 if safe_div(rev, rev4) is not None and rev4 > 0 else None,
               'gm_change': (safe_div(gp, rev) - safe_div(gp4, rev4)) if (safe_div(gp, rev) is not None and safe_div(gp4, rev4) is not None) else None,
               'om_change': (safe_div(oi, rev) - safe_div(oi4, rev4)) if (safe_div(oi, rev) is not None and safe_div(oi4, rev4) is not None) else None,
               'eps_sue': sue,
               'fcf_margin': safe_div((cfo - capex) if (cfo is not None and capex is not None) else None, rev),
               'accruals': safe_div((ni - cfo) if (ni is not None and cfo is not None) else None, assets_prev),
               'lev_change': (lev - lev4) if (lev is not None and lev4 is not None) else None,
               'capex_growth': (safe_div(capex, capex4) - 1) if (safe_div(capex, capex4) is not None and capex4 > 0) else None,
               'buyback_yield': safe_div(bb if bb is not None else (0.0 if cfo is not None else None), assets_prev),
               'rnd_intensity': safe_div(rnd, rev),
               'rev_level': rev, 'eps_level': eps,
               }
        for kk in ('rev_growth', 'gm_change', 'om_change', 'fcf_margin', 'accruals', 'capex_growth', 'rnd_intensity'):
            v = row[kk]
            if v is not None and (math.isnan(v) or abs(v) > 50):
                row[kk] = None
        row.update(oc)
        # placebos
        row['placebo_weekday'] = int(pd.Timestamp(row['day0']).weekday())
        row['placebo_code'] = int(hashlib.md5(rid.encode()).hexdigest()[:2], 16) % 3
        rows.append(row)
    return rows


MK = None


def init():
    global MK
    MK = Market()


if __name__ == '__main__':
    U = pd.read_csv(f'{RAW}/universe.csv')
    ye = pd.read_parquet(f'{EQ}/processed/earnings_events.parquet')
    ye['earnings_datetime'] = pd.to_datetime(ye.earnings_datetime, utc=True)
    yd = {t: g.earnings_datetime.sort_values().reset_index(drop=True) for t, g in ye.groupby('ticker')}
    jobs = []
    for cik, g in U.groupby('cik'):
        tks = g.ticker.tolist()
        y = pd.concat([yd[t] for t in tks if t in yd]) if any(t in yd for t in tks) else None
        jobs.append((int(cik), tks, g.gics_sector.iloc[0], y))
    only = sys.argv[1:] and [int(x) for x in sys.argv[1:]]
    if only:
        jobs = [j for j in jobs if j[0] in only]
    rows = []
    with ProcessPoolExecutor(32, initializer=init) as ex:
        for r in ex.map(build_cik, jobs, chunksize=2):
            rows.extend(r)
    df = pd.DataFrame(rows)
    # Yahoo consensus surprise (currently listed names; survivorship-biased coverage -> availability flag)
    ye['tk'] = ye.ticker
    df['day0'] = pd.to_datetime(df.day0)
    ye['d'] = ye.earnings_datetime.dt.tz_convert('America/New_York').dt.tz_localize(None).dt.normalize()
    sp = []
    for _, r in df.iterrows():
        x = ye[(ye.tk == r.ticker) & (ye.d >= r.day0 - pd.Timedelta(days=5)) & (ye.d <= r.day0 + pd.Timedelta(days=1))]
        sp.append(float(x.surprise_pct.iloc[0]) if len(x) and pd.notna(x.surprise_pct.iloc[0]) else None)
    df['analyst_surprise'] = sp
    df['split'] = None
    for s, a, b in SPLITS:
        df.loc[(df.day0 >= a) & (df.day0 <= b), 'split'] = s
    df = df[df.split.notna()].reset_index(drop=True)
    # point-in-time index membership at day 0 (avoids selecting firms on their FUTURE inclusion in the S&P 500)
    mem = pd.read_parquet(f'{EQ}/processed/member_D.parquet')
    df['pit_member'] = [bool(mem.at[d, t]) if (d in mem.index and t in mem.columns) else False for d, t in zip(df.day0, df.ticker)]
    print('pit_member share', df.pit_member.mean())
    df = df[df.pit_member].reset_index(drop=True)
    df['day0'] = df.day0.dt.date.astype(str)
    out = f'{DATA}/records_base.parquet' if not only else f'{DATA}/records_base_test.parquet'
    df.to_parquet(out)
    print(df.shape, df.split.value_counts().to_dict())
    print(df.notna().mean().round(3).to_string())
