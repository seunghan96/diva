"""Validation of (a) the LLM extraction and (b) the release-time proxy for current-quarter XBRL values.

(a) LLM extraction (primary = qwen3_32b, second extractor = mistral24b, different family):
    - parse/validity rate; label distributions
    - inter-model agreement + Cohen's kappa on the collapsed labels used as items (guidance 4-way, segments 4-way, tone 3-way)
    - quote grounding: the primary's guidance_quote is a (whitespace/case-normalized) substring of the masked input
    - lexical consistency: 'raised' quotes contain raise-type verbs, 'lowered' quotes contain lower-type verbs, etc.
    - market validity: mean day-0 abnormal return by guidance label (raised should exceed lowered)
    - audit sample (stratified, 20 per primary guidance label) written to voi/llm_audit_sample.jsonl for manual audit
(b) proxy: does the XBRL current-quarter revenue (in $ millions / billions, several roundings) or diluted EPS appear in the
    8-K release text? Reported per year.
output: voi/validation.json
"""
import os, sys, json, glob, re, random
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, RAW, VOI, read_jsonl_gz
from mask import mask_text, select_snippet

GM = {'raised': 'raised', 'maintained': 'maintained/new', 'initiated': 'maintained/new', 'lowered': 'lowered/withdrawn',
      'withdrawn': 'lowered/withdrawn', 'mixed': 'mixed', 'none': 'none'}
SM = {'all_up': 'up', 'mostly_up': 'up', 'mixed': 'mixed', 'mostly_down': 'down', 'all_down': 'down', 'not_reported': 'nr'}


def load(tag):
    d = {}
    for f in glob.glob(f'{DATA}/llm/{tag}/*.jsonl.gz'):
        for r in read_jsonl_gz(f):
            d[r['accn']] = r
    return d


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    cats = sorted(set(a) | set(b))
    po = float(np.mean(a == b))
    pe = sum(float(np.mean(a == c)) * float(np.mean(b == c)) for c in cats)
    return po, (po - pe) / (1 - pe) if pe < 1 else float('nan')


def norm(s):
    return re.sub(r'\s+', ' ', re.sub(r'[‘’“”"\']', '', s or '')).strip().lower()


def money_strings(v):
    out = set()
    for div, unit in ((1e6, 'million'), (1e9, 'billion')):
        x = v / div
        for nd in (0, 1, 2, 3):
            s = f'{x:,.{nd}f}'
            if float(s.replace(',', '')) >= 1:
                out.add(s)
    return out


def main():
    P, S, G = load('qwen3_32b'), load('mistral24b'), load('gemma3_12b')
    res = {}
    res['n_primary'] = len(P); res['n_primary_valid'] = sum(1 for r in P.values() if r.get('out'))
    ag = {}
    for nm, Q in (('qwen3_32b~mistral24b', S), ('qwen3_32b~gemma3_12b', G), ('mistral24b~gemma3_12b', None)):
        A_, B_ = (P, Q) if Q is not None else (S, G)
        keys = [k for k in A_ if k in B_ and A_[k].get('out') and B_[k].get('out')]
        ag[nm] = {'n': len(keys)}
        for field, mp in (('guidance', GM), ('segments', SM), ('tone', None)):
            a = [mp[A_[k]['out'][field]] if mp else A_[k]['out'][field] for k in keys]
            b = [mp[B_[k]['out'][field]] if mp else B_[k]['out'][field] for k in keys]
            if not keys:
                continue
            po, kp = kappa(a, b)
            ag[nm][field] = {'agreement': po, 'kappa': kp, 'dist_a': pd.Series(a).value_counts(normalize=True).round(4).to_dict(),
                             'dist_b': pd.Series(b).value_counts(normalize=True).round(4).to_dict()}
    # primary vs. majority of the two other-family extractors (where they agree with each other)
    keys = [k for k in P if k in S and k in G and P[k].get('out') and S[k].get('out') and G[k].get('out')]
    for field, mp in (('guidance', GM), ('segments', SM), ('tone', None)):
        f_ = (lambda o: mp[o[field]]) if mp else (lambda o: o[field])
        ks = [k for k in keys if f_(S[k]['out']) == f_(G[k]['out'])]
        ag.setdefault('primary_vs_agreeing_others', {})[field] = {
            'n': len(ks), 'accuracy': float(np.mean([f_(P[k]['out']) == f_(S[k]['out']) for k in ks])) if ks else None}
    res['agreement'] = ag
    S = {**S}
    # quote grounding + lexical consistency (needs masked inputs)
    U = pd.read_csv(f'{RAW}/universe.csv')
    names = U.groupby('cik').company_name.apply(list).to_dict(); tick = U.groupby('cik').ticker.apply(list).to_dict()
    ground, lex = [], []
    LEX = {'raised': r'rais|increas|higher|lift|boost|improv|upgrad|up ', 'lowered': r'lower|reduc|decreas|cut|below|revis|withdr|suspend',
           'maintained': r'reaffirm|reiterat|maintain|confirm|unchang|consistent|remain|continu|expect',
           'initiated': r'expect|outlook|guidance|anticipat|project|forecast|provid|initiat|introduc'}
    texts = {}
    for f in glob.glob(f'{RAW}/ex99/*.jsonl.gz'):
        for r in read_jsonl_gz(f):
            if r['accn'] in P and r.get('text'):
                cik = r['cik']
                texts[r['accn']] = mask_text(select_snippet(r['text']), names.get(cik, []),
                                             [str(t).split('~')[0] for t in tick.get(cik, [])], int(r['filingDate'][:4]))
    for k, r in P.items():
        o = r.get('out')
        if not o or k not in texts:
            continue
        g = o['guidance']
        q = norm(o.get('guidance_quote', ''))
        if g != 'none' and q:
            ground.append(q[:200] in norm(texts[k]))
            if g in LEX:
                lex.append(bool(re.search(LEX[g], q)))
    res['quote_grounding_rate'] = float(np.mean(ground)) if ground else None
    res['quote_lexical_consistency'] = float(np.mean(lex)) if lex else None
    res['n_quotes'] = len(ground)
    # market validity
    rec = pd.read_parquet(f'{DATA}/records_base.parquet')
    rec['g'] = rec.accn_8k.map(lambda a: GM.get(P[a]['out']['guidance']) if a in P and P[a].get('out') else None)
    mv = rec.groupby('g').agg(n=('car_1', 'size'), car1_mean=('car_1', 'mean'), car1_se=('car_1', lambda x: x.std() / np.sqrt(len(x))),
                              car5_mean=('car_5', 'mean'))
    res['market_validity_guidance'] = mv.round(5).to_dict(orient='index')
    rec['t'] = rec.accn_8k.map(lambda a: P[a]['out']['tone'] if a in P and P[a].get('out') else None)
    res['market_validity_tone'] = rec.groupby('t').car_1.agg(['size', 'mean']).round(5).to_dict(orient='index')
    # audit sample
    random.seed(0)
    aud = []
    for lab in ['raised', 'maintained', 'initiated', 'lowered', 'withdrawn', 'mixed', 'none']:
        ks = [k for k in P if P[k].get('out') and P[k]['out']['guidance'] == lab and k in texts]
        for k in random.sample(ks, min(20, len(ks))):
            aud.append({'accn': k, 'primary': P[k]['out'], 'mistral24b': S.get(k, {}).get('out'),
                        'gemma3_12b': G.get(k, {}).get('out'), 'masked_input': texts[k]})
    with open(f'{VOI}/llm_audit_sample.jsonl', 'w') as f:
        for a in aud:
            f.write(json.dumps(a) + '\n')
    res['audit_sample_n'] = len(aud)
    # (b) release-time proxy validation
    raw = {}
    for f in glob.glob(f'{RAW}/ex99/*.jsonl.gz'):
        for r in read_jsonl_gz(f):
            if r.get('text'):
                raw[r['accn']] = r['text']
    hit_rev, hit_eps, yrs = [], [], []
    for _, r in rec.iterrows():
        t = raw.get(r.accn_8k)
        if t is None:
            continue
        tt = t.replace('$', ' ')
        if pd.notna(r.rev_level) and r.rev_level > 0:
            hit_rev.append(any(s in tt for s in money_strings(r.rev_level)))
        else:
            hit_rev.append(None)
        if pd.notna(r.eps_level):
            e = abs(r.eps_level)
            hit_eps.append(any(s in tt for s in {f'{e:.2f}', f'({e:.2f})'}))
        else:
            hit_eps.append(None)
        yrs.append(r.day0[:4])
    df = pd.DataFrame({'y': yrs, 'rev': hit_rev, 'eps': hit_eps})
    res['proxy_revenue_in_release'] = float(df.rev.dropna().astype(float).mean())
    res['proxy_eps_in_release'] = float(df.eps.dropna().astype(float).mean())
    res['proxy_by_year'] = df.groupby('y').agg(lambda x: float(x.dropna().astype(float).mean())).round(4).to_dict(orient='index')
    res['proxy_n'] = len(df)
    json.dump(res, open(f'{VOI}/validation.json', 'w'), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float)[:4000])


if __name__ == '__main__':
    main()
