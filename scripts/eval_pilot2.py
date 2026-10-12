import sys,os,json,collections,numpy as np; sys.path.insert(0,os.path.dirname(__file__))
from evalx import *
env={s['id']:s for s in read('data/pilot2.jsonl.gz')}
for t in sys.argv[1].split(','):
    fac={r['id']:r for r in read(f'outputs/pilot2/facnum_{t}.jsonl.gz')}; dr={r['id']:r for r in read(f'outputs/pilot2/direct_{t}.jsonl.gz')}
    R=collections.defaultdict(list); mass=[]
    for i,sc in env.items():
        if i in fac:
            fs=fac_scores(sc,fac[i]); R['fiovoi'].append(score(sc,choose_max(fs['net']))); R['fac_eig'].append(score(sc,choose_max(fs['eig'],0.05)))
        if i in dr and dr[i]['dist']:
            d=dr[i]['dist']; R['direct'].append(score(sc,max(d,key=d.get))); mass+= dr[i]['mass']
        R['always_stop'].append(score(sc,'S'))
    for m,rows in R.items():
        s=summarize(rows); print(f"{t:4s} {m:11s} n={s['n']} regret={s['norm_regret']:.4f} dpreg={s.get('dp_regret',float('nan')):.4f} top1={s['top1']:.2f} dptop1={s.get('dp_top1',float('nan')):.2f} over={s['over_acq']:.2f} under={s['under_acq']:.2f} trap={s.get('eig_trap',float('nan')):.2f}")
    print(t,'direct letter mass',np.mean(mass) if mass else None)
