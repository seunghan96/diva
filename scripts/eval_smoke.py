import sys,os,json,gzip,collections; sys.path.insert(0,os.path.dirname(__file__))
from metrics import *
env={json.loads(l)['id']:json.loads(l) for l in gzip.open(sys.argv[1],'rt')}
for f in sys.argv[2:]:
    prs=[json.loads(l) for l in gzip.open(f,'rt')]
    for pmode in ['martingale','given','uniform']:
        res=collections.defaultdict(list); rf=collections.defaultdict(list)
        for pr in prs:
            sc=env[pr['id']]; d=derive(sc,pr,pmode=pmode)
            for m,(ch,scores) in d.items():
                r=score_choice(sc,ch); r['stratum']=sc['stratum']; res[m].append(r)
                rf[m].append(rank_fidelity(sc,scores))
        for m in res:
            if pmode!='martingale' and m=='direct': continue
            s=summarize(res[m]); import numpy as np
            print(f"{os.path.basename(f)[:20]:20s} {pmode:10s} {m:11s} n={s['n']} regret={s['norm_regret']:.3f} top1={s['top1']:.2f} over={s['over_acq']:.2f} under={s['under_acq']:.2f} trapS1={s['eig_trap_S1']:.2f} placebo={s['placebo']:.2f} rho={np.nanmean(rf[m]):.2f}")
