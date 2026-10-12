import sys,os,json,gzip,collections,numpy as np; sys.path.insert(0,os.path.dirname(__file__))
from metrics import derive, score_choice, summarize, spearman
from eval_fac import fac_choices
env={json.loads(l)['id']:json.loads(l) for l in gzip.open(sys.argv[1],'rt')}
D=sys.argv[2]; tags=sys.argv[3].split(',')
def rd(p):
    return [json.loads(l) for l in gzip.open(p,'rt')] if os.path.exists(p) else []
allres={}
for t in tags:
    res=collections.defaultdict(list); rho=collections.defaultdict(list)
    def add(m,sc,ch,s):
        r=score_choice(sc,ch); r['stratum']=sc['stratum']; res[m].append(r)
        if s: rho[m].append(spearman([s.get(c['label'],0) for c in sc['checks']],[sc['exact']['net'][c['label']] for c in sc['checks']]))
    for pr in rd(f'{D}/probe_{t}.jsonl.gz'):
        sc=env[pr['id']]; d=derive(sc,pr)
        for m in ('direct','iovoi'):
            if m in d: add({'iovoi':'raw_iovoi'}.get(m,m),sc,*d[m])
    for name,f in (('fiovoi_bin','facbin'),('fiovoi','facnum')):
        for fr in rd(f'{D}/{f}_{t}.jsonl.gz'):
            sc=env[fr['id']]; d=fac_choices(sc,fr)
            if d is None: continue
            add(name,sc,*d['fiovoi'])
            if f=='facnum': add('fac_eig',sc,*d['fac_eig']); add('fiovoi_dp',sc,*d['fiovoi_dp'])
    for sc in env.values():
        add('always_stop',sc,'S',None)
    allres[t]={m:summarize(v)|{'rho':float(np.nanmean(rho[m])) if rho[m] else float('nan')} for m,v in res.items()}
    for m,s in allres[t].items():
        print(f"{t:5s} {m:12s} n={s['n']:4d} regret={s['norm_regret']:.3f} top1={s['top1']:.2f} over={s['over_acq']:.2f} under={s['under_acq']:.2f} trapS1={s['eig_trap_S1']:.2f} placebo={s['placebo']:.3f} rho={s['rho']:.2f}")
json.dump(allres,open(sys.argv[4],'w'),indent=1) if len(sys.argv)>4 else None
