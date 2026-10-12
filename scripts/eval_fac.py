import sys,os,json,gzip,collections,numpy as np; sys.path.insert(0,os.path.dirname(__file__))
from metrics import score_choice, summarize, spearman
from dcore import posterior, voi_exact, eig_exact, dp_value
def fac_choices(sc, fr, prior_mode='elicit'):
    if fr['prior'] is None or any(v is None for o in fr['obs'] for v in o) or any(v is None for v in sum(fr['chk'].values(),[])): return None
    clip=lambda x:min(max(x,0.01),0.99)
    b=posterior(clip(fr['prior']), [(clip(o[0]),clip(o[1]),ob['value']) for o,ob in zip(fr['obs'],sc['observed'])])
    g,l=sc['g'],sc['l']; voi={};eig={}
    for c in sc['checks']:
        s1,s0=map(clip,fr['chk'][c['label']]); voi[c['label']]=voi_exact(b,s1,s0,g,l)-c['cost']; eig[c['label']]=eig_exact(b,s1,s0)
    bv=max(voi,key=voi.get); be=max(eig,key=eig.get)
    dpv,dpa=dp_value(b,[(clip(fr['chk'][c['label']][0]),clip(fr['chk'][c['label']][1]),c['cost']) for c in sc['checks']],g,l)
    return {'fiovoi':(bv if voi[bv]>0 else 'S',voi),'fiovoi_dp':('S' if dpa=='S' else sc['checks'][dpa]['label'],voi),'fac_eig':(be if eig[be]>0.05 else 'S',eig),'fac_eig_always':(be,eig)}
if __name__=='__main__':
  env={json.loads(l)['id']:json.loads(l) for l in gzip.open(sys.argv[1],'rt')}
  for f in sys.argv[2:]:
      res=collections.defaultdict(list); rf=collections.defaultdict(list)
      for l in gzip.open(f,'rt'):
          fr=json.loads(l); sc=env[fr['id']]
          d=fac_choices(sc,fr)
          if d is None: continue
          d['oracle']=(sc['exact']['gold_choice'],sc['exact']['net']); d['always_stop']=('S',{c['label']:0 for c in sc['checks']})
          for m,(ch,s) in d.items():
              r=score_choice(sc,ch); r['stratum']=sc['stratum']; res[m].append(r); rf[m].append(spearman([s[c['label']] for c in sc['checks']],[sc['exact']['net'][c['label']] for c in sc['checks']]))
      for m in res:
          s=summarize(res[m]); print(f"{os.path.basename(f)[:18]:18s} {m:15s} n={s['n']} regret={s['norm_regret']:.3f} top1={s['top1']:.2f} over={s['over_acq']:.2f} under={s['under_acq']:.2f} trapS1={s['eig_trap_S1']:.2f} placebo={s['placebo']:.2f} rho={np.nanmean(rf[m]):.2f}")
