"""Dev grid for knowledge-required DIVA knobs: form x aggregation x temper lam x stop margin tau (fraction of l)."""
import sys, os, json, collections, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from evalx import read, score, summarize
from eval_tab import diva_scores
def grid(env_p, fk_p, forms=("lik","cond","both"), modes=("greedy","mean"), lams=(1.0,0.75,0.5,0.35), taus=(0.0,0.005,0.01,0.02), priors=("elicit","stated")):
    env={s['id']:s for s in read(env_p)}; fk={r['id']:r for r in read(fk_p)}
    ids=[i for i in env if i in fk]; out={}
    for form,mode,lam,pr in itertools.product(forms,modes,lams,priors):
        S=[diva_scores(env[i],fk[i],form,mode,pr,lam) for i in ids]
        for tau in taus:
            rows=[]
            for i,f in zip(ids,S):
                sc=env[i]; k=max(f['net'],key=f['net'].get); ch=k if f['net'][k]>tau*float(sc['l'])+1e-9 else 'S'
                rows.append(score(sc,ch))
            out[(form,mode,lam,pr,tau)]=summarize(rows)
    base=summarize([score(env[i],'S') for i in ids])
    return out, base, len(ids)
if __name__=='__main__':
    for fk in sys.argv[2:]:
        o,b,n=grid(sys.argv[1],fk)
        best=sorted(o.items(),key=lambda kv: kv[1]['norm_regret'])[:6]
        print(os.path.basename(fk), 'n',n,'always_stop',round(b['norm_regret'],4))
        for k,s in best: print('  ',k,round(s['norm_regret'],4),'capt',round(s.get('captured',float('nan')),2),'over',round(s['over_acq'],2),'under',round(s['under_acq'],2))
        d=o[('cond','greedy',1.0,'elicit',0.0)]; print('   default cond/greedy/1/elicit/0:',round(d['norm_regret'],4))


def grid_robust(env_p, fk_p, forms=("lik","cond","both"), qs=(0.5,0.25,0.0), lams=(1.0,0.75,0.5)):
    from eval_tab import diva_robust
    env={s['id']:s for s in read(env_p)}; fk={r['id']:r for r in read(fk_p)}
    ids=[i for i in env if i in fk]; out={}
    for form,q,lam in itertools.product(forms,qs,lams):
        out[(form,q,lam)]=summarize([score(env[i],diva_robust(env[i],fk[i],form,q,lam)[0]) for i in ids])
    return out
