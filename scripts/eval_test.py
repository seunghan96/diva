"""Evaluate FIO-VOI variants + fac-EIG(+cost-aware tuned on dev) + references on a scenario file given factored outputs.
usage: eval_test.py ENV FAC_OUT [--eig_tau X --eig_lam Y]"""
import sys,os,json,gzip,argparse,collections,numpy as np; sys.path.insert(0,os.path.dirname(__file__))
from evalx import *
from dcore import dp_value
def run(env_p, fac_p, eig_tau=0.05, eig_lam=0.0, label=''):
    env={s['id']:s for s in read(env_p)}; fac={r['id']:r for r in read(fac_p)}
    res=collections.defaultdict(list); rho=collections.defaultdict(list); fbs=0
    for i,sc in env.items():
        if i not in fac: continue
        fs=fac_scores(sc,fac[i]); fbs+=fs['fallback']>0
        ch={'fiovoi':choose_max(fs['net']),
            'fac_eig':choose_max(fs['eig'],eig_tau),
            'fac_eig_cost':choose_max({k:fs['eig'][k]-eig_lam*fs['par'][k][2] for k in fs['eig']},eig_tau),
            'always_stop':'S','oracle_myopic':(sc['exact']['gold_choice'])}
        keys=list(fs['par'])
        if len(keys)<=8:
            v,a=dp_value(fs['b'],[fs['par'][k] for k in keys],float(sc['g']),float(sc['l'])); ch['fiovoi_dp']='S' if a=='S' else keys[a]
        else: ch['fiovoi_dp']=ch['fiovoi']
        for m,c in ch.items():
            res[m].append(score(sc,c))
        for m,s in (('fiovoi',fs['net']),('fac_eig',fs['eig'])):
            rho[m].append(spearman([s[k] for k in keys],[sc['exact']['net'][k] for k in keys]))
    out={}
    for m,rows in res.items():
        s=summarize(rows); s['rho']=float(np.nanmean(rho[m])) if rho[m] else float('nan'); out[m]=s
    out['_meta']={'n_env':len(env),'n_fac':len(fac),'fallback_scen':fbs}
    return out
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('env'); ap.add_argument('fac'); ap.add_argument('--eig_tau',type=float,default=0.05); ap.add_argument('--eig_lam',type=float,default=0.0)
    a=ap.parse_args(); o=run(a.env,a.fac,a.eig_tau,a.eig_lam)
    print(o.pop('_meta'))
    for m,s in o.items(): print(f"{m:13s} n={s['n']} regret={s['norm_regret']:.4f} dpreg={s.get('dp_regret',float('nan')):.4f} capt={s.get('captured',float('nan')):.2f} top1={s['top1']:.2f} dptop1={s.get('dp_top1',float('nan')):.2f} over={s['over_acq']:.2f} under={s['under_acq']:.2f} trap={s.get('eig_trap',float('nan')):.2f} plac={s['placebo']:.3f} rho={s['rho']:.2f}")
