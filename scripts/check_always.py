import sys,os,glob,collections,numpy as np; sys.path.insert(0,os.path.dirname(__file__))
from evalx import read,score,summarize,choose_max
from eval_real import fio_real, fio_real_u
from eval_tab import diva_scores, diva_u
for kind,pat in [('sec','outputs/test_real/fr_*_{t}.jsonl.gz'),('tab','outputs/test_tab/fk_*_{t}.jsonl.gz')]:
    targets=['ret_1','ret_5','vol_5','vol_20'] if kind=='sec' else ['taiwan','german','cdc','heart']
    for t in targets:
        R=collections.defaultdict(list)
        ep=f'envs_real/data/{t}_test.jsonl.gz' if kind=='sec' else f'envs_tab/data/{t}_test.jsonl.gz'
        env={s['id']:s for s in read(ep)}
        for f in glob.glob(pat.format(t=t))[:6]:
            fr={r['id']:r for r in read(f)}
            for i in list(fr)[:400]:
                if i not in env: continue
                sc=env[i]
                fs=fio_real(sc,fr[i],lam=0.75) if kind=='sec' else diva_scores(sc,fr[i],'cond','greedy','elicit',0.75)
                fu=fio_real_u(sc,fr[i],1.0) if kind=='sec' else diva_u(sc,fr[i],1.0)
                top=max(fs['net'],key=fs['net'].get); topu=max(fu['net'],key=fu['net'].get)
                for m,ch in [('diva',choose_max(fs['net'])),('diva_u',choose_max(fu['net'])),('always_top',top),('never','S')]:
                    R[m].append(score(sc,ch))
        print(kind,t,' '.join(f"{m}:dp={summarize(v)['dp_regret']*100:.2f}/my={summarize(v)['norm_regret']*100:.2f}/stop={summarize(v)['stopped']:.2f}" for m,v in R.items()))
