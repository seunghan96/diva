import sys,os,glob,math,collections,numpy as np; sys.path.insert(0,os.path.dirname(__file__))
from evalx import read,score,summarize,choose_max,clip
from eval_tab import diva_scores
from dcore import voi_exact
K='dp_regret'; SIG=[1.0,1.5,2.0,3.0]; GH=[(-math.sqrt(3),1/6),(0.0,2/3),(math.sqrt(3),1/6)]
def lg(x): x=clip(x); return math.log(x/(1-x))
def sg_(x): return 1/(1+math.exp(-x))
R=collections.defaultdict(list)
for f in glob.glob('outputs/tab_dev/fk_*.jsonl.gz'):
    tag,ds=os.path.basename(f)[3:-9].rsplit('_',1)
    env={s['id']:s for s in read(f'envs_tab/data/{ds}_dev.jsonl.gz')}; fr={r['id']:r for r in read(f)}
    ids=[i for i in env if i in fr]
    if len(ids)<100: continue
    base=summarize([score(env[i],'S') for i in ids])[K]; row={}
    F={i:diva_scores(env[i],fr[i],'cond','greedy','elicit',0.75) for i in ids}
    for s_ in SIG:
        rows=[]
        for i in ids:
            sc=env[i]; f_=F[i]; g,l=float(sc['g']),float(sc['l']); nets={}
            for c in sc['checks']:
                s1,s0,cost=f_['par'][c['label']]; v=0.0
                for z,w in GH:
                    d=s_*z/2; v+=w*voi_exact(f_['b'],clip(sg_(lg(s1)+d)),clip(sg_(lg(s0)-d)),g,l)
                nets[c['label']]=v-cost
            rows.append(score(sc,choose_max(nets)))
        row[s_]=summarize(rows)[K]/base
    print(tag,ds,len(ids),' '.join(f'{k}:{v:.2f}' for k,v in row.items()),flush=True)
    for k,v in row.items(): R[k].append(v)
for k,v in sorted(R.items()): print(('tab',k),round(float(np.mean(v)),3),len(v))
