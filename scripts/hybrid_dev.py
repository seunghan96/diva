"""Dev-only evaluation of DIVA+direct arbitration: follow DIVA unless DIVA stops and the direct policy's top non-stop
choice has probability >= tau (then acquire it). Reports ratio to never-ask per cell for several tau."""
import sys,os,glob,collections,numpy as np; sys.path.insert(0,os.path.dirname(__file__))
from evalx import read,score,summarize,choose_max
from eval_real import fio_real
from eval_tab import diva_scores
K='dp_regret'; TAUS=[1.01,0.9,0.8,0.7,0.6,0.5]
R=collections.defaultdict(list)
cells=[]
for f in glob.glob('outputs/real_dev/fr_*.jsonl.gz'):
    b=os.path.basename(f)[3:-9]; tg=next((t for t in ['ret_1','ret_5','vol_5','vol_20','ret1'] if b.endswith(t)),None)
    if tg is None: continue
    tag=b[:-len(tg)-1]; t2='ret_1' if tg=='ret1' else tg
    cells.append(('sec',t2,tag,f'envs_real/data/{t2}_dev.jsonl.gz',f,f'outputs/real_dev/dir_{tag}_{tg}.jsonl.gz'))
for f in glob.glob('outputs/tab_dev/fk_*.jsonl.gz'):
    tag,ds=os.path.basename(f)[3:-9].rsplit('_',1)
    cells.append(('tab',ds,tag,f'envs_tab/data/{ds}_dev.jsonl.gz',f,f'outputs/tab_dev/dir_{tag}_{ds}.jsonl.gz'))
for kind,e,tag,ep,fp,dp in cells:
    env={s['id']:s for s in read(ep)}; fr={r['id']:r for r in read(fp)}; dr={r['id']:r for r in read(dp)}
    ids=[i for i in env if i in fr and i in dr and dr[i].get('dist')]
    if len(ids)<100: continue
    base=summarize([score(env[i],'S') for i in ids])[K]
    row={}
    for tau in TAUS:
        rows=[]
        for i in ids:
            sc=env[i]; f_=fio_real(sc,fr[i],lam=0.75) if kind=='sec' else diva_scores(sc,fr[i],'cond','greedy','elicit',0.75)
            ch=choose_max(f_['net']); d=dr[i]['dist']; nd={k:v for k,v in d.items() if k!='S'}
            if ch=='S' and nd:
                k=max(nd,key=nd.get)
                if nd[k]>=tau: ch=k
            rows.append(score(sc,ch))
        row[tau]=summarize(rows)[K]/base
    rd=summarize([score(env[i],max(dr[i]['dist'],key=dr[i]['dist'].get)) for i in ids])[K]/base
    print(kind,e,tag,len(ids),'direct %.2f'%rd,' '.join(f'{t}:{v:.2f}' for t,v in row.items()))
    for t,v in row.items(): R[(kind if not e.startswith('vol') else 'vol',t)].append(v)
    R[(kind if not e.startswith('vol') else 'vol','direct')].append(rd)
for k,v in sorted(R.items(),key=str): print(k, '%.3f'%np.mean(v), len(v))
