import sys,os,glob,collections,numpy as np; sys.path.insert(0,os.path.dirname(__file__))
from evalx import read,score,summarize,choose_max
from eval_real import fio_real
res=collections.defaultdict(list)
for f in glob.glob('outputs/real_dev_joint/fr_*.jsonl.gz'):
    b=os.path.basename(f)[3:-9]; tg=next(t for t in ['ret_1','ret_5','vol_5','vol_20'] if b.endswith(t)); tag=b[:-len(tg)-1]
    env={s['id']:s for s in read(f'envs_real/data/{tg}_dev.jsonl.gz')}; J={r['id']:r for r in read(f)}
    sp=f'outputs/real_dev/fr_{tag}_{tg}.jsonl.gz' if tg!='ret_1' else f'outputs/real_dev/fr_{tag}_ret1.jsonl.gz'
    S={r['id']:r for r in read(sp)}; D={r['id']:r for r in read(f'outputs/real_dev/dir_{tag}_{tg}.jsonl.gz' if tg!='ret_1' else f'outputs/real_dev/dir_{tag}_ret1.jsonl.gz')}
    ids=[i for i in env if i in J and i in S]
    if len(ids)<100: continue
    base=summarize([score(env[i],'S') for i in ids])['dp_regret']
    row={}
    import math
    def avgfr(a,b):
        out={'prior':a['prior'],'obs':a['obs'],'chk':{}}
        for k in a['chk']:
            out['chk'][k]=[]
            for x,y in zip(a['chk'][k],b['chk'].get(k,a['chk'][k])):
                xs=[v for v in (x[0] if isinstance(x,list) else x, y[0] if isinstance(y,list) else y) if v is not None]
                if not xs: out['chk'][k].append([None]); continue
                lo=sum(math.log(min(max(v,.01),.99)/(1-min(max(v,.01),.99))) for v in xs)/len(xs); out['chk'][k].append([1/(1+math.exp(-lo))])
        return out
    for lam in [1.0,0.75,0.5]:
        row[f'avg{lam}']=summarize([score(env[i],choose_max(fio_real(env[i],avgfr(J[i],S[i]),lam=lam)['net'])) for i in ids])['dp_regret']/base
        row[f'joint{lam}']=summarize([score(env[i],choose_max(fio_real(env[i],J[i],lam=lam)['net'])) for i in ids])['dp_regret']/base
        row[f'single{lam}']=summarize([score(env[i],choose_max(fio_real(env[i],S[i],lam=lam)['net'])) for i in ids])['dp_regret']/base
    di=[i for i in ids if i in D]
    row['direct']=summarize([score(env[i],max(D[i]['dist'],key=D[i]['dist'].get)) for i in di])['dp_regret']/base if di else float('nan')
    print(tag,tg,len(ids),' '.join(f'{k}:{v:.2f}' for k,v in row.items()))
    for k,v in row.items(): res[k].append(v)
print('MEAN',' '.join(f'{k}:{np.nanmean(v):.3f}' for k,v in res.items()))
