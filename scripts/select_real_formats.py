import sys,os,glob,collections,numpy as np; sys.path.insert(0,os.path.dirname(__file__))
from evalx import read,score,summarize,choose_max
from eval_real import fio_real
K='dp_regret'; R=collections.defaultdict(list)
for f in glob.glob('outputs/real_dev_ratio/fr_*.jsonl.gz'):
    b=os.path.basename(f)[3:-9]; tg=next(t for t in ['ret_1','ret_5','vol_5','vol_20'] if b.endswith(t)); tag=b[:-len(tg)-1]
    env={s['id']:s for s in read(f'envs_real/data/{tg}_dev.jsonl.gz')}
    src={'ratio':{r['id']:r for r in read(f)},'joint':{r['id']:r for r in read(f'outputs/real_dev_joint/fr_{tag}_{tg}.jsonl.gz')},
         'single':{r['id']:r for r in read(f'outputs/real_dev/fr_{tag}_{"ret1" if tg=="ret_1" else tg}.jsonl.gz')}}
    dr={r['id']:r for r in read(f'outputs/real_dev/dir_{tag}_{"ret1" if tg=="ret_1" else tg}.jsonl.gz')}
    ids=[i for i in env if all(i in s for s in src.values())]
    if len(ids)<150: continue
    base=summarize([score(env[i],'S') for i in ids])[K]; row={}
    for name,s in src.items():
        for lam in ([1.0,0.75,0.5] if name!='single' else [0.75]):
            row[f'{name}{lam}']=summarize([score(env[i],choose_max(fio_real(env[i],s[i],lam=lam)['net'])) for i in ids])[K]/base
    import math
    def comb(a,b,pri_lam=1.0):
        # average, in log-odds, the joint per-bin risks and the ratio-derived per-bin risks
        out={'prior':a['prior'],'obs':a['obs'],'chk':{}}
        for k,v in a['chk'].items():
            rv=b['chk'].get(k)
            pri=a['prior'][0] if isinstance(a['prior'],list) else a['prior']
            pri=min(max(pri or 0.5,.01),.99)
            if rv and isinstance(rv[0],list) and rv[0] and rv[0][0]=='RATIO':
                rho=float(rv[0][1]); lo=min(.99,max(.01,pri*2/(1+rho))); hi=min(.99,max(.01,rho*lo)); rr=[lo]+[pri]*(len(v)-2)+[hi]
            else:
                rr=[pri]*len(v)
            new=[]
            for x,y in zip(v,rr):
                x=x[0] if isinstance(x,list) else x
                if x is None: new.append([y]); continue
                x=min(max(x,.01),.99); lo_=(math.log(x/(1-x))+math.log(y/(1-y)))/2; new.append([1/(1+math.exp(-lo_))])
            out['chk'][k]=new
        return out
    for lam in [1.0,0.75]:
        row[f'comb{lam}']=summarize([score(env[i],choose_max(fio_real(env[i],comb(src['joint'][i],src['ratio'][i]),lam=lam)['net'])) for i in ids])[K]/base
    di=[i for i in ids if i in dr and dr[i].get('dist')]
    row['direct']=summarize([score(env[i],max(dr[i]['dist'],key=dr[i]['dist'].get)) for i in di])[K]/base
    grp='vol' if tg.startswith('vol') else 'ret'
    print(tag,tg,len(ids),' '.join(f'{k}:{v:.2f}' for k,v in row.items()))
    for k,v in row.items(): R[(grp,k)].append(v); R[('all',k)].append(v)
for k,v in sorted(R.items()): print(k,'%.3f'%np.mean(v),len(v))
