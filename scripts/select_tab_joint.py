import sys,os,glob,random,collections,numpy as np; sys.path.insert(0,os.path.dirname(__file__))
from evalx import read,score,summarize,choose_max
from eval_tab import diva_scores
from dcore import voi_exact
K='dp_regret'
def kfree(sc,f,i):
    labs=[c['label'] for c in sc['checks']]; rs=[]
    for seed in range(5):
        rnd=random.Random(hash((i,seed))&0xffffffff); perm=labs[:]; rnd.shuffle(perm)
        net={l:voi_exact(f['b'],f['par'][p][0],f['par'][p][1],float(sc['g']),float(sc['l']))-f['par'][l][2] for l,p in zip(labs,perm)}
        rs.append(score(sc,choose_max(net))[K])
    return np.mean(rs)
R=collections.defaultdict(list)
for f in glob.glob('outputs/tab_dev_joint/fk_*.jsonl.gz'):
    tag,ds=os.path.basename(f)[3:-9].rsplit('_',1)
    env={s['id']:s for s in read(f'envs_tab/data/{ds}_dev.jsonl.gz')}; J={r['id']:r for r in read(f)}; S={r['id']:r for r in read(f'outputs/tab_dev/fk_{tag}_{ds}.jsonl.gz')}
    ids=[i for i in env if i in J and i in S]
    if len(ids)<100: continue
    base=summarize([score(env[i],'S') for i in ids])[K]; row={}
    for name,src,lam in [('single0.75',S,0.75),('single1',S,1.0),('joint1',J,1.0),('joint0.75',J,0.75),('joint0.5',J,0.5)]:
        fs=[diva_scores(env[i],src[i],'cond','greedy','elicit',lam) for i in ids]
        row[name]=np.mean([score(env[i],choose_max(f_['net']))[K] for i,f_ in zip(ids,fs)])/base
        row[name+'_kf']=np.mean([kfree(env[i],f_,i) for i,f_ in zip(ids,fs)])/base
    print(tag,ds,len(ids),' '.join(f'{k}:{v:.2f}' for k,v in row.items()))
    for k,v in row.items(): R[k].append(v)
print('MEAN',' '.join(f'{k}:{np.mean(v):.3f}' for k,v in R.items()), 'cells',len(R['joint1']))
