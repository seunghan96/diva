import sys,os,collections,numpy as np; sys.path.insert(0,os.path.dirname(__file__))
from evalx import read,score,summarize,choose_max
from eval_ms import build, es
tag=sys.argv[1]
for dom in ['finance','credit','clinical','legal']:
    env={s['id']:s for s in read(f'envs_synth/data/ms/{dom}_dev.jsonl.gz')}; fr={r['id']:r for r in read(f'outputs/ms_dev/ms_{tag}_{dom}.jsonl.gz')}
    R=collections.defaultdict(list)
    for i,sc in list(env.items())[:150]:
        if i not in fr: continue
        M,_=build(sc,fr[i]); labs=[c['label'] for c in sc['checks']]
        T=es.Model(sc['model'])
        for name,mod in [('elicited',M),('true_util',es.Model(dict(prior=list(M.prior),utility=sc['model']['utility'],items=[dict(label=l,lik=M.L[j].tolist(),cost=float(M.cost[j])) for j,l in enumerate(labs)]))),
                         ('true_beliefs',es.Model(dict(prior=sc['model']['prior'],utility=M.U.tolist(),items=sc['model']['items']))),('true_all',T)]:
            net={l:mod.item_stats([],j)[0]-mod.cost[j] for j,l in enumerate(labs)}
            R[name].append(score(sc,choose_max(net)))
    print(dom,' '.join(f"{k}:{summarize(v)['dp_regret']:.4f}/{summarize(v)['norm_regret']:.4f}" for k,v in R.items()),flush=True)
