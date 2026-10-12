import sys,os,glob,collections,numpy as np; sys.path.insert(0,os.path.dirname(__file__))
from evalx import read,fac_scores,choose_max,score,summarize
tag=sys.argv[1]; R=collections.defaultdict(lambda: collections.defaultdict(list))
for d in ['finance','credit','clinical','legal']:
    env={s['id']:s for s in read(f'envs_synth/data/binary/{d}_test.jsonl.gz')[:1000]}
    fac={r['id']:r for r in read(f'outputs/test_bin/facnum_{tag}_{d}.jsonl.gz')}
    for p in glob.glob(f'baselines/outputs/{tag}/*/{d}_test.jsonl.gz'):
        m=p.split('/')[-2]
        for r in read(p):
            i=r['id']
            if i in env and i in fac and r.get('choice') is not None:
                sc=env[i]; fs=fac_scores(sc,fac[i]); R[m]['base'].append(score(sc,r['choice'])); R[m]['diva'].append(score(sc,choose_max(fs['net'])))
for m in sorted(R):
    a=summarize(R[m]['base']); b=summarize(R[m]['diva'])
    print(f"{m:16s} n={a['n']:4d} base dpreg={a['dp_regret']:.4f} myreg={a['norm_regret']:.4f} over={a['over_acq']:.2f} under={a['under_acq']:.2f} | DIVA dpreg={b['dp_regret']:.4f} myreg={b['norm_regret']:.4f}")
