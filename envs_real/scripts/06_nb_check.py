"""How good is the naive-Bayes (class-conditional independence) approximation used for multi-step quantities?

For test scenario rows, compare after observing TWO offered items (actual values):
  NB posterior  = envs_synth Model belief with both observations (what DP / policy metrics use)
  REF posterior = reference model p(Y | E, e1, e2) (Platt-calibrated LightGBM, conditioning jointly)
and the second-step VOI ranking under both. Output voi/nb_check.json.
"""
import os, sys, json, glob
import numpy as np, pandas as pd
import lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location('be', os.path.join(os.path.dirname(os.path.abspath(__file__)), '04_build_env.py'))
be = importlib.util.module_from_spec(spec); spec.loader.exec_module(be)
from common import DATA, VOI
from core import Model


def main(n_per=400):
    items = pd.read_parquet(f'{DATA}/items.parquet').set_index('rid')
    keys = [c[2:] for c in items.columns if c.startswith('b_')]
    out = {}
    for t in be.TARGETS:
        f = f'{DATA}/{t}_test.jsonl.gz'
        mf = f'{DATA}/ref_models/{t}.txt'
        if not (os.path.exists(f) and os.path.exists(mf)):
            continue
        bst = lgb.Booster(model_file=mf)
        ab = json.load(open(f'{DATA}/ref_models/{t}.platt.json'))
        feats = bst.feature_name()
        import gzip
        rows = [json.loads(l) for i, l in zip(range(n_per), gzip.open(f, 'rt'))]
        d_nb, d_ref, agree = [], [], []
        for r in rows:
            M = Model(r['model'])
            labs = r['label_order']
            nonpl = [c for c in r['checks'] if not c['is_placebo']]
            if len(nonpl) < 3:
                continue
            c1, c2 = nonpl[0], nonpl[1]
            i1, i2 = M.lab2i[c1['label']], M.lab2i[c2['label']]
            b = M.belief([(i1, c1['actual_bin']), (i2, c2['actual_bin'])])[1]
            rid = r['_meta_do_not_show']['rid']
            x = items.loc[[rid], feats].copy()
            for k in keys:
                if f'b_{k}' in x:
                    x[f'b_{k}'] = np.nan
            for o in r['observed']:
                x[f'b_{o["item"]}'] = o['bin']
            x[f'b_{c1["item"]}'] = c1['actual_bin']; x[f'b_{c2["item"]}'] = c2['actual_bin']
            z = be.logit(bst.predict(x))
            pr = float(be.sigmoid(ab['a'] * z + ab['b'])[0])
            d_nb.append(b); d_ref.append(pr)
            thr = r['g'] / (r['g'] + r['l'])
            agree.append((b < thr) == (pr < thr))
        d_nb, d_ref = np.array(d_nb), np.array(d_ref)
        out[t] = {'n': int(len(d_nb)), 'mean_abs_diff_posterior': float(np.mean(np.abs(d_nb - d_ref))),
                  'p90_abs_diff': float(np.quantile(np.abs(d_nb - d_ref), 0.9)), 'corr': float(np.corrcoef(d_nb, d_ref)[0, 1]),
                  'same_decision_rate': float(np.mean(agree))}
        print(t, out[t], flush=True)
    json.dump(out, open(f'{VOI}/nb_check.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
