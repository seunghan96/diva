"""Evaluation API for the real market-grounded slice; mirrors ../envs_synth/eval_api.py.

Rows (data/{target}_{split}.jsonl.gz) use the envs_synth binary-track schema, with track = "real":
  model    : prior = [1-p0, p0] from the calibrated reference model given the observed set E (+ sector);
             items[i].lik[y][v] = p(v|E) p~(y|E,v) / p(y|E) over the item's value bins (3 terciles, or 3-5 categories),
             where p~ is the reference model's p(Y|E,e=v) shifted by one log-odds constant so the preposterior is coherent
             (sum_v p(v|E) p~(Y|E,v) = p(Y|E)); first-step VOI / EIG / net value are therefore exact w.r.t. the reference model.
             Multi-step quantities (DP, policy regret) treat items as conditionally independent given Y and E (naive Bayes):
             an approximation that is disclosed and should be reported as such.
  episodes : ONE episode = what actually happened: y = realized outcome, outcomes = the realized value bin of every item.
  checks[i]: item, name, ask (what the request reports), cost, is_placebo, bins [{bin, value, level, text}] (imagination
             support: tercile-median values verbalized), p_bins = p(v|E), p_y_given_bin, actual_bin, actual_text.
  exact    : as envs_synth (posterior, voi, eig, eig_dec, net, q, gold_choice, gold_ties, dp_*, stratum) + all_items
             (VOI/EIG at E for every available item, not only the six offered).
  realized : y, continuous abnormal returns car[h] and log volatility ratios lvr[h] for h in 1, 5, 20, 60.
  _meta_do_not_show: cik / ticker / period / accession (for analysis joins only; never put into a prompt).

Interfaces (same names and semantics as envs_synth):
  evaluate(row, requests, decision, episode=0) -> dict          (exact first-step + NB-approximate policy metrics)
  rank_metrics(row, scores) -> dict
  aggregate(results, by=("domain", "family", "stratum")) -> dict
  Env(row)  .prompt() / .request(label) -> ACTUAL verbalized value / .decide(action) -> evaluate(...)
  realized_metrics(row, requests, decision) -> realized utility of the decision given the actual outcome (market-grounded),
            minus request costs; and the realized log-score of the final belief.
  load(target, split) -> list of rows
  CLI: python eval_api.py --data data/ret_5_test.jsonl.gz --preds preds.jsonl.gz [--out summary.json]
"""
import os, sys, json, gzip, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SYN = os.path.join(os.path.dirname(HERE), 'envs_synth')
sys.path.insert(0, SYN)
import eval_api as _syn                      # noqa: E402
from eval_api import evaluate, rank_metrics, aggregate, read_jsonl_gz, get_model   # noqa: E402,F401

TARGETS = [f'{k}_{h}' for k in ('ret', 'vol') for h in (1, 5, 20, 60)]


def load(target, split='test'):
    return read_jsonl_gz(os.path.join(HERE, 'data', f'{target}_{split}.jsonl.gz'))


class Env(_syn.Env):
    """Reveals the item's ACTUAL value (verbalized), not just its bin."""

    def observation_text(self, lab):
        c = [c for c in self.row['checks'] if c['label'] == lab][0]
        return f"Result of ({lab}) {c['name']}: {c['actual_text']}."


def realized_metrics(row, requests, decision):
    M = get_model(row)
    labs = row['label_order']
    obs, cost, seen = [], 0.0, set()
    ep = row['episodes'][0]
    for raw in requests:
        lab = str(raw).strip().strip('()').upper()
        if lab not in labs or lab in seen:
            continue
        seen.add(lab)
        i = M.lab2i[lab]
        cost += float(M.cost[i])
        obs.append((i, int(ep['outcomes'][lab])))
    b = M.belief(obs)
    y = int(row['realized']['y'])
    ai = _syn._norm_action(row, decision)
    u = float(row['model']['utility'][ai][y]) if ai is not None else float(np.min(np.asarray(row['model']['utility'])[:, y]))
    return {'realized_utility': u, 'realized_cau': u - cost, 'cost': cost, 'n_requests': len(seen),
            'final_belief_adverse': float(b[1]), 'realized_logscore': float(np.log(max(b[y], 1e-12)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', nargs='+', required=True)
    ap.add_argument('--preds', required=True)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    rows = {}
    for p in a.data:
        for r in read_jsonl_gz(p):
            rows[r['id']] = r
    res, ranks = [], []
    for p in read_jsonl_gz(a.preds):
        r = rows.get(p['id'])
        if r is None:
            continue
        e = evaluate(r, p.get('requests', []), p.get('decision'), 0)
        e.update(realized_metrics(r, p.get('requests', []), p.get('decision')))
        e['target'] = r['target']
        if p.get('scores'):
            rm = rank_metrics(r, p['scores'])
            e.update({'rank_' + k: v for k, v in rm.items()})
            ranks.append(rm)
        res.append(e)
    agg = aggregate(res, by=('target', 'stratum'))
    agg['realized'] = {k: float(np.mean([x[k] for x in res])) for k in ('realized_utility', 'realized_cau', 'realized_logscore', 'n_requests')}
    if ranks:
        agg['ranking'] = {k: float(np.nanmean([x[k] for x in ranks if k in x])) for k in ranks[0]}
    s = json.dumps(agg, indent=1, default=float)
    if a.out:
        open(a.out, 'w').write(s)
    else:
        print(s)


if __name__ == '__main__':
    main()
