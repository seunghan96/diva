import sys, json, math; sys.path.insert(0,'scripts')
from evalx import read, score, choose_max, scale, clip, summarize
from eval_tab import diva_scores, diva_u, agg, temper
from dcore import pred, bayes, vstar, voi_exact, eig_exact, eu, dp_value
from episodes_know import spec_dp_first
import run_direct
tag, env, i = sys.argv[1], sys.argv[2], sys.argv[3]
E = read(f'envs_tab/data/{env}_test.jsonl.gz')[:300]
sc = {s['id']: s for s in E}[i]
FR = {r['id']: r for r in read(f'outputs/test_tab/fk_{tag}_{env}.jsonl.gz')}; fr = FR[i]
DR = {r['id']: r for r in read(f'outputs/test_tab/dir_{tag}_{env}.jsonl.gz')}; dr = DR[i]
O = {'tag': tag, 'env': env, 'id': i}
O['text'] = sc['text']; O['direct_prompt'], O['direct_map'] = run_direct.build(sc, 0)
O['direct_dist'] = dr['dist']; O['raw'] = fr
g, l = float(sc['g']), float(sc['l']); O['g'], O['l'] = g, l; O['L'] = scale(sc)
pri = agg(fr['prior'], 'greedy'); O['prior_elicited'] = pri; O['prior_stated'] = sc['prior']
def conv(rec):
    c1, c0, pf = [clip(agg(rec[k], 'greedy')) for k in ('c1', 'c0', 'pf')]
    a1 = c1*pf/(c1*pf+c0*(1-pf)); a0 = (1-c1)*pf/((1-c1)*pf+(1-c0)*(1-pf))
    return dict(c1=c1, c0=c0, pf=pf, a1=clip(a1), a0=clip(a0))
b = clip(pri); steps = []
for rec, ob in zip(fr['obs'], sc['observed']):
    d = conv(rec); s1, s0 = temper((d['a1'], d['a0']), 0.75); nb = bayes(b, s1, s0, int(ob['value']))
    steps.append(dict(name=ob['name'], finding=ob['finding_text'], value=ob['value'], **d, s1=s1, s0=s0, b_before=b, b_after=nb)); b = nb
O['obs_steps'] = steps; O['b'] = b
fs = diva_scores(sc, fr, 'cond', 'greedy', 'elicit', 0.75); assert abs(fs['b']-b) < 1e-12
fu = diva_u(sc, fr, 1.0)
GH = [(-math.sqrt(3), 1/6), (0.0, 2/3), (math.sqrt(3), 1/6)]
lg = lambda x: math.log(clip(x)/(1-clip(x))); sg = lambda x: 1/(1+math.exp(-x))
O['eu_b'] = eu(b, g, l); O['vstar_b'] = vstar(b, g, l); O['threshold'] = g/(g+l)
ex = sc['exact']; chks = []
for c in sc['checks']:
    k = c['label']; d = conv(fr['chk'][k]); s1, s0, cost = fs['par'][k]
    p1 = pred(b, s1, s0); b1 = bayes(b, s1, s0, 1); b0 = bayes(b, s1, s0, 0)
    nodes = [dict(z=z, w=w, s1=clip(sg(lg(s1)+z/2)), s0=clip(sg(lg(s0)-z/2)), voi=voi_exact(b, clip(sg(lg(s1)+z/2)), clip(sg(lg(s0)-z/2)), g, l)) for z, w in GH]
    chks.append(dict(label=k, name=c['name'], finding=c['finding_text'], cost=cost, **{('el_'+x): d[x] for x in d}, s1=s1, s0=s0,
        p1=p1, b1=b1, b0=b0, eu1=eu(b1, g, l), eu0=eu(b0, g, l), v1=vstar(b1, g, l), v0=vstar(b0, g, l), voi=voi_exact(b, s1, s0, g, l), net=fs['net'][k], eig=fs['eig'][k],
        net_u=fu['net'][k], gh=nodes, emp_s1=c['emp_s1'], emp_s0=c['emp_s0'], true_net=ex['net'][k], true_q=ex['q'][k], true_voi=ex['voi'][k]))
O['checks'] = chks
keys = list(fs['par'])
def V(b_, rem):
    vals = {'S': vstar(b_, g, l)}
    for j in rem:
        s1, s0, c = fs['par'][j]; p = pred(b_, s1, s0); r = tuple(x for x in rem if x != j)
        vals[j] = -c + p*V(bayes(b_, s1, s0, 1), r)[0] + (1-p)*V(bayes(b_, s1, s0, 0), r)[0]
    m = max(vals, key=vals.get); return vals[m], m, vals
_, best, top = V(b, tuple(keys)); O['dp_top'] = top
par_ = {k: ([1-a1, a1], [1-a0, a0], c) for k, (a1, a0, c) in fs['par'].items()}
O['dp_first_specdp'] = spec_dp_first(b, par_, g, l)
dk = O['dp_first_specdp']
if dk != 'S':
    s1, s0, c = fs['par'][dk]; r = tuple(x for x in keys if x != dk); sub = {}
    for val in (1, 0):
        bb = bayes(b, s1, s0, val); vv, mm, vals = V(bb, r)
        sub[val] = dict(p=pred(b, s1, s0) if val else 1-pred(b, s1, s0), b=bb, best=mm, value=vv, vals=vals, act=('APPROVE' if eu(bb, g, l)[0] > 0 else 'DECLINE'))
        if mm != 'S':
            t1, t0, cc = fs['par'][mm]; r2 = tuple(x for x in r if x != mm)
            sub[val]['next'] = {v2: dict(b=bayes(bb, t1, t0, v2), best=V(bayes(bb, t1, t0, v2), r2)[1], act=('APPROVE' if eu(bayes(bb, t1, t0, v2), g, l)[0] > 0 else 'DECLINE')) for v2 in (1, 0)}
    O['dp_sub'] = sub
ch = dict(diva=choose_max(fs['net']), diva_u=choose_max(fu['net']), diva_dp=dk, eig=choose_max(fs['eig'], 0.05), direct=max(dr['dist'], key=dr['dist'].get), never='S')
O['choices'] = ch; O['regret'] = {m: score(sc, cc) for m, cc in ch.items()}
O['exact'] = {k: ex.get(k) for k in ('posterior', 'dp_first_action', 'dp_first_ties', 'dp_value', 'value_of_stopping', 'gold_choice', 'q', 'net', 'voi', 'eig')}
O['y_true'] = sc['y_true']
# representativeness: averages on first 300 test rows for this LLM/env
K = json.load(open('results/know_main.json')).get(f'tab:{env}', {}).get(tag, {})
O['s300'] = {m: K.get(f's300:{m}', {}).get('dp_regret') for m in ['direct', 'never', 'diva', 'diva_u', 'diva_dp', 'bed_llm', 'voi_ask', 'cot_need']}
O['s300_stop'] = {m: K.get(f's300:{m}', {}).get('stopped') for m in ['direct', 'never', 'diva', 'diva_u', 'diva_dp']}
dpacq = sum(1 for s in E if s['exact'].get('dp_first_action') not in (None, 'S')); O['dp_acquire_share'] = dpacq/len(E)
json.dump(O, open(sys.argv[4], 'w'), indent=1, default=float)
print('b', b, 'thr', g/(g+l), ch, {m: round(100*r['dp_regret'], 2) for m, r in O['regret'].items()})
for c in chks: print(c['label'], c['name'], 'c1/c0/pf', [round(c[x], 3) for x in ('el_c1', 'el_c0', 'el_pf')], 'a', round(c['el_a1'], 3), round(c['el_a0'], 3), 's', round(c['s1'], 3), round(c['s0'], 3), 'p1', round(c['p1'], 3), 'b1/b0', round(c['b1'], 3), round(c['b0'], 3), 'voi', round(c['voi'], 4), 'net', round(c['net'], 4), 'eig', round(c['eig'], 4), 'netU', round(c['net_u'], 4), 'trueq', round(c['true_q'], 3))
print('dp top', {k: round(v, 4) for k, v in top.items()}); print('sub', json.dumps(O.get('dp_sub'), default=float)[:900])
for s in steps: print(s)
print(dr['dist']); print('s300', O['s300'], O['s300_stop'], 'dp acquire share', O['dp_acquire_share'])
print('exact', O['exact']['dp_first_action'], O['exact']['dp_value'], O['exact']['value_of_stopping'], 'L', O['L'])
