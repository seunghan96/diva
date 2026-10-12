"""Exact (non-LLM) reference policies run through eval_api on every stored episode. They calibrate the metrics
and give the EIG-vs-VOI contrast for idealized agents that know the model exactly:
  oracle_dp      follow the DP-optimal policy, decide the Bayes action                  (policy_regret = 0 by construction)
  myopic_voi     acquire argmax VOI - cost while > 0, then Bayes action
  eig_greedy     acquire argmax EIG (about the hidden state) while EIG > 0.05 bits and budget allows, then Bayes action
  eig_per_cost   acquire argmax EIG while EIG > 0.05 bits, but stop when the cheapest such item costs more than
                 0 (i.e. EIG ignoring decision value, but cost-aware tie-breaking: argmax EIG / (cost + eps))
  eigdec_greedy  acquire argmax EIG about the optimal action (Entropy-Search-style) while > 0.05 bits
  always_stop    decide the Bayes action at the start
  acquire_all    acquire every item (in label order) up to the budget, then Bayes action
  random         acquire a uniformly random number (0..budget) of random items, then Bayes action
usage: python ref_policies.py --files data/ms/*_test.jsonl.gz --out results/ref_policies_ms_test.json [--workers 32]
"""
import argparse, gzip, json, os, random, sys
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from core import Model
from eval_api import evaluate, aggregate, get_model

POLICIES = ["oracle_dp", "myopic_voi", "eig_greedy", "eig_per_cost", "eigdec_greedy", "always_stop", "acquire_all", "random"]
TAU = 0.05


def run_policy(row, pol, e):
    M = get_model(row)
    ep = row["episodes"][e]
    obs = list(M.observed0)
    reqs = []
    rng = random.Random(sum(map(ord, row["id"])) * 31 + e)
    real = [M.lab2i[l] for l in row["label_order"]]          # excludes pre-observed (binary "_obs") items
    if pol == "random":
        k = rng.randint(0, M.budget)
        order = rng.sample(real, min(k, len(real)))
        for i in order:
            reqs.append(M.labels[i]); obs.append((i, ep["outcomes"][M.labels[i]]))
    elif pol == "acquire_all":
        for i in real[:M.budget]:
            reqs.append(M.labels[i]); obs.append((i, ep["outcomes"][M.labels[i]]))
    elif pol != "always_stop":
        while len(reqs) < M.budget:
            done = {i for i, _ in obs}
            cand = [i for i in real if i not in done]
            if not cand:
                break
            if pol == "oracle_dp":
                f = M.dp_first(obs)
                if f["move"] == "S":
                    break
                i = f["item"]
            else:
                st = {i: M.item_stats(obs, i) for i in cand}
                if pol == "myopic_voi":
                    i = max(cand, key=lambda j: st[j][0] - M.cost[j])
                    if st[i][0] - M.cost[i] <= 1e-9:
                        break
                elif pol == "eig_greedy":
                    i = max(cand, key=lambda j: st[j][1])
                    if st[i][1] <= TAU:
                        break
                elif pol == "eig_per_cost":
                    ok = [j for j in cand if st[j][1] > TAU]
                    if not ok:
                        break
                    i = max(ok, key=lambda j: st[j][1] / (M.cost[j] + 1e-3))
                elif pol == "eigdec_greedy":
                    i = max(cand, key=lambda j: st[j][2])
                    if st[i][2] <= TAU:
                        break
            reqs.append(M.labels[i]); obs.append((i, ep["outcomes"][M.labels[i]]))
    b = M.belief(obs)
    a = int(np.argmax(M.U @ b))
    return evaluate(row, reqs, row["actions"][a], e)


def work(path_line):
    row = json.loads(path_line)
    out = {p: [] for p in POLICIES}
    for e in range(len(row["episodes"])):
        for p in POLICIES:
            r = run_policy(row, p, e)
            r.pop("invalid", None)
            out[p].append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    lines = [l for p in a.files for l in gzip.open(p, "rt")]
    allres = {p: [] for p in POLICIES}
    with Pool(a.workers) as pool:
        for out in pool.imap_unordered(work, lines, chunksize=4):
            for p in POLICIES:
                allres[p] += out[p]
    summ = {p: aggregate(allres[p], by=("domain", "family", "stratum")) for p in POLICIES}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"files": a.files, "n_rows": len(lines), "summary": summ}, open(a.out, "w"), indent=1)
    keys = ["policy_regret_norm", "cau_expected", "match_dp_first", "mean_requests", "chose_argmax_eig",
            "eig_trap_rate", "chose_voi0_eigpos", "chose_placebo", "over_ask_first", "under_ask_first"]
    print("policy".ljust(14) + "".join(k[:16].rjust(17) for k in keys))
    for p in POLICIES:
        s = summ[p]["all"]
        print(p.ljust(14) + "".join(f"{s.get(k, float('nan')):17.4f}" for k in keys))


if __name__ == "__main__":
    main()
