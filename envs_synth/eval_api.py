"""Evaluation API for the synthetic information environments (both tracks).

Given a scenario row, a model's sequence of requests (item labels, in order) and its final decision (action name),
returns exact decision-theoretic scores along the realized trajectory of one stored episode:

  first-step:  voi_regret_first   = max_i VOI_i - VOI_chosen       (0 if the model stopped and all VOI are 0; if it stopped
                                                                   while some VOI > 0, the regret is max VOI)
               net_regret_first   = max(0, max_i net_i) - (net_chosen or 0 if stop)
               match_gold_first   = first move in exact.gold_ties   (myopic gold: argmax net if > 0 else "S")
               match_dp_first     = first move in exact.dp_first_ties (non-myopic DP gold)
  policy:      policy_regret      = sum over steps of [V*(h_t) - Q*(h_t, move_t)], including the final decision
                                    (Q*(h, decide a) = E[U(a)|h]); an unbiased per-episode estimate of V*(root) - V^pi(root)
                                    for any policy that is deterministic given the observations (average over episodes).
  utility:     cau_expected       = E[U(decision)|final history] - total cost   (cost-adjusted utility, low variance)
               cau_realized       = U(decision, true state) - total cost
               v_star_root        = optimal expected cost-adjusted utility (DP)
  EIG vs VOI:  per request step: chose_argmax_voi, chose_argmax_eig, divergent_node (argmax sets disjoint),
               eig_trap (divergent and chose argmax EIG), chose_voi0_eigpos, chose_placebo, over_ask (DP says stop here),
               voi_rank / eig_rank of the chosen item; under_ask at the stop node (DP says acquire).

Interfaces
  evaluate(row, requests, decision, episode=0) -> dict
  rank_metrics(row, scores: {label: float}) -> dict (Spearman with VOI / net / EIG, top-1 matches)
  aggregate(results, by=("domain", "family", "stratum")) -> dict
  Env(row, episode=0, mode="numeric"|"qualitative")   interactive: .prompt(), .request(label) -> text, .decide(action)
  CLI: python eval_api.py --data data/ms/finance_test.jsonl.gz --preds preds.jsonl.gz [--out summary.json]
       preds rows: {"id", "requests": [labels], "decision": action, "episode": int (optional, default 0),
                    "scores": {label: float} (optional)}
"""
import argparse, gzip, json, math, os, sys
from collections import defaultdict, OrderedDict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from core import Model

TOL = 1e-7
_CACHE = OrderedDict()


def get_model(row):
    k = row["id"]
    if k in _CACHE:
        _CACHE.move_to_end(k)
        return _CACHE[k]
    M = Model(row["model"])
    _CACHE[k] = M
    if len(_CACHE) > 256:
        _CACHE.popitem(last=False)
    return M


def read_jsonl_gz(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


def _norm_action(row, decision):
    acts = row["actions"]
    if decision is None:
        return None
    d = str(decision).strip().upper()
    for i, a in enumerate(acts):
        if d == a.upper():
            return i
    for i, a in enumerate(acts):          # lenient prefix match (e.g. "APPROVE_REDUCED" vs "APPROVE")
        if d.startswith(a.upper()) and not any(d.startswith(b.upper()) and len(b) > len(a) for b in acts):
            return i
    return None


def _rank(d, key):
    """1-based rank of key in dict d by descending value (ties share the best rank)."""
    v = d[key]
    return 1 + sum(1 for x in d.values() if x > v + TOL)


def util_range(row):
    U = np.asarray(row["model"]["utility"], float)
    return float(U.max() - U.min()) or 1.0


def evaluate(row, requests, decision, episode=0):
    M = get_model(row)
    labs = M.labels
    real = row["label_order"]
    ep = row["episodes"][episode % len(row["episodes"])]
    placebo = {c["label"]: c.get("is_placebo", False) for c in row["checks"]}
    obs = list(M.observed0)
    t0 = M.node_table(obs)
    v_root = t0["v_star"]
    steps, invalid = [], []
    cost = 0.0
    adv_sum = 0.0
    seen = set()
    for raw in requests:
        lab = str(raw).strip().strip("()").upper()
        if lab not in real or lab in seen:
            invalid.append({"label": raw, "why": "unknown" if lab not in real else "repeat"})
            continue
        if len(seen) >= M.budget:
            invalid.append({"label": raw, "why": "over_budget"})
            continue
        i = M.lab2i[lab]
        t = M.node_table(obs)
        voi = {labs[j]: v for j, v in t["voi"].items()}
        eig = {labs[j]: v for j, v in t["eig"].items()}
        net = {labs[j]: v for j, v in t["net"].items()}
        q = {labs[j]: v for j, v in t["q"].items()}
        mv, me = max(voi.values()), max(eig.values())
        a_voi = {l for l, v in voi.items() if v >= mv - TOL}
        a_eig = {l for l, v in eig.items() if v >= me - TOL}
        divergent = mv > TOL and not (a_voi & a_eig)
        stop_val = t["v0"]
        dp_stop = stop_val >= t["v_star"] - TOL
        adv = t["v_star"] - q[lab]
        adv_sum += adv
        steps.append({
            "label": lab, "voi": voi[lab], "eig": eig[lab], "net": net[lab], "cost": float(M.cost[i]),
            "voi_regret": mv - voi[lab], "net_regret": max(0.0, max(net.values())) - net[lab], "q_regret": adv,
            "chose_argmax_voi": lab in a_voi, "chose_argmax_eig": lab in a_eig, "divergent_node": divergent,
            "eig_trap": divergent and lab in a_eig, "chose_voi0_eigpos": voi[lab] == 0 and eig[lab] > 0.01,
            "chose_placebo": bool(placebo.get(lab)), "over_ask_dp": dp_stop,
            "over_ask_myopic": max(net.values()) <= TOL,
            "voi_rank": _rank(voi, lab), "eig_rank": _rank(eig, lab), "n_avail": len(voi),
        })
        cost += float(M.cost[i])
        seen.add(lab)
        obs.append((i, int(ep["outcomes"][lab])))
    tf = M.node_table(obs)
    ai = _norm_action(row, decision)
    eu = tf["eu"]
    can_more = len(seen) < M.budget and len(tf["q"]) > 0
    under_ask = bool(can_more and tf["v_star"] > tf["v0"] + TOL)
    if ai is None:
        dec_adv = tf["v_star"] - float(eu.min())        # unparseable decision scored as the worst action
        eu_dec = float(eu.min())
        realized = float(np.min(np.asarray(row["model"]["utility"])[:, ep["state"]]))
    else:
        dec_adv = tf["v_star"] - float(eu[ai])
        eu_dec = float(eu[ai])
        realized = float(row["model"]["utility"][ai][ep["state"]])
    policy_regret = adv_sum + dec_adv
    first = steps[0]["label"] if steps else "S"
    ex = row["exact"]
    t0voi = list(ex["voi"].values())
    mv0 = max(t0voi) if t0voi else 0.0
    if steps:
        voi_regret_first = steps[0]["voi_regret"]
        net_regret_first = steps[0]["net_regret"]
    else:
        voi_regret_first = mv0
        net_regret_first = max(0.0, max(ex["net"].values()))
    R = util_range(row)
    res = {
        "id": row["id"], "domain": row["domain"], "track": row["track"], "family": row["family"],
        "stratum": row["stratum"], "episode": episode, "n_requests": len(steps), "n_invalid": len(invalid),
        "first_move": first, "match_gold_first": first in ex["gold_ties"], "match_dp_first": first in ex["dp_first_ties"],
        "gold_stop": ex["gold_choice"] == "S", "dp_stop": ex["dp_first_action"] == "S", "model_stop_first": first == "S",
        "voi_regret_first": voi_regret_first, "net_regret_first": net_regret_first,
        "voi_regret_first_norm": voi_regret_first / R, "net_regret_first_norm": net_regret_first / R,
        "policy_regret": policy_regret, "policy_regret_norm": policy_regret / R,
        "decision": None if ai is None else row["actions"][ai], "decision_valid": ai is not None,
        "decision_bayes": row["actions"][tf["best_action"]], "decision_correct": ai is not None and eu[ai] >= tf["v0"] - TOL,
        "decision_regret": tf["v0"] - eu_dec, "under_ask": under_ask,
        "total_cost": cost, "cau_expected": eu_dec - cost, "cau_realized": realized - cost, "v_star_root": v_root,
        "v_stop_root": t0["v0"], "util_range": R, "steps": steps, "invalid": invalid,
    }
    return res


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    ra = _avg_rank(a)
    rb = _avg_rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def _avg_rank(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    i = 0
    xs = x[order]
    while i < len(x):
        j = i
        while j + 1 < len(x) and abs(xs[j + 1] - xs[i]) < 1e-12:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def rank_metrics(row, scores):
    """scores: {label: float}, higher = more worth requesting. Compared against root-level exact values."""
    ex = row["exact"]
    labs = [l for l in row["label_order"] if l in scores]
    s = [float(scores[l]) for l in labs]
    out = {"spearman_voi": _spearman(s, [ex["voi"][l] for l in labs]),
           "spearman_net": _spearman(s, [ex["net"][l] for l in labs]),
           "spearman_eig": _spearman(s, [ex["eig"][l] for l in labs])}
    if labs:
        top = labs[int(np.argmax(s))]
        mv, me = max(ex["voi"].values()), max(ex["eig"].values())
        out["top1_is_argmax_voi"] = ex["voi"][top] >= mv - TOL
        out["top1_is_argmax_eig"] = ex["eig"][top] >= me - TOL
        out["top1_voi_regret"] = mv - ex["voi"][top]
    return out


def aggregate(results, by=("domain", "family", "stratum")):
    def summ(rs):
        if not rs:
            return {}
        st = [s for r in rs for s in r["steps"]]
        div = [s for s in st if s["divergent_node"]]
        g_stop = [r for r in rs if r["gold_stop"]]
        g_acq = [r for r in rs if not r["gold_stop"]]
        mean = lambda xs: float(np.mean(xs)) if len(xs) else float("nan")
        out = {
            "n": len(rs),
            "voi_regret_first_norm": mean([r["voi_regret_first_norm"] for r in rs]),
            "net_regret_first_norm": mean([r["net_regret_first_norm"] for r in rs]),
            "policy_regret_norm": mean([r["policy_regret_norm"] for r in rs]),
            "policy_regret": mean([r["policy_regret"] for r in rs]),
            "cau_expected": mean([r["cau_expected"] for r in rs]),
            "cau_realized": mean([r["cau_realized"] for r in rs]),
            "v_star_root": mean([r["v_star_root"] for r in rs]),
            "match_gold_first": mean([r["match_gold_first"] for r in rs]),
            "match_dp_first": mean([r["match_dp_first"] for r in rs]),
            "decision_correct": mean([r["decision_correct"] for r in rs]),
            "decision_valid": mean([r["decision_valid"] for r in rs]),
            "mean_requests": mean([r["n_requests"] for r in rs]),
            "mean_cost": mean([r["total_cost"] for r in rs]),
            "invalid_rate": mean([r["n_invalid"] > 0 for r in rs]),
            # stopping
            "stop_recall": mean([r["model_stop_first"] for r in g_stop]),            # stopped when gold = stop
            "over_ask_first": mean([not r["model_stop_first"] for r in g_stop]),
            "under_ask_first": mean([r["model_stop_first"] for r in g_acq]),       # stopped when gold = acquire
            "under_ask_final": mean([r["under_ask"] for r in rs]),
            # EIG vs VOI choice statistics over all request steps
            "n_steps": len(st),
            "chose_argmax_voi": mean([s["chose_argmax_voi"] for s in st]),
            "chose_argmax_eig": mean([s["chose_argmax_eig"] for s in st]),
            "n_divergent_steps": len(div),
            "eig_trap_rate": mean([s["eig_trap"] for s in div]),
            "voi_choice_rate_divergent": mean([s["chose_argmax_voi"] for s in div]),
            "chose_voi0_eigpos": mean([s["chose_voi0_eigpos"] for s in st]),
            "chose_placebo": mean([s["chose_placebo"] for s in st]),
            "over_ask_dp_steps": mean([s["over_ask_dp"] for s in st]),
            "mean_voi_rank": mean([s["voi_rank"] for s in st]),
            "mean_eig_rank": mean([s["eig_rank"] for s in st]),
        }
        return out
    agg = {"all": summ(results)}
    for k in by:
        groups = defaultdict(list)
        for r in results:
            groups[r.get(k)].append(r)
        agg[k] = {str(g): summ(rs) for g, rs in sorted(groups.items(), key=lambda x: str(x[0]))}
    return agg


# ------------------------------------------------------------------ interactive environment
class Env:
    """Interactive wrapper: reveals stored episode outcomes for requested items; scoring via evaluate()."""

    def __init__(self, row, episode=0, mode="numeric"):
        self.row, self.episode, self.mode = row, episode, mode
        self.ep = row["episodes"][episode % len(row["episodes"])]
        self.requests = []
        self.done = False

    def prompt(self):
        r = self.row
        if r["track"] == "ms":
            return r["text_qualitative"] if self.mode == "qualitative" else r["text_numeric"]
        return r["text"]

    def observation_text(self, lab):
        r = self.row
        c = [c for c in r["checks"] if c["label"] == lab][0]
        o = int(self.ep["outcomes"][lab])
        if r["track"] == "binary":
            return (f"Result of ({lab}) {c['name']}: it found that {c['finding_text']}." if o == 1 else
                    f"Result of ({lab}) {c['name']}: it did not find that {c['finding_text']}.")
        return f"Result of ({lab}) {c['name']}: '{c['outcomes'][o]}'."

    def request(self, label):
        lab = str(label).strip().strip("()").upper()
        if self.done:
            return "The episode is over."
        if lab not in self.row["label_order"]:
            return f"There is no request ({lab})."
        if lab in self.requests:
            return f"({lab}) was already requested; its result is unchanged: " + self.observation_text(lab)
        if len(self.requests) >= self.row["budget"]:
            return "The request budget is exhausted; you must decide now."
        self.requests.append(lab)
        return self.observation_text(lab)

    def decide(self, action):
        self.done = True
        return evaluate(self.row, self.requests, action, self.episode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--per_item", default=None, help="optional jsonl.gz of per-item results")
    a = ap.parse_args()
    rows = {}
    for p in a.data:
        for r in read_jsonl_gz(p):
            rows[r["id"]] = r
    res, ranks = [], []
    for p in read_jsonl_gz(a.preds):
        r = rows.get(p["id"])
        if r is None:
            continue
        e = evaluate(r, p.get("requests", []), p.get("decision"), p.get("episode", 0))
        if p.get("scores"):
            rm = rank_metrics(r, p["scores"])
            e.update({"rank_" + k: v for k, v in rm.items()})
            ranks.append(rm)
        res.append(e)
    agg = aggregate(res)
    if ranks:
        agg["ranking"] = {k: float(np.nanmean([x[k] for x in ranks if k in x])) for k in ranks[0]}
    s = json.dumps(agg, indent=1)
    if a.out:
        open(a.out, "w").write(s)
    else:
        print(s)
    if a.per_item:
        with gzip.open(a.per_item, "wt") as f:
            for e in res:
                f.write(json.dumps(e) + "\n")


if __name__ == "__main__":
    main()
