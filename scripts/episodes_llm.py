"""Score multi-step prompting episodes on the knowledge-required environments (baselines.run --mode episode on *_epk
envs, worlds sampled from the reference model). For each episode: posterior of the reference model given the acquired
outcomes, utility with (a) the reference-posterior final action and (b) the policy's own final action, net of costs.
Episode regret = (DP value of the reference model - utility) / l, averaged over scenarios x worlds (unbiased for the
exact expected regret); worlds are averaged within a scenario first ('n' = scenarios, 'n_ep' = episodes). Slices: s150 (rows 0-149, used by the tables), s300 (rows 0-299) and k4 (rows 300-999).
Writes results/episodes_llm.json {env: {tag: {method: {slice: {"regret_ref", "regret", "n_acq", "n"}}}}}."""
import os, sys, glob, json, gzip, ast, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read

TAB = ["taiwan", "german", "cdc", "heart"]; SEC = ["ret_1", "ret_5", "vol_5", "vol_20"]
WORLDS = int(os.environ.get("EPL_WORLDS", "1"))  # user rule 10-11: same number of sampled worlds for all 20 LLMs (1; world 0)
METHODS = ["direct", "cot_need", "bed_llm", "voi_prompt", "pal", "sc_cot", "react", "finrisk_ask", "voi_paper"]


def post_fn(sc, kind):
    if kind == "tab":
        J = np.asarray(sc["joint"], dtype=float).reshape(64, 2)
        BITS = (np.arange(64)[:, None] >> np.arange(6)) & 1
        idx = {L: i for i, L in enumerate(sc.get("label_order") or [c["label"] for c in sc["checks"]])}
        def f(acq):
            w = np.ones(64, bool)
            for k, v in acq:
                w &= BITS[:, idx[k]] == v
            p = J[w].sum(0); return p[1] / p.sum()
        return f
    m = ast.literal_eval(sc["model"]) if isinstance(sc["model"], str) else sc["model"]
    pri = np.asarray(m["prior"], dtype=float); items = {it["label"]: np.asarray(it["lik"], dtype=float) for it in m["items"]}
    def f(acq):
        p = pri.copy()
        for k, v in acq:
            p = p * items[k][:, v]
        p = p / p.sum(); return p[1]
    return f


def main():
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    for kind, envs in [("tab", TAB), ("sec", SEC)]:
        for e in envs:
            ep = f"{IS}/envs_tab/data/{e}_epk.jsonl.gz" if kind == "tab" else f"{IS}/envs_real/data/{e}_epk.jsonl.gz"
            envl = read(ep); env = {s["id"]: s for s in envl}; pos = {s["id"]: k for k, s in enumerate(envl)}
            PF = {}
            for f in glob.glob(f"{IS}/baselines/outputs/*/*/{e}_epk.episode.jsonl.gz"):
                tag, meth = f.split("/")[-3], f.split("/")[-2]
                if meth not in METHODS:
                    continue
                SL = {"s150": lambda k: k < 150, "s300": lambda k: k < 300, "k4": lambda k: 300 <= k < 1000}
                R = {sl: collections.defaultdict(lambda: collections.defaultdict(list)) for sl in SL}
                seen = set()
                for r in read(f):
                    key = (r.get("id"), r.get("rep"))
                    if r.get("id") not in env or key in seen or "requests" not in r or int(r.get("rep", 0) or 0) >= WORLDS:
                        continue
                    seen.add(key)
                    sc = env[r["id"]]; g, l = float(sc["g"]), float(sc["l"])
                    if r["id"] not in PF:
                        PF[r["id"]] = post_fn(sc, kind)
                    acq = [(a["label"], int(a["value"])) for a in r.get("acquired", [])]
                    b = PF[r["id"]](tuple(acq)) if acq else PF[r["id"]](())
                    cost = float(r.get("cost", 0.0)); u_act = g * (1 - b) - l * b
                    acts = sc.get("actions") or []
                    own = u_act if (acts and r.get("decision") == acts[0]) else 0.0
                    dpv = float(sc["exact"]["dp_value"])
                    for sl, inside in SL.items():
                        if inside(pos[r["id"]]):
                            R[sl][r["id"]]["ref"].append((dpv - (max(u_act, 0.0) - cost)) / l); R[sl][r["id"]]["own"].append((dpv - (own - cost)) / l)
                            R[sl][r["id"]]["n_acq"].append(len(acq))
                for sl, d in R.items():
                    if d:  # average the sampled worlds within a scenario first, then over scenarios (1 or 2 worlds per scenario)
                        mean = lambda k: float(np.mean([np.mean(v[k]) for v in d.values()]))
                        out[e][tag].setdefault(meth, {})[sl] = {"regret_ref": mean("ref"), "regret": mean("own"), "n_acq": mean("n_acq"),
                                                               "n": len(d), "n_ep": sum(len(v["ref"]) for v in d.values())}
            print(e, {t: sorted(v) for t, v in out[e].items()}, flush=True)
    json.dump(out, open(f"{IS}/results/episodes_llm.json", "w"), indent=1)


if __name__ == "__main__":
    main()
