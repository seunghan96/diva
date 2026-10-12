"""Information-trap anatomy for the methods NOT covered by results/trap.json (Never, Always ask, Reason., DIVA-kappa,
DIVA-beta, DIVA-DP). Pure post-processing of existing outputs with the SAME error classifier (trap_analysis.acc) and the
SAME scenario sets (controlled: first 300 test scenarios per domain; knowledge: first 300 per environment).
  controlled  Never (always stop), Always ask (top net-VOI check), DIVA-DP (exact DP on DIVA's elicited quantities),
              Reason. (native reasoning, outputs/think)
  knowledge   Never, Always ask, DIVA-kappa, DIVA-beta, Reason. (from results/know_choices, written by agg_know_main.py)
Writes results/trap_extra.json (trap.json is left untouched); read by figs_all.fig_trap."""
import sys, os, json, glob, collections
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read, fac_scores, choose_max
from dcore import dp_value
from trap_analysis import acc, new


def controlled(out):
    for f in glob.glob(f"{IS}/outputs/test_bin/facnum_*_finance.jsonl.gz"):
        tag = os.path.basename(f)[7:-9].rsplit("_", 1)[0]
        dpc = json.load(open(f"{IS}/results/dp_cache/{tag}.json")) if os.path.exists(f"{IS}/results/dp_cache/{tag}.json") else {}
        S = collections.defaultdict(new)
        for d in ["finance", "credit", "clinical", "legal"]:
            env = {s["id"]: s for s in read(f"{IS}/envs_synth/data/binary/{d}_test.jsonl.gz")[:300]}
            fac = {r["id"]: r for r in read(f"{IS}/outputs/test_bin/facnum_{tag}_{d}.jsonl.gz")}
            th = {r["id"]: max(r["dist"], key=r["dist"].get) for r in read(f"{IS}/outputs/think/dir_{tag}_{d}.jsonl.gz") if r.get("dist")}
            for i, sc in env.items():
                acc(S, "never", sc, "S")
                if i in fac:
                    fs = fac_scores(sc, fac[i])
                    acc(S, "always_top", sc, max(fs["net"], key=fs["net"].get))
                    if len(sc["checks"]) <= 8:
                        keys = list(fs["par"]); _, a = dp_value(fs["b"], [fs["par"][k] for k in keys], float(sc["g"]), float(sc["l"]))
                        acc(S, "diva_dp", sc, "S" if a == "S" else keys[a])
                    elif i in dpc:
                        acc(S, "diva_dp", sc, dpc[i])
                if i in th:
                    acc(S, "think", sc, th[i])
        out["controlled"][tag] = {k: dict(v) for k, v in S.items()}


def knowledge(out):
    for kind, envs in [("tab", ["taiwan", "german", "cdc", "heart"]), ("sec", ["ret_1", "ret_5", "vol_5", "vol_20"])]:
        for e in envs:
            ep = f"{IS}/envs_tab/data/{e}_test.jsonl.gz" if kind == "tab" else f"{IS}/envs_real/data/{e}_test.jsonl.gz"
            env = {s["id"]: s for s in read(ep)[:300]}
            for p in glob.glob(f"{IS}/results/know_choices/{e}_*.json"):
                tag = os.path.basename(p)[len(e) + 1:-5]
                C = json.load(open(p))
                S = collections.defaultdict(new)
                for i, sc in env.items():
                    for m in ["never", "always_top", "diva_kappa", "diva_beta", "think", "diva_dp"]:
                        if i in C and m in C[i]:
                            acc(S, m, sc, C[i][m])
                out["knowledge"][f"{tag}:{e}"] = {k: dict(v) for k, v in S.items()}


if __name__ == "__main__":
    out = {"controlled": {}, "knowledge": {}}
    controlled(out); knowledge(out)
    json.dump(out, open(f"{IS}/results/trap_extra.json", "w"), indent=1)
    for tr, cells in out.items():
        cnt = collections.Counter(m for c in cells.values() for m in c)
        print(tr, len(cells), dict(cnt))
