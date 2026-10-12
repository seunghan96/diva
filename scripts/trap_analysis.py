"""Information-trap anatomy (headline analysis). For every first acquisition decision, classify the error type against
the DP gold and attribute its regret:
  ok            choice is DP-optimal (incl. ties)
  trap          acquired a check with VOI ~ 0 but EIG > 0.05 bits (informative but cannot change the decision)
  placebo       acquired a check with EIG <= 0.05 bits (uninformative)
  overpriced    acquired a check with VOI > 0 but VOI < cost
  suboptimal    acquired a valuable check (VOI >= cost) that is not DP-optimal
  premature     stopped although acquiring is DP-optimal
Reports, per method (and per model), the share of decisions and the share of total regret in each type, plus
over-acquisition when the evidence is sufficient. Tracks: controlled binary (exact) and credit/clinical records (reference model).
Writes results/trap.json."""
import sys, os, json, glob, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read, fac_scores, choose_max, score
from eval_tab import diva_scores, diva_u

TYPES = ["ok", "trap", "placebo", "overpriced", "suboptimal", "premature"]


def classify(sc, ch):
    ex = sc["exact"]
    dties = ex.get("dp_first_ties") or [ex.get("dp_first_action")]
    dties = ["S" if t in (None, "S") else t for t in dties]
    r = score(sc, ch)
    if ch in dties:
        return "ok", r
    if ch == "S":
        return "premature", r
    voi, eig = ex["voi"][ch], ex["eig"][ch]
    cost = next(float(c["cost"]) for c in sc["checks"] if c["label"] == ch)
    if voi <= 1e-9:
        return ("trap" if eig > 0.05 else "placebo"), r
    if voi < cost:
        return "overpriced", r
    return "suboptimal", r


def acc(store, key, sc, ch):
    t, r = classify(sc, ch)
    s = store[key]
    s["n"] += 1; s[t] += 1; s["reg_" + t] += r.get("dp_regret", 0.0); s["reg"] += r.get("dp_regret", 0.0)
    if sc["exact"]["gold_choice"] == "S" and (sc["exact"].get("dp_first_action") in (None, "S")):
        s["n_suff"] += 1; s["over_suff"] += ch != "S"


def new():
    return collections.defaultdict(float)


XC = (json.load(open(f"{IS}/results/extra_controls.json")).get("controlled", {}) if os.path.exists(f"{IS}/results/extra_controls.json") else {})


def controlled(out):
    for f in glob.glob(f"{IS}/outputs/test_bin/facnum_*_finance.jsonl.gz"):
        tag = os.path.basename(f)[7:-9].rsplit("_", 1)[0]
        S = collections.defaultdict(new)
        for d in ["finance", "credit", "clinical", "legal"]:
            env = {s["id"]: s for s in read(f"{IS}/envs_synth/data/binary/{d}_test.jsonl.gz")[:300]}
            fac = {r["id"]: r for r in read(f"{IS}/outputs/test_bin/facnum_{tag}_{d}.jsonl.gz")}
            dr = {r["id"]: max(r["dist"], key=r["dist"].get) for r in read(f"{IS}/outputs/test_dir/dir_{tag}_{d}.jsonl.gz") if r.get("dist")}
            bls = {}
            for p in glob.glob(f"{IS}/baselines/outputs/{tag}/*/{d}_test.jsonl.gz"):
                bls[p.split("/")[-2]] = {r["id"]: r["choice"] for r in read(p) if r.get("choice") is not None}
            for i, sc in env.items():
                if i in fac:
                    fs = fac_scores(sc, fac[i])
                    acc(S, "DIVA", sc, choose_max(fs["net"])); acc(S, "Fac-EIG", sc, choose_max(fs["eig"], 0.05))
                    if tag in XC:
                        from extra_controls import eig_cost_choice
                        acc(S, "Fac-EIG-tuned", sc, eig_cost_choice(sc, fs["eig"], XC[tag]["lam"], XC[tag]["tau"]))
                if i in dr:
                    acc(S, "Direct", sc, dr[i])
                for m, v in bls.items():
                    if i in v and m not in ("random", "oracle_voi", "oracle_dp", "never"):
                        acc(S, m, sc, v[i])  # "direct" = the Direct baseline of Table tab:know (fact-check 10-12)
        out["controlled"][tag] = {k: dict(v) for k, v in S.items()}


def records(out):
    for f in glob.glob(f"{IS}/outputs/test_tab/fk_*.jsonl.gz"):
        tag, ds = os.path.basename(f)[3:-9].rsplit("_", 1)
        if tag in ("q72",):  # knowledge runs incomplete for Qwen2.5-72B
            continue
        env = {s["id"]: s for s in read(f"{IS}/envs_tab/data/{ds}_test.jsonl.gz")}
        fr = {r["id"]: r for r in read(f)}
        dr = {r["id"]: max(r["dist"], key=r["dist"].get) for r in read(f"{IS}/outputs/test_tab/dir_{tag}_{ds}.jsonl.gz") if r.get("dist")}
        bls = {}
        for p in glob.glob(f"{IS}/baselines/outputs/{tag}/*/{ds}_test.jsonl.gz"):
            bls[p.split("/")[-2]] = {r["id"]: r["choice"] for r in read(p) if r.get("choice") is not None}
        S = collections.defaultdict(new)
        for i, sc in env.items():
            if i in fr:
                acc(S, "DIVA", sc, choose_max(diva_scores(sc, fr[i], "cond", "greedy", "elicit", 0.75)["net"]))
                acc(S, "DIVA-U", sc, choose_max(diva_u(sc, fr[i], 1.0)["net"]))
            if i in dr:
                acc(S, "Direct", sc, dr[i])
            for m, v in bls.items():
                if i in v:
                    acc(S, m, sc, v[i])
        out["records"][f"{tag}:{ds}"] = {k: dict(v) for k, v in S.items()}


def knowledge(out):
    """Knowledge-required environments (tabular + SEC), first 300 test scenarios (as Table 3), every method incl. Fac-EIG‡."""
    from eval_real import fio_real, fio_real_u
    from extra_controls import eig_cost_choice
    X = json.load(open(f"{IS}/results/extra_controls.json")) if os.path.exists(f"{IS}/results/extra_controls.json") else {}
    EX = dict(X.get("knowledge", {})); EX.update(X.get("sec", {}))
    for kind, envs in [("tab", ["taiwan", "german", "cdc", "heart"]), ("sec", ["ret_1", "ret_5", "vol_5", "vol_20"])]:
        for e in envs:
            ep = f"{IS}/envs_tab/data/{e}_test.jsonl.gz" if kind == "tab" else f"{IS}/envs_real/data/{e}_test.jsonl.gz"
            envl = read(ep)[:300]; env = {s["id"]: s for s in envl}
            pat = f"{IS}/outputs/test_tab/fk_*_{e}.jsonl.gz" if kind == "tab" else f"{IS}/outputs/test_real/fr_*_{e}.jsonl.gz"
            for f in glob.glob(pat):
                tag = os.path.basename(f)[3:-9][: -(len(e) + 1)]
                fr = {r["id"]: r for r in read(f)}
                dp = f"{IS}/outputs/test_tab/dir_{tag}_{e}.jsonl.gz" if kind == "tab" else f"{IS}/outputs/test_real/dir_{tag}_{e}.jsonl.gz"
                dr = {r["id"]: max(r["dist"], key=r["dist"].get) for r in read(dp) if r.get("dist")} if os.path.exists(dp) else {}
                bls = {}
                for p in glob.glob(f"{IS}/baselines/outputs/{tag}/*/{e}_test.jsonl.gz"):
                    bls[p.split("/")[-2]] = {r["id"]: r["choice"] for r in read(p) if r.get("choice") is not None}
                ex = EX.get(f"{tag}:{e}", {}); lam, tau = ex.get("lam", 0.1), ex.get("tau", 0.02)
                S = collections.defaultdict(new)
                for i, sc in env.items():
                    if i in fr:
                        fs = diva_scores(sc, fr[i], "cond", "greedy", "elicit", 0.75) if kind == "tab" else fio_real(sc, fr[i], lam=0.75)
                        fu = diva_u(sc, fr[i], 1.0) if kind == "tab" else fio_real_u(sc, fr[i], 1.0)
                        acc(S, "DIVA", sc, choose_max(fs["net"])); acc(S, "DIVA-U", sc, choose_max(fu["net"]))
                        acc(S, "Fac-EIG-tuned", sc, eig_cost_choice(sc, fs["eig"], lam, tau))
                    if i in dr:
                        acc(S, "Direct", sc, dr[i])
                    for m, v in bls.items():
                        if i in v:
                            acc(S, m, sc, v[i])
                out["knowledge"][f"{tag}:{e}"] = {k: dict(v) for k, v in S.items()}


if __name__ == "__main__":
    out = {"controlled": {}, "records": {}, "knowledge": {}}
    controlled(out); records(out); knowledge(out)
    json.dump(out, open(f"{IS}/results/trap.json", "w"), indent=1)
    for tr in out:
        agg = collections.defaultdict(lambda: collections.defaultdict(float))
        for cell in out[tr].values():
            for m, s in cell.items():
                for k, v in s.items():
                    agg[m][k] += v
        for m, s in agg.items():
            print(tr, m, "n=%d" % s["n"], " ".join(f"{t}:{s[t]/s['n']:.2f}/{(s['reg_'+t]/s['reg'] if s['reg'] else 0):.2f}" for t in TYPES), "overSuff=%.2f" % (s["over_suff"] / max(s["n_suff"], 1)))
