"""paired clustered-bootstrap CIs for the headline tables.
Table 2 (knowledge, first 300 test rows): for each environment, the mean over LLMs of the per-scenario regret
difference (DP optimum and myopic optimum) of DIVA-U and DIVA against Never, Direct and Fac-EIG‡; scenarios are resampled
(clusters = scenario groups / ids), with the same resample applied to every LLM.
Table 6 (controlled, first 300 test rows per domain): per LLM, DIVA minus its best LLM baseline (lowest mean), resampling
scenarios. Writes results/ci_tables.json."""
import sys, os, glob, json, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read, score, choose_max, fac_scores
from eval_tab import diva_scores, diva_u
from eval_real import fio_real, fio_real_u
from extra_controls import eig_cost_choice

B = 1000
rng = np.random.default_rng(0)
_X = json.load(open(f"{IS}/results/extra_controls.json"))
EX = dict(_X.get("knowledge", {})); EX.update(_X.get("sec", {}))


def boot(mat, groups):
    """mat: LLMs x scenarios differences; groups: cluster id per scenario. CI of the mean over LLMs of scenario means."""
    ug = sorted(set(groups)); idx = collections.defaultdict(list)
    for j, g in enumerate(groups):
        idx[g].append(j)
    est = float(np.mean(mat.mean(1)))
    bs = []
    for _ in range(B):
        pick = rng.choice(len(ug), len(ug), replace=True)
        cols = np.concatenate([idx[ug[p]] for p in pick])
        bs.append(float(np.mean(mat[:, cols].mean(1))))
    return [est, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def knowledge():
    out = {}
    for kind, envs in [("tab", ["taiwan", "german", "cdc", "heart"]), ("sec", ["ret_1", "ret_5", "vol_5", "vol_20"])]:
        for e in envs:
            ep = f"{IS}/envs_tab/data/{e}_test.jsonl.gz" if kind == "tab" else f"{IS}/envs_real/data/{e}_test.jsonl.gz"
            envl = read(ep)[:300]; env = {s["id"]: s for s in envl}; ids = [s["id"] for s in envl]
            per = collections.defaultdict(dict)
            for f in glob.glob(f"{IS}/outputs/test_tab/fk_*_{e}.jsonl.gz" if kind == "tab" else f"{IS}/outputs/test_real/fr_*_{e}.jsonl.gz"):
                tag = os.path.basename(f)[3:-9][: -(len(e) + 1)]
                fr = {r["id"]: r for r in read(f)}
                dp = (f"{IS}/outputs/test_tab/dir_{tag}_{e}.jsonl.gz" if kind == "tab" else f"{IS}/outputs/test_real/dir_{tag}_{e}.jsonl.gz")
                if not os.path.exists(dp):
                    continue
                dr = {r["id"]: max(r["dist"], key=r["dist"].get) for r in read(dp) if r.get("dist")}
                if not all(i in fr and i in dr for i in ids):
                    continue
                ex = EX.get(f"{tag}:{e}", {}); lam, tau = ex.get("lam", 0.1), ex.get("tau", 0.02)
                for i in ids:
                    sc = env[i]
                    fs = diva_scores(sc, fr[i], "cond", "greedy", "elicit", 0.75) if kind == "tab" else fio_real(sc, fr[i], lam=0.75)
                    fu = diva_u(sc, fr[i], 1.0) if kind == "tab" else fio_real_u(sc, fr[i], 1.0)
                    ch = {"diva": choose_max(fs["net"]), "diva_u": choose_max(fu["net"]), "never": "S", "direct": dr[i],
                          "fac_eig_cost": eig_cost_choice(sc, fs["eig"], lam, tau)}
                    per[tag][i] = {m: score(sc, c) for m, c in ch.items()}
            tags = sorted(per)
            if not tags:
                continue
            groups = [env[i].get("group_id") or i for i in ids]
            res = {"n_llm": len(tags)}
            for a in ("diva_u", "diva"):
                for b_ in ("never", "direct", "fac_eig_cost"):
                    for k, nm in (("dp_regret", "dp"), ("norm_regret", "myo")):
                        mat = np.array([[per[t][i][a][k] - per[t][i][b_][k] for i in ids] for t in tags])
                        res[f"{a}-{b_}:{nm}"] = boot(mat, groups)
            out[e] = res
            print(e, len(tags), {k: [round(100 * x, 2) for x in v] for k, v in res.items() if k.startswith("diva_u") and "dp" in k}, flush=True)
    return out


def controlled():
    out = {}
    BL = ["direct", "cot_need", "sc_cot", "react", "voi_prompt", "voi_paper", "finrisk_ask", "bed_llm", "pal"]
    for f in glob.glob(f"{IS}/outputs/test_bin/facnum_*_finance.jsonl.gz"):
        tag = os.path.basename(f)[7:-9].rsplit("_", 1)[0]
        diffs = collections.defaultdict(list); groups = []
        ok = True
        for d in ["finance", "credit", "clinical", "legal"]:
            envl = read(f"{IS}/envs_synth/data/binary/{d}_test.jsonl.gz")[:300]
            fac = {r["id"]: r for r in read(f"{IS}/outputs/test_bin/facnum_{tag}_{d}.jsonl.gz")}
            bls = {}
            for m in BL:
                p = f"{IS}/baselines/outputs/{tag}/{m}/{d}_test.jsonl.gz"
                if os.path.exists(p):
                    bls[m] = {r["id"]: r["choice"] for r in read(p) if r.get("choice") is not None}
            for sc in envl:
                i = sc["id"]
                if i not in fac:
                    ok = False; break
                dv = score(sc, choose_max(fac_scores(sc, fac[i])["net"]))["dp_regret"]
                groups.append(sc.get("group_id") or i)
                for m, v in bls.items():
                    diffs[m].append(dv - score(sc, v[i])["dp_regret"] if i in v else np.nan)
        if not ok or not diffs:
            continue
        cand = {m: v for m, v in diffs.items() if np.mean(np.isnan(v)) < 0.2}
        if len(cand) < 5:
            continue
        # best baseline = the one with the smallest mean regret, i.e. the largest mean (DIVA - baseline)
        best = max(cand, key=lambda m: np.nanmean(cand[m]))
        v = np.array(cand[best]); keep = ~np.isnan(v)
        out[tag] = {"best": best, "ci": boot(v[keep][None, :], [g for g, k in zip(groups, keep) if k])}
        print(tag, best, [round(100 * x, 2) for x in out[tag]["ci"]], flush=True)
    return out


if __name__ == "__main__":
    res = {"knowledge": knowledge(), "controlled": controlled()}
    json.dump(res, open(f"{IS}/results/ci_tables.json", "w"), indent=1)
