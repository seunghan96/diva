"""Sig. wins of DIVA-U for Table tab:know (user-approved review item, 2026-10-11).
For every environment column of Table tab:know and every LLM of the fixed set: paired clustered bootstrap (B=1000, clusters =
scenario group_id or id, as boot() in ci_tables.py) of the per-scenario first-decision DP-regret difference DIVA-U minus that
LLM's best external baseline in that environment (lowest mean among the Prompting ... Program-aided rows of Table tab:know, the
set of the Wins row; controlled: DIVA-U = DIVA). A significant win = 95% CI upper bound < 0.
Scenarios: knowledge-required = the first-300 rows cached by agg_know_main.py (results/know_choices), with the row set of
Table tab:know; controlled = the first 300 test scenarios per domain, as agg_bin.py '@sub'. Writes results/sig_wins.json."""
import sys, os, json, glob, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read, score, fac_scores, choose_max
from complete import KNOW_LLMS, CTRL_BL_LLMS, METHOD_GROUP, DOM_ORDER
import make_tables as mt

B = 1000
EXT_K = [m for m, _ in mt.KNOW_METHS if METHOD_GROUP.get(mt.mname(m)) in mt.EXTERNAL_GROUPS]
CKEY = {mt.mname(k): k for k, _ in mt.CTRL_METHS}
EXT_C = [CKEY[mt.mname(m)] for m in EXT_K]
CORE = ["direct", "cot_need", "bed_llm", "voi_prompt", "pal", "sc", "react", "ask_p", "voi_ask"]  # agg_know_main CORE
CORE0 = ["direct", "cot_need", "bed_llm", "voi_prompt", "pal"]


def boot_ci(diff, groups, rng):
    ug = sorted(set(groups)); gi = {g: k for k, g in enumerate(ug)}
    gid = np.array([gi[g] for g in groups])
    sums = np.bincount(gid, weights=diff, minlength=len(ug)); cnts = np.bincount(gid, minlength=len(ug)).astype(float)
    pick = rng.integers(0, len(ug), size=(B, len(ug)))
    bs = sums[pick].sum(1) / cnts[pick].sum(1)
    return [float(diff.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def knowledge(R, rng):
    out = {}
    for ek, _ in mt.KNOW_ENVS:
        kind, envn = ek.split(":")
        ep = f"{IS}/envs_tab/data/{envn}_test.jsonl.gz" if kind == "tab" else f"{IS}/envs_real/data/{envn}_test.jsonl.gz"
        envl = read(ep); env = {s["id"]: s for s in envl}; first300 = [s["id"] for s in envl[:300]]
        for t in KNOW_LLMS:
            p = f"{IS}/results/know_choices/{envn}_{t}.json"
            if not os.path.exists(p):
                continue
            C = json.load(open(p)); sub = [i for i in first300 if i in C]
            ids_all = [i for i in sub if all(m in C[i] for m in CORE)]
            ids = ids_all if len(ids_all) >= 0.95 * len(first300) else [i for i in sub if all(m in C[i] for m in CORE0)]
            reg = {m: {i: score(env[i], C[i][m])["dp_regret"] for i in ids if m in C[i]} for m in EXT_K + ["diva_u"]}
            means = {m: float(np.mean(list(v.values()))) for m, v in reg.items() if len(v) >= 0.95 * len(first300)}
            rec = {"means": means}
            exts = [m for m in EXT_K if m in means]
            if "diva_u" in means and len(exts) == len(EXT_K):
                best = min(exts, key=lambda m: R[ek][t][f"s300:{m}"][mt.KEY])  # the Table-3 per-LLM means decide the best baseline
                pi = [i for i in ids if i in reg["diva_u"] and i in reg[best]]
                d = np.array([reg["diva_u"][i] - reg[best][i] for i in pi])
                rec.update(best=best, n=len(pi), ci=boot_ci(d, [env[i].get("group_id") or i for i in pi], rng))
            out.setdefault(ek, {})[t] = rec
    return out


def bl_choices(tag, dom):
    out = {}
    for p in glob.glob(f"{IS}/baselines/outputs/{tag}/*/{dom}_test*.jsonl.gz"):
        if p.endswith(".calls.jsonl.gz") or "episode" in p:
            continue
        for r in read(p):
            if r.get("choice") is not None:
                out.setdefault("bl_" + p.split("/")[-2], {})[r["id"]] = r["choice"]
    return out


def controlled(rng):
    out = {}
    for d in DOM_ORDER:
        envl = read(f"{IS}/envs_synth/data/binary/{d}_test.jsonl.gz"); env = {s["id"]: s for s in envl}
        ids = [s["id"] for s in envl[:300]]
        for t in CTRL_BL_LLMS:
            ch = bl_choices(t, d)
            fp = f"{IS}/outputs/test_bin/facnum_{t}_{d}.jsonl.gz"
            if os.path.exists(fp):
                ch["diva"] = {r["id"]: choose_max(fac_scores(env[r["id"]], r)["net"]) for r in read(fp) if r["id"] in env}
            reg = {m: {i: score(env[i], ch[m][i])["dp_regret"] for i in ids if i in ch.get(m, {})} for m in EXT_C + ["diva"]}
            means = {m: float(np.mean(list(v.values()))) for m, v in reg.items() if len(v) >= 300}
            rec = {"means": means}
            if "diva" in means and all(m in means for m in EXT_C):
                best = min(EXT_C, key=lambda m: means[m])
                d_ = np.array([reg["diva"][i] - reg[best][i] for i in ids])
                rec.update(best=best, n=len(ids), ci=boot_ci(d_, [env[i].get("group_id") or i for i in ids], rng))
            out.setdefault(d, {})[t] = rec
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    R = json.load(open(f"{IS}/results/know_main.json"))
    res = {"knowledge": knowledge(R, rng), "controlled": controlled(rng)}
    cnt = {}
    for grp, E in res.items():
        for e, d in E.items():
            ok = [t for t in (KNOW_LLMS if grp == "knowledge" else CTRL_BL_LLMS) if "ci" in d.get(t, {})]
            cnt[e] = {"n_llm": len(ok), "sig": sum(d[t]["ci"][2] < 0 for t in ok)}
    res["counts"] = cnt
    json.dump(res, open(f"{IS}/results/sig_wins.json", "w"), indent=1)
    print({e: f"{c['sig']}/{c['n_llm']}" for e, c in cnt.items()})
