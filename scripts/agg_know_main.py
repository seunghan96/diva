"""Main knowledge-required table data: for each (model, env) the methods are scored on IDENTICAL row sets.
Subset S300 = first 300 test rows (where LLM baselines from baselines/outputs exist); FULL = all rows with DIVA output.
Methods: DIVA (frozen PREREG_KNOW config; SEC: frozen single-question config, joint reported separately), direct,
baselines (cot_need, bed_llm, voi_prompt, pal), Fac-EIG-cost (dev-tuned lam/tau from results/extra_controls.json when
available), never (always-stop). Writes results/know_main.json."""
import sys, os, json, glob, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read, score, summarize, choose_max
from eval_tab import diva_scores, diva_u
from eval_real import fio_real, fio_real_u
from extra_controls import eig_cost_choice

_X = json.load(open(f"{IS}/results/extra_controls.json")) if os.path.exists(f"{IS}/results/extra_controls.json") else {}
EX = dict(_X.get("knowledge", {})); EX.update(_X.get("sec", {}))
BL = ["cot_need", "bed_llm", "voi_prompt", "pal"]
NEWBL = {"sc_cot": "sc", "react": "react", "finrisk_ask": "ask_p", "voi_paper": "voi_ask"}
KAP = json.load(open(f"{IS}/results/kappa.json")) if os.path.exists(f"{IS}/results/kappa.json") else {}
from kappa_tune import kappa_choice, beta_choice


def bl_rows(tag, envn):
    out = {}
    for m in BL + list(NEWBL):
        p = f"{IS}/baselines/outputs/{tag}/{m}/{envn}_test.jsonl.gz"
        out[NEWBL.get(m, m)] = {r["id"]: r["choice"] for r in read(p) if r.get("choice") is not None}
    tp = f"{IS}/outputs/think_know/dir_{tag}_{envn}.jsonl.gz"
    if os.path.exists(tp):
        out["think"] = {r["id"]: max(r["dist"], key=r["dist"].get) for r in read(tp) if r.get("dist")}
    return out


def main():
    res = collections.defaultdict(dict)
    specs = [("tab", ds, f"{IS}/outputs/test_tab/fk_*_{ds}.jsonl.gz", f"{IS}/envs_tab/data/{ds}_test.jsonl.gz") for ds in ["taiwan", "german", "cdc", "heart"]]
    specs += [("sec", tg, f"{IS}/outputs/test_real/fr_*_{tg}.jsonl.gz", f"{IS}/envs_real/data/{tg}_test.jsonl.gz") for tg in ["ret_1", "ret_5", "vol_5", "vol_20"]]
    for kind, envn, pat, ep in specs:
        envl = read(ep); env = {s["id"]: s for s in envl}; first300 = [s["id"] for s in envl[:300]]
        for f in glob.glob(pat):
            tag = os.path.basename(f)[3:-9][: -(len(envn) + 1)]
            fr = {r["id"]: r for r in read(f)}
            dd = f"{IS}/outputs/test_tab/dir_{tag}_{envn}.jsonl.gz" if kind == "tab" else f"{IS}/outputs/test_real/dir_{tag}_{envn}.jsonl.gz"
            dr = {r["id"]: max(r["dist"], key=r["dist"].get) for r in read(dd) if r.get("dist")}
            bl = bl_rows(tag, envn)
            ex = EX.get(f"{tag}:{envn}", {})
            lam, tau = ex.get("lam", 0.1), ex.get("tau", 0.02)

            def choices(i):
                sc = env[i]
                if kind == "tab":
                    fs = diva_scores(sc, fr[i], "cond", "greedy", "elicit", 0.75)
                else:
                    fs = fio_real(sc, fr[i], lam=0.75)
                fu = diva_u(sc, fr[i], 1.0) if kind == "tab" else fio_real_u(sc, fr[i], 1.0)
                out = {"diva": choose_max(fs["net"]), "diva_u": choose_max(fu["net"]), "never": "S", "fac_eig": choose_max(fs["eig"], 0.05),
                       "always_top": max(fs["net"], key=fs["net"].get)}
                out["fac_eig_cost"] = eig_cost_choice(sc, fs["eig"], lam, tau)
                from episodes_know import spec_dp_first
                g_, l_ = float(sc["g"]), float(sc["l"])
                if kind == "tab":
                    par_ = {k: ([1 - a1, a1], [1 - a0, a0], c) for k, (a1, a0, c) in fs["par"].items()}
                    b_ = fs["b"]
                else:
                    par_ = fs["par"]; b_ = fs["b0"]
                out["diva_dp"] = spec_dp_first(b_, par_, g_, l_)
                kap = KAP.get(f"{tag}:{envn}", {}).get("kappa")
                if kap is not None:
                    out["diva_kappa"] = kappa_choice(sc, fs, kap)
                bet = KAP.get(f"{tag}:{envn}", {}).get("beta")
                if bet is not None:
                    out["diva_beta"] = beta_choice(sc, fs, bet)
                if i in dr:
                    out["direct"] = dr[i]
                for m, v in bl.items():
                    if i in v:
                        out[m] = v[i]
                return out

            full_ids = [i for i in env if i in fr]
            if len(full_ids) < 200:
                continue
            C = {i: choices(i) for i in full_ids}
            rec = {"n_full": len(full_ids), "fac_eig_tuned": f"{tag}:{envn}" in EX and EX[f"{tag}:{envn}"].get("tuned", True), "kappa_tuned": f"{tag}:{envn}" in KAP and "beta" in KAP.get(f"{tag}:{envn}", {})}
            pos = {s_["id"]: k for k, s_ in enumerate(envl)}
            f1000 = [i for i in full_ids if pos[i] >= 1000] if kind == "sec" else []
            for m in ["diva", "diva_u", "diva_kappa", "diva_beta", "never", "direct", "fac_eig_cost", "always_top"]:
                ids = [i for i in f1000 if m in C[i] and "direct" in C[i]]
                if len(ids) >= 100:
                    rec[f"f1000:{m}"] = summarize([score(env[i], C[i][m]) for i in ids])
            for m in ["diva", "diva_u", "diva_dp", "diva_kappa", "diva_beta", "never", "fac_eig", "fac_eig_cost", "direct", "always_top"]:
                ids = [i for i in full_ids if m in C[i] and "direct" in C[i]]  # identical rows for the main comparison
                if ids:
                    rec[f"full:{m}"] = summarize([score(env[i], C[i][m]) for i in ids])
            sub = [i for i in first300 if i in C]
            # identical rows for every method of this LLM: rows where all core methods have a choice
            CORE = ["direct"] + BL + list(NEWBL.values())
            ids_all = [i for i in sub if all(m in C[i] for m in CORE)]
            ok_all = len(ids_all) >= 0.95 * len(first300)
            ids_core = [i for i in sub if all(m in C[i] for m in ["direct"] + BL)]
            for m in ["diva", "diva_u", "diva_dp", "diva_kappa", "diva_beta", "never", "direct", "fac_eig_cost", "always_top"] + BL + list(NEWBL.values()) + ["think"]:
                ids = ids_all if ok_all else ids_core
                ids = [i for i in ids if m in C[i]]
                if len(ids) >= 0.95 * len(first300):
                    rec[f"s300:{m}"] = summarize([score(env[i], C[i][m]) for i in ids])
            rec["s300_rows"] = len(ids_all) if ok_all else len(ids_core)
            os.makedirs(f"{IS}/results/know_choices", exist_ok=True)
            json.dump({i: C[i] for i in sub}, open(f"{IS}/results/know_choices/{envn}_{tag}.json", "w"))
            res[f"{kind}:{envn}"][tag] = rec
    json.dump(res, open(f"{IS}/results/know_main.json", "w"), indent=1)
    for e, d in res.items():
        for tag, rec in sorted(d.items()):
            print(e, tag, rec["n_full"], " ".join(f"{k.split(':')[1]}:{v['dp_regret']:.4f}" for k, v in rec.items() if k.startswith("full:")),
                  "|", " ".join(f"{k.split(':')[1]}:{v['dp_regret']:.4f}" for k, v in rec.items() if k.startswith("s300:")))


if __name__ == "__main__":
    main()
