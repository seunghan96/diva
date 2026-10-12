"""Aggregate the multi-state (ms), knowledge-required tabular (tab) and SEC (real) test tracks.
usage: agg_other.py ms|tab|real [--q Q --lam L --form F]   (tab/real use the frozen PREREG_KNOW config)
Writes results/{track}_main.json: {tag: {env: {method: summary}}}."""
import sys, os, json, glob, collections, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read, score, summarize, choose_max


def ms(folder="test_ms", dp=False):
    from eval_ms import build, entropy_search, LAST_SPEC
    res = {}
    for f in glob.glob(f"{IS}/outputs/{folder}/ms_*.jsonl.gz"):
        tag, dom = os.path.basename(f)[3:-9].rsplit("_", 1)
        env = {s["id"]: s for s in read(f"{IS}/envs_synth/data/ms/{dom}_test.jsonl.gz")}
        fr = {r["id"]: r for r in read(f)}
        dr = {r["id"]: r for r in read(f"{IS}/outputs/{folder}/dir_{tag}_{dom}.jsonl.gz")}
        R = collections.defaultdict(list)
        for i, sc in env.items():
            if i not in fr:
                continue
            M, _ = build(sc, fr[i]); esv = entropy_search(sc, LAST_SPEC[0]); Me, _ = build(sc, fr[i], stated_utility=False); labs = [c["label"] for c in sc["checks"]]
            net = {}; eig = {}
            for j, lab in enumerate(labs):
                st = M.item_stats([], j); net[lab] = st[0] - M.cost[j]; eig[lab] = st[1]
            R["diva"].append(score(sc, choose_max(net)))
            if dp:
                d = M.dp_first([]); R["diva_dp"].append(score(sc, "S" if d["move"] == "S" else labs[d["item"]]))
            ne = {lab: Me.item_stats([], j)[0] - Me.cost[j] for j, lab in enumerate(labs)}
            R["diva_elicited_payoff"].append(score(sc, choose_max(ne)))
            R["fac_eig"].append(score(sc, choose_max(eig, 0.05)))
            R["fac_es"].append(score(sc, choose_max(esv, 0.05)))
            if i in dr and dr[i].get("dist"):
                R["direct"].append(score(sc, max(dr[i]["dist"], key=dr[i]["dist"].get)))
            R["always_stop"].append(score(sc, "S"))
        res.setdefault(tag, {})[dom] = {m: summarize(v) for m, v in R.items()}
    return res


def know(track, q, lam, form, minrow=0):
    from eval_tab import diva_robust as tab_rob, diva_scores
    from eval_real import fio_real_robust, fio_real
    res = {}
    pat = {"tab": f"{IS}/outputs/test_tab/fk_*.jsonl.gz", "real": f"{IS}/outputs/test_real/fr_*.jsonl.gz",
           "realj": f"{IS}/outputs/test_real_joint/fr_*.jsonl.gz"}[track]
    for f in glob.glob(pat):
        b = os.path.basename(f)[3:-9]
        if track == "tab":
            tag, envn = b.rsplit("_", 1); ep = f"{IS}/envs_tab/data/{envn}_test.jsonl.gz"
            dp = f"{IS}/outputs/test_tab/dir_{tag}_{envn}.jsonl.gz"
        else:
            envn = next(t for t in ("ret_1", "ret_5", "vol_5", "vol_20") if b.endswith(t)); tag = b[: -len(envn) - 1]
            ep = f"{IS}/envs_real/data/{envn}_test.jsonl.gz"; dp = f"{IS}/outputs/test_real/dir_{tag}_{envn}.jsonl.gz"
        env = {s["id"]: s for s in read(ep)}; fr = {r["id"]: r for r in read(f)}; dr = {r["id"]: r for r in read(dp)}
        if minrow:
            env = dict(list(env.items())[minrow:])
        sing = {r["id"]: r for r in read(f"{IS}/outputs/test_real/fr_{tag}_{envn}.jsonl.gz")} if track == "realj" else {}
        R = collections.defaultdict(list)
        for i, sc in env.items():
            if i not in fr:
                continue
            if track == "tab":
                fp = diva_scores(sc, fr[i], form, "greedy", "elicit", lam)          # frozen primary (PREREG_KNOW)
                chr_, _ = tab_rob(sc, fr[i], form, q, lam)                           # robust variant
                f0 = diva_scores(sc, fr[i], form, "greedy", "elicit", 1.0)          # no shrinkage
                fl = diva_scores(sc, fr[i], "lik", "greedy", "elicit", 1.0)         # controlled-track config
            else:
                fp = fio_real(sc, fr[i], lam=lam)
                chr_, _ = fio_real_robust(sc, fr[i], q, 5, lam)
                f0 = fio_real(sc, fr[i]); fl = f0
                if track == "realj":
                    single = sing.get(i)
                    if single is not None:
                        R["diva_single"].append(score(sc, choose_max(fio_real(sc, single, lam=0.75)["net"])))
            R["diva"].append(score(sc, choose_max(fp["net"])))
            R["diva_robust"].append(score(sc, chr_))
            R["diva_noshrink"].append(score(sc, choose_max(f0["net"])))
            R["diva_likform"].append(score(sc, choose_max(fl["net"])))
            R["fac_eig"].append(score(sc, choose_max(f0["eig"], 0.05)))
            if i in dr and dr[i].get("dist"):
                R["direct"].append(score(sc, max(dr[i]["dist"], key=dr[i]["dist"].get)))
            R["always_stop"].append(score(sc, "S"))
            R["oracle"].append(score(sc, sc["exact"]["gold_choice"]))
        R.pop("_row", None)
        res.setdefault(tag, {})[envn] = {m: summarize(v) for m, v in R.items()}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("track")
    ap.add_argument("--q", type=float, default=0.5); ap.add_argument("--lam", type=float, default=0.75)
    ap.add_argument("--form", default="cond"); ap.add_argument("--folder", default="test_ms"); ap.add_argument("--minrow", type=int, default=0)
    a = ap.parse_args()
    r = ms(a.folder) if a.track == "ms" else know(a.track, a.q, a.lam, a.form, a.minrow)
    json.dump(r, open(f"{IS}/results/{a.track}_{a.folder if a.track == 'ms' else 'main'}{'_from%d' % a.minrow if a.minrow else ''}.json", "w"), indent=1)
    for tag in sorted(r):
        for e, row in sorted(r[tag].items()):
            print(tag, e, " ".join(f"{m}:{s.get('dp_regret', s['norm_regret']):.4f}" for m, s in sorted(row.items())), "n", row.get("diva", {}).get("n"))
