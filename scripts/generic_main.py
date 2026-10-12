"""LLM-free generic specification for the MAIN tables: rows "Generic DIVA" / "Generic DIVA-U" in
Table tab:know (first-decision DP regret, first 300 test scenarios) and Table tab:episodes (episode regret).
Generic specification: every check (and known finding) has the same strength delta in log-odds, the direction is the
designer-set one (a finding raises the risk; controlled: the stated direction), the base rate is the stated one, and the
DIVA / DIVA-U solver of the paper is applied (DIVA-U: sigma = 1, 3-point Gauss-Hermite, as in the paper).
delta: ONE value per variant and environment group, selected on development data (knowledge-required: results/const_fair.json,
mean development DP regret over the eight environments; controlled: mean development DP regret over the four domains, first
1,000 development scenarios each). The value is LLM-independent.
Episodes: knowledge-required = scripts/episodes_know.py episode_value (test rows 0-149, reference-posterior final action);
controlled = baselines.episode with common random worlds on the first 100 test scenarios (reps 0-3), averaged per LLM over
exactly the (scenario, world) pairs of that LLM's baseline episodes (as DIVA@bl in results/episodes.json), then over the
fixed controlled episode LLM set. Writes results/generic_main.json."""
import sys, os, json, math, glob, asyncio, collections
from multiprocessing import Pool
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, IS)
from evalx import read, score, choose_max
from dcore import posterior, voi_exact
from fiovoi_policy import FIOVOI, clip

GRID = [0.25, 0.5, 1.0, 2.0, 3.0]
DOMS = ["finance", "credit", "clinical", "legal"]
KENVS = ["taiwan", "german", "cdc", "heart", "ret_1", "ret_5", "vol_5", "vol_20"]
GH = [(-math.sqrt(3), 1 / 6), (0.0, 2 / 3), (math.sqrt(3), 1 / 6)]
SIGMA = 1.0
sg = lambda x: 1 / (1 + math.exp(-x)); lg = lambda p: math.log(p / (1 - p))


def gfac(sc, d):
    """generic factored specification of a controlled scenario: s1 = P(finding | adverse), s0 = P(finding | benign)"""
    def pair(x):
        up = float(x["s1"]) >= float(x["s0"])  # designer-set direction (the stated one; placebos count as risk-raising)
        return [sg(d / 2), sg(-d / 2)] if up else [sg(-d / 2), sg(d / 2)]
    return {"prior": float(sc["prior"]), "obs": [pair(o) for o in sc.get("observed", [])], "chk": {c["label"]: pair(c) for c in sc["checks"]}}


def nets(b, rem, g, l, unc):
    """net value of each remaining check; unc: average over log-odds shifts of the check's strength (DIVA-U)"""
    out = {}
    for k, (s1, s0, c) in rem.items():
        if not unc:
            out[k] = voi_exact(b, s1, s0, g, l) - c
        else:
            out[k] = sum(w * voi_exact(b, clip(sg(lg(s1) + SIGMA * z / 2)), clip(sg(lg(s0) - SIGMA * z / 2)), g, l) for z, w in GH) - c
    return out


def first_choice(sc, d, unc):
    fr = gfac(sc, d)
    obs = [(clip(o[0]), clip(o[1]), int(ob["value"])) for o, ob in zip(fr["obs"], sc.get("observed", []))]
    b = posterior(clip(fr["prior"]), obs)
    rem = {c["label"]: (clip(fr["chk"][c["label"]][0]), clip(fr["chk"][c["label"]][1]), float(c["cost"])) for c in sc["checks"]}
    return choose_max(nets(b, rem, float(sc["g"]), float(sc["l"]), unc))


def ctrl_first(rows, d, unc):
    return float(np.mean([score(sc, first_choice(sc, d, unc))["dp_regret"] for sc in rows]))


class GenPolicy(FIOVOI):
    """generic specification as an episode policy (myopic DIVA / DIVA-U)"""
    def __init__(self, fac, unc, name):
        super().__init__(fac, "myopic", name=name); self.unc = unc

    async def step(self, llm, st):
        b, rem = self._model(st)
        if not rem:
            return {"choice": "S", "scores": {}}
        v = nets(b, rem, float(st.scn["g"]), float(st.scn["l"]), self.unc)
        k = max(v, key=v.get)
        return {"choice": k if v[k] > 0 else "S", "scores": v, "info": {"b": b}}


def ctrl_episodes(arg):
    dom, dd, du = arg
    from baselines import episode as E
    env = read(f"{IS}/envs_synth/data/binary/{dom}_test.jsonl.gz")[:100]
    pols = [GenPolicy({s["id"]: gfac(s, dd) for s in env}, False, "gen_diva"), GenPolicy({s["id"]: gfac(s, du) for s in env}, True, "gen_u")]

    async def run():
        out = {}
        for sc in env:
            for rep in range(4):
                for p in pols:
                    r = await E.episode(p, None, sc, rep=rep)
                    out[f"{p.name}|{sc['id']}|{rep}"] = r["eval"]["policy_regret_norm"]
        return out
    return dom, asyncio.run(run())


def know_episode(arg):
    e, var, d = arg
    import episodes_know as EK
    kind = "tab" if e in ("taiwan", "german", "cdc", "heart") else "sec"
    EK.CONST = {e: {"delta": d, "delta_u": d}}; EK.CUR["env"] = e; EK.CUR["tag"] = "generic"
    p = f"{IS}/envs_tab/data/{e}_test.jsonl.gz" if kind == "tab" else f"{IS}/envs_real/data/{e}_test.jsonl.gz"
    vals = []
    for sc in read(p)[:150]:
        _, rr, _ = EK.episode_value(kind, var, sc, {"prior": [None], "obs": [], "chk": {}}, None)
        vals.append(rr)
    return e, var, float(np.mean(vals)), len(vals)


def main():
    out = {"know_first": {}, "know_ep": {}, "ctrl_first": {}, "ctrl_ep": {}}
    # knowledge-required, first decision: const_fair.json (one delta per variant, selected on development data)
    C = json.load(open(f"{IS}/results/const_fair.json")); sel = C["_selected"]
    out["delta_know"] = sel
    for var in ("diva", "diva_u"):
        out["know_first"][var] = {e: C[e][f"s300:{var}:{sel[var]}"]["dp_regret"] for e in KENVS}
    # controlled, first decision: one delta per variant selected on the development splits of the four domains
    dev = {d: read(f"{IS}/envs_synth/data/binary/{d}_dev.jsonl.gz")[:1000] for d in DOMS}
    test = {d: read(f"{IS}/envs_synth/data/binary/{d}_test.jsonl.gz")[:300] for d in DOMS}
    dsel = {}
    for var, unc in (("diva", False), ("diva_u", True)):
        devr = {x: float(np.mean([ctrl_first(dev[d], x, unc) for d in DOMS])) for x in GRID}
        dsel[var] = min(GRID, key=devr.get); out.setdefault("ctrl_dev", {})[var] = devr
        out["ctrl_first"][var] = {d: ctrl_first(test[d], dsel[var], unc) for d in DOMS}
    out["delta_ctrl"] = dsel
    with Pool(32) as pool:
        # knowledge-required episodes (rows 0-149, reference-posterior final action), LLM-independent
        for e, var, v, n in pool.imap_unordered(know_episode, [(e, v, sel["diva" if v == "gen_diva" else "diva_u"]) for e in KENVS for v in ("gen_diva", "gen_u")]):
            out["know_ep"].setdefault("diva" if var == "gen_diva" else "diva_u", {})[e] = v
            print("know ep", e, var, round(100 * v, 2), n, flush=True)
        # controlled episodes (common random worlds), matched per LLM to the pairs of its baseline episodes
        R = dict(pool.imap_unordered(ctrl_episodes, [(d, dsel["diva"], dsel["diva_u"]) for d in DOMS]))
    from complete import EP_CTRL_LLMS
    for var, key in (("diva", "gen_diva"), ("diva_u", "gen_u")):
        per = {}
        for d in DOMS:
            vals = []
            for t in EP_CTRL_LLMS:
                pairs = None
                for g in glob.glob(f"{IS}/baselines/outputs/{t}/*/{d}_test.episode.jsonl.gz") + glob.glob(f"{IS}/baselines/outputs_palfix/{t}/*/{d}_test.episode.jsonl.gz"):
                    if g.split("/")[-2] == "pal" and "/outputs/" in g:
                        continue
                    pr = {(r["id"], r.get("rep", 0)) for r in read(g) if r.get("eval")}
                    pairs = pr if pairs is None else (pairs & pr)
                v = [R[d][f"{key}|{i}|{rep}"] for i, rep in (pairs or set()) if f"{key}|{i}|{rep}" in R[d]]
                vals.append(float(np.mean(v)) if len(v) >= 200 else None)
            per[d] = float(np.mean(vals)) if all(x is not None for x in vals) else None
        out["ctrl_ep"][var] = per
    json.dump(out, open(f"{IS}/results/generic_main.json", "w"), indent=1)
    for k in ("know_first", "ctrl_first", "know_ep", "ctrl_ep"):
        print(k, {v: {e: (None if x is None else round(100 * x, 2)) for e, x in d.items()} for v, d in out[k].items()})
    print("delta know", sel, "delta ctrl", dsel)


if __name__ == "__main__":
    main()
