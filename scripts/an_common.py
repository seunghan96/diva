"""Shared loader for the reviewer analyses (an_*.py): per (LLM tag, environment) the first-300 test scenarios with the
LLM's direct distribution and its elicited specification under the frozen configuration (DIVA, DIVA-U, DIVA-DP inputs).
Uses existing outputs only (no LLM calls)."""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read, fac_scores, choose_max, score, TOL
from complete import KNOW_LLMS, CDOMS, ENV_ORDER

N300 = 300
TAB = ["taiwan", "german", "cdc", "heart"]
SEC = ["ret_1", "ret_5", "vol_5", "vol_20"]


def env_rows(e):
    if e in CDOMS:
        return read(f"{IS}/envs_synth/data/binary/{e}_test.jsonl.gz")
    if e in TAB:
        return read(f"{IS}/envs_tab/data/{e}_test.jsonl.gz")
    return read(f"{IS}/envs_real/data/{e}_test.jsonl.gz")


def spec_path(e, tag):
    if e in CDOMS:
        return f"{IS}/outputs/test_bin/facnum_{tag}_{e}.jsonl.gz"
    if e in TAB:
        return f"{IS}/outputs/test_tab/fk_{tag}_{e}.jsonl.gz"
    return f"{IS}/outputs/test_real/fr_{tag}_{e}.jsonl.gz"


def dir_path(e, tag):
    if e in CDOMS:
        return f"{IS}/outputs/test_dir/dir_{tag}_{e}.jsonl.gz"
    if e in TAB:
        return f"{IS}/outputs/test_tab/dir_{tag}_{e}.jsonl.gz"
    return f"{IS}/outputs/test_real/dir_{tag}_{e}.jsonl.gz"


def spec_scores(e, sc, fr, lam=0.75, form="cond"):
    """frozen DIVA specification: dict(b, net, eig, par) with par[label] = (P(v|1) list, P(v|0) list, cost)"""
    if e in CDOMS:
        fs = fac_scores(sc, fr)
        par = {k: ([s1, 1 - s1][::-1], [s0, 1 - s0][::-1], c) for k, (s1, s0, c) in fs["par"].items()}  # [P(0|y), P(1|y)]
        return dict(b=fs["b"], net=fs["net"], eig=fs["eig"], par=par, voi=fs["voi"])
    if e in TAB:
        from eval_tab import diva_scores
        fs = diva_scores(sc, fr, form, "greedy", "elicit", lam)
        par = {k: ([1 - a1, a1], [1 - a0, a0], c) for k, (a1, a0, c) in fs["par"].items()}
        return dict(b=fs["b"], net=fs["net"], eig=fs["eig"], par=par)
    from eval_real import fio_real
    fs = fio_real(sc, fr, lam=lam)
    return dict(b=fs["b0"], net=fs["net"], eig=fs["eig"], par=fs["par"])


def spec_u(e, sc, fr):
    """DIVA-U net values (controlled: equals DIVA, as every reliability is stated)"""
    if e in CDOMS:
        return None
    if e in TAB:
        from eval_tab import diva_u
        return diva_u(sc, fr, 1.0)["net"]
    from eval_real import fio_real_u
    return fio_real_u(sc, fr, 1.0)["net"]


def load(e, tag, need_spec=True):
    """list of (scenario, spec record, direct dist) over the first 300 test scenarios; None if incomplete"""
    envl = env_rows(e)[:N300]
    fr = {r["id"]: r for r in read(spec_path(e, tag))} if need_spec else {}
    if e in CDOMS:  # Table tab:know's Direct on the controlled environments is the baseline runner (stop option rotated)
        dr = {r["id"]: r["scores"] for r in read(f"{IS}/baselines/outputs/{tag}/direct/{e}_test.jsonl.gz") if r.get("scores") and r.get("choice") is not None}
    else:
        dr = {r["id"]: r["dist"] for r in read(dir_path(e, tag)) if r.get("dist")}
    out = []
    for sc in envl:
        i = sc["id"]
        if (need_spec and i not in fr) or i not in dr:
            continue
        out.append((sc, fr.get(i), dr[i]))
    return out if len(out) >= 0.95 * len(envl) else None


def dp_stop_ok(sc):
    ex = sc["exact"]
    t = ex.get("dp_first_ties") or [ex.get("dp_first_action")]
    return any(x in (None, "S", "STOP", "stop") for x in t)


def oracle_check(sc):
    q = sc["exact"]["q"]
    return max(q, key=lambda k: float(q[k]))
