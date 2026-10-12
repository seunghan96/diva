"""First-step scoring and full acquire-then-decide episodes for any Policy, scored with the env's own eval_api.

Common random numbers: episode `rep` uses the row's stored world row["episodes"][rep] (true state + every check's
outcome), the same world the the harness and eval_api.evaluate(row, requests, decision, episode=rep) use.
"""
import os, time

from . import common as C
from .common import STOP, State
from . import policies as P

_EVAL = {}


def eval_api(scn):
    tr = C.track(scn)
    key = "real" if tr == "real" else "synth"
    if key not in _EVAL:
        d = os.path.join(C.INFOSEEK, "envs_real" if key == "real" else "envs_synth")
        import sys
        if os.path.join(C.INFOSEEK, "envs_synth") not in sys.path:
            sys.path.insert(0, os.path.join(C.INFOSEEK, "envs_synth"))
        _EVAL[key] = C._load(os.path.join(d, "eval_api.py"), f"infoseek_eval_{key}")
    return _EVAL[key]


def world(scn, rep=0):
    ep = scn["episodes"][rep % len(scn["episodes"])]
    return int(ep.get("state", ep.get("y", 0))), {k: int(v) for k, v in ep["outcomes"].items()}


async def first_step(policy, llm, scn):
    st = State(scn=scn)
    t0 = time.time()
    out = await policy.step(llm, st)
    row = {"id": scn["id"], "method": policy.name, "choice": out["choice"], "scores": out["scores"],
           "info": out.get("info", {}), "sec": round(time.time() - t0, 2)}
    ex = scn.get("exact") or {}
    if ex:
        row["gold_choice"] = ex.get("gold_choice")
    if "model" in scn and "exact" in scn:
        try:
            row["rank"] = eval_api(scn).rank_metrics(scn, {k: v for k, v in out["scores"].items() if k != STOP})
        except Exception as e:
            row["rank_err"] = repr(e)[:200]
    return row


async def episode(policy, llm, scn, rep=0, budget=None, max_steps=None):
    """acquire ... STOP, then decide. Row budget always applies; `budget` can only tighten it."""
    st = State(scn=scn)
    state, vals = world(scn, rep)
    traj = []
    t0 = time.time()
    lim = st.budget_left if budget is None else min(st.budget_left, budget)
    if max_steps is not None:
        lim = min(lim, max_steps)
    while True:
        if not st.remaining or len(st.acquired) >= lim:
            traj.append({"choice": STOP, "forced": True})
            break
        out = await policy.step(llm, st)
        ch = out["choice"]
        rec = {"choice": ch, "scores": out["scores"], "info": out.get("info", {})}
        if ch != STOP and ch not in st.remaining:
            rec.update(invalid=True, choice=STOP)
            ch = STOP
        traj.append(rec)
        if ch == STOP:
            break
        st.acquired.append({"label": ch, "value": vals[ch]})
    dec = await policy.decide(llm, st)
    requests = [a["label"] for a in st.acquired]
    row = {"id": scn["id"], "method": policy.name, "rep": rep, "episode": rep, "state": state, "requests": requests,
           "acquired": st.acquired, "n_acq": len(requests), "cost": st.spent(), "decision": dec.get("action"),
           "decision_info": dec, "traj": traj, "budget": budget, "sec": round(time.time() - t0, 2)}
    if "model" in scn and "exact" in scn:
        try:
            row["eval"] = eval_api(scn).evaluate(scn, requests, dec.get("action"), rep)
        except Exception as e:
            row["eval_err"] = repr(e)[:300]
    return row
