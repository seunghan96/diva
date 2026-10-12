"""VoI-Tool: the LLM is given a correct value-of-information calculator (additional analyses: a fair solver baseline).

The prompt is the scenario, the options, and the source code of voi_net() below. The LLM fills in the numbers it believes:
its current probability of the adverse case given everything known, and for each available check the probability of each of
its results under the adverse and the benign case. We run voi_net() with these numbers and the STATED payoffs (g, l) and
costs, and take the check with the largest positive net value, or stop if none is positive. Unlike DIVA, the numbers come
from one answer of the LLM (no independent short questions, no coherent imagination by Bayes' rule over a specification, no
shrinkage). Binned checks (e.g. SEC terciles) use one probability per result. One generation call per step; parse failures
are retried (3 tries, as every baseline) and then fall back to stopping, logged as info.fail."""
import json

from . import common as C
from .common import STOP
from .policies import Policy, _retry_gen, _stop_out

TOOL_SRC = '''def voi_net(p_adverse, checks, g, l):
    """p_adverse: your probability that the adverse case holds, given everything known now.
    checks: {letter: {"cost": c, "p_if_adverse": [P(result k | adverse) for each result k],
                      "p_if_benign": [P(result k | benign) for each result k]}}
    g: gain of acting when the case is benign; l: loss of acting when it is adverse (passing pays 0).
    Returns {letter: expected value of deciding after seeing the check, minus its cost, minus the value of
    deciding now} -- positive means the check is worth its cost."""
    best = lambda b: max((1 - b) * g - b * l, 0.0)       # value of the best action at belief b
    out = {}
    for k, c in checks.items():
        ev = 0.0
        for pa, pb in zip(c["p_if_adverse"], c["p_if_benign"]):
            pr = p_adverse * pa + (1 - p_adverse) * pb   # probability of this result
            if pr > 0:
                ev += pr * best(p_adverse * pa / pr)     # Bayes' rule, then the best action
        out[k] = ev - c["cost"] - best(p_adverse)
    return out'''


def voi_net(p, checks, g, l):
    best = lambda b: max((1 - b) * g - b * l, 0.0)
    out = {}
    for k, c in checks.items():
        ev = 0.0
        for pa, pb in zip(c["p_if_adverse"], c["p_if_benign"]):
            pr = p * pa + (1 - p) * pb
            if pr > 0:
                ev += pr * best(p * pa / pr)
        out[k] = ev - c["cost"] - best(p)
    return out


def _prob(x):
    v = C.to_float(x)
    if v is None:
        return None
    if v > 1.0 and v <= 100.0:  # percentages
        v /= 100.0
    return min(1.0, max(0.0, v))


def _plist(x, n):
    if not isinstance(x, list) or len(x) != n:
        return None
    v = [_prob(a) for a in x]
    if any(a is None for a in v):
        return None
    s = sum(v)
    if s <= 0:
        return None
    return [a / s for a in v]  # normalize (the LLM's numbers rarely sum to exactly 1)


class VoITool(Policy):
    name = "voi_tool"

    def __init__(self, max_tokens=1500):
        self.max_tokens = max_tokens

    async def step(self, llm, st):
        rem = st.remaining
        if not rem:
            return _stop_out()
        scn = st.scn
        d = C.domain_info(scn)
        g, l = float(scn["g"]), float(scn["l"])
        opts, d2o = C.options_text(st, include_stop=True)
        o2d = {v: k for k, v in d2o.items()}
        lines, nout = [], {}
        for lab in rem:
            outs = C.outcomes(scn, lab)
            if C.is_binary_finding(scn, lab) or outs == ["no finding", "finding"]:
                outs = ["no (the reported condition does not hold)", "yes (it holds)"]
            nout[lab] = len(outs)
            lines.append(f"({o2d[lab]}) results, in this order: " + "; ".join(f"[{i}] {o}" for i, o in enumerate(outs)))
        ex = ", ".join(f"\"{o2d[lab]}\": {{\"p_if_adverse\": [{', '.join(['...'] * nout[lab])}], \"p_if_benign\": "
                       f"[{', '.join(['...'] * nout[lab])}]}}" for lab in rem[:2])
        u = (f"{C.evidence_text(st)}\n\n{opts}\n\nYou have a correct calculator for the value of information of each check:\n"
             f"```python\n{TOOL_SRC}\n```\nThe calculator is run with the payoffs and costs stated above (g = {g:g}, l = {l:g}), "
             f"and the check with the largest positive value is run next; if no value is positive, you stop and decide now.\n"
             f"Fill in the numbers you believe. The adverse case is that {d['adverse']}.\n" + "\n".join(lines) +
             f"\n\nOutput only STRICT JSON {{\"p_adverse\": ..., \"checks\": {{{ex}{', ...' if len(rem) > 2 else ''}}}}} with "
             f"every check letter above and one probability per result, in the order listed.")

        def parse(t):
            o = C.parse_json(t)
            if not o or not isinstance(o.get("checks"), dict):
                return None
            p = _prob(o.get("p_adverse"))
            if p is None:
                return None
            ck = {str(k).strip().strip("()").upper(): v for k, v in o["checks"].items()}
            out = {}
            for lab in rem:
                c = ck.get(o2d[lab])
                if not isinstance(c, dict):
                    return None
                pa, pb = _plist(c.get("p_if_adverse"), nout[lab]), _plist(c.get("p_if_benign"), nout[lab])
                if pa is None or pb is None:
                    return None
                out[lab] = {"cost": st.cost(lab), "p_if_adverse": pa, "p_if_benign": pb}
            return p, out

        r, raws, ok = await _retry_gen(llm, C.system_prompt(scn), u, parse, max_tokens=self.max_tokens, tag="voi_tool",
                                       ctx=self.ctx(st))
        if not ok:
            return _stop_out({"fail": "parse"})
        p, checks = r
        net = voi_net(p, checks, g, l)
        best = max(rem, key=lambda k: net[k])
        choice = best if net[best] > 1e-12 else STOP
        return {"choice": choice, "scores": dict(net, **{STOP: 0.0}),
                "info": {"p_adverse": p, "spec": {k: [v["p_if_adverse"], v["p_if_benign"]] for k, v in checks.items()}}}
