"""Canonical verbalizer for controlled scenarios (ALL methods and baselines must build prompts through this module,
so that surface text is identical across methods). Scenario dict fields: domain, prior, g, l, checks[{label,name,
finding_text,s1,s0,cost}], observed[{name,finding_text,value,s1,s0}], style (int seed for phrasing variety)."""
import random
from domains import DOMAINS

LETTERS = "ABCDEFGHIJ"


def pct(x, style):
    v = int(round(100 * x))
    r = style % 3
    if r == 0:
        return f"{v}%"
    if r == 1:
        return f"about {v}%"
    return f"{v} in 100"


def num(x):
    return (f"{x:.2f}".rstrip("0").rstrip(".")) if isinstance(x, float) else str(x)


def reliability(dom, c, style):
    D = DOMAINS[dom]
    return (f"among cases where {D['adverse']}, {pct(c['s1'], style)} show this finding; "
            f"among the other cases, {pct(c['s0'], style)} do")


def finding_sentence(c, value):
    if value:
        return f"{c['name']}: FINDING PRESENT ({c['finding_text']})."
    return f"{c['name']}: FINDING ABSENT (it is not the case that {c['finding_text']})."


def scenario_text(sc, extra_observed=None, include_checks=True, include_payoff=True):
    D = DOMAINS[sc["domain"]]
    st = sc.get("style", 0)
    lines = [f"You are {D['role']}. {D['context']}"]
    if include_payoff:
        lines.append("Payoffs: " + D["gain_text"].format(g=num(sc["g"]), l=num(sc["l"])))
    lines.append(f"Base rate: among {D['population']}, {pct(sc['prior'], st)} end up in the situation where {D['adverse']}.")
    obs = list(sc.get("observed", [])) + list(extra_observed or [])
    if obs:
        lines.append("What you already know (each check's reliability is given; checks are independent given the outcome):")
        for o in obs:
            lines.append(f"- {finding_sentence(o, o['value'])} Reliability: {reliability(sc['domain'], o, st)}.")
    else:
        lines.append("You have not run any checks yet.")
    if include_checks:
        lines.append("Checks you could still run before deciding (each reveals FINDING PRESENT or FINDING ABSENT; "
                     "costs are in the same points as the payoffs):")
        for c in sc["checks"]:
            lines.append(f"({c['label']}) {c['name']} — cost {num(c['cost'])} points. Finding: {c['finding_text']}. "
                         f"Reliability: {reliability(sc['domain'], c, st)}.")
    return "\n".join(lines)


def belief_question(sc):
    D = DOMAINS[sc["domain"]]
    return f"Question: Given only the information above, what is the probability that {D['adverse']}?"


BINS = [i / 10 for i in range(11)]


def belief_prompt(sc, extra_observed=None, descending=False):
    body = scenario_text(sc, extra_observed=extra_observed, include_checks=False, include_payoff=False)
    opts = list(enumerate(BINS))
    if descending:
        opts = opts[::-1]
    lab = "ABCDEFGHIJK"
    olines = [f"({lab[k]}) {int(round(100*p))}%" for k, (_, p) in enumerate(opts)]
    return (body + "\n\n" + belief_question(sc) + "\nOptions:\n" + "\n".join(olines) +
            "\n\nAnswer with the letter of the closest option only."), [p for _, p in opts]


def choice_prompt(sc, order=None):
    """Direct acquisition choice: which check to run next, or stop. Returns prompt and label list in display order."""
    sc2 = dict(sc)
    checks = list(sc["checks"])
    if order is not None:
        checks = [checks[i] for i in order]
    relab = []
    for k, c in enumerate(checks):
        c2 = dict(c); c2["orig_label"] = c["label"]; c2["label"] = LETTERS[k]; relab.append(c2)
    sc2["checks"] = relab
    stop = LETTERS[len(relab)]
    D = DOMAINS[sc["domain"]]
    txt = (scenario_text(sc2) + f"\n({stop}) Stop running checks and decide now ({D['act']} or {D['pass']}).\n\n"
           f"Question: You may run at most one more check before deciding. Which option do you choose?\n"
           f"Answer with the letter of one option only.")
    mapping = {c["label"]: c["orig_label"] for c in relab}
    mapping[stop] = "S"
    return txt, mapping
