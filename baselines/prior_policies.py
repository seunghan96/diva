"""Closest prior work run as acquisition baselines (reviewer gap M4), behind the same policy interface as policies.py.

DeLLMa (Liu et al., 2025; `dellma`), adapted to the acquire-or-stop decision:
  1) state forecasting: the LLM rates each state of the latent case (adverse / benign) on DeLLMa's six-point verbal
     scale (very unlikely ... very likely); the ratings are mapped to 1..6 and normalized into a forecast P(y);
  2) the action set is the remaining checks (each "run the check, then decide") and stopping to decide now;
  3) utility elicitation by ranking: the LLM ranks all state-action pairs (2 states x options) from most to least
     preferred, twice with two shuffled pair orders (DeLLMa's overlapping minibatches); a Bradley-Terry model fitted
     to the implied pairwise preferences gives the utility u(y, a);
  4) the option with the largest expected utility sum_y P(y) u(y, a) is taken (stopping when it is "stop").
  Payoffs and costs are stated in the scenario text, as for every other baseline. 3 generation calls per step.

UoT, Uncertainty of Thoughts (Hu et al., 2024; `uot`), adapted to candidate checks with a stop option:
  1) possibility set: the LLM generates N=8 distinct, plausible complete descriptions of the case consistent with the
     evidence, each labelled with whether the adverse case holds;
  2) answer simulation: per candidate check, the LLM states the result the check would show under every possibility,
     which splits the set (UoT's yes/no split; one group per outcome for binned checks);
  3) uncertainty-based reward: with a uniform weight on the remaining possibilities, the reward of a check is the
     entropy of its split (expected information gain in bits), accumulated over a two-level simulation tree
     (UoT's reward propagation: R(q) = IG(q) + sum_o p(o) max_q' IG(q' | split o));
  4) UoT answers once the possibilities agree on the answer: it stops when every remaining possibility has the same
     label, or when the best accumulated reward is below tau = 0.05 bits (the threshold of the BED-LLM baseline).
  Costs are not part of UoT's reward (as in the original method). 1 + |checks| generation calls per step.

BIRD (Feng et al., 2025) is not run: it fits factor weights per question by constrained optimization over many sampled
LLM judgments and outputs a calibrated probability, with no acquisition or stopping rule to adapt.
"""
import asyncio, json, math, random, re

import numpy as np

from . import common as C
from .common import STOP, H
from .policies import Policy, _retry_gen, _stop_out, _seed, _argmax

VERBAL = ["very unlikely", "unlikely", "somewhat unlikely", "somewhat likely", "likely", "very likely"]


def _verbal_score(v):
    if v is None:
        return None
    s = str(v).strip().lower().strip(".\"'")
    best = None  # longest contained label wins ("somewhat unlikely" over "unlikely")
    for i, lab in enumerate(VERBAL):
        if s == lab:
            return i + 1
        if lab in s and (best is None or len(lab) > len(VERBAL[best - 1])):
            best = i + 1
    return best


def bradley_terry(n, prefs, iters=300, l2=0.05, lr=0.5):
    """MAP utilities of n items from (winner, loser) pairs (logistic BT with a small ridge); items never compared get 0."""
    u = np.zeros(n)
    if not prefs:
        return u
    W = np.array([p[0] for p in prefs]); L = np.array([p[1] for p in prefs])
    for _ in range(iters):
        d = u[W] - u[L]
        g = 1 / (1 + np.exp(d))  # d log sigma(d) / d d
        grad = np.zeros(n)
        np.add.at(grad, W, g); np.add.at(grad, L, -g)
        grad -= l2 * u
        u += lr * grad / max(1, len(prefs) / n)
    return u - u.mean()


class DeLLMa(Policy):
    name = "dellma"

    def __init__(self, max_tokens=1024, n_rank=2):
        self.max_tokens, self.n_rank = max_tokens, n_rank

    async def _forecast(self, llm, st):
        d = C.domain_info(st.scn)
        a_s, b_s = d["adverse_short"], d["benign_short"]
        u = (f"{C.evidence_text(st)}\n\nState forecasting. The unknown state is whether {d['adverse']}. Rate the likelihood "
             f"of each possible state with one of the labels {', '.join(VERBAL)}. Return STRICT JSON "
             f"{{\"{a_s}\": \"<label>\", \"{b_s}\": \"<label>\"}}. Output only the JSON.")

        def parse(t):
            o = C.parse_json(t)
            if not o:
                return None
            lo = {str(k).strip().lower(): v for k, v in o.items()}
            sa, sb = _verbal_score(lo.get(a_s.lower())), _verbal_score(lo.get(b_s.lower()))
            if sa is None or sb is None:
                return None
            return sa / (sa + sb)

        p, raws, ok = await _retry_gen(llm, C.system_prompt(st.scn), u, parse, max_tokens=self.max_tokens,
                                       tag="dellma_forecast", ctx=self.ctx(st))
        return p

    async def step(self, llm, st):
        rem = st.remaining
        if not rem:
            return _stop_out()
        scn = st.scn
        d = C.domain_info(scn)
        p_adv = await self._forecast(llm, st)
        if p_adv is None:
            return _stop_out({"fail": "forecast_parse"})
        opts, d2o = C.options_text(st, include_stop=True)
        o2d = {v: k for k, v in d2o.items()}
        options = rem + [STOP]
        states = [("adverse", f"it turns out that {d['adverse']}"), ("benign", f"it turns out that it is NOT the case that {d['adverse']}")]
        pairs = [(y, o) for y in range(2) for o in options]
        rows, prefs, info = [], [], {"p_adverse": p_adv, "rankings": []}
        calls = []
        for r in range(self.n_rank):
            order = list(range(len(pairs)))
            random.Random(_seed("dellma", scn.get("id"), len(st.acquired), r)).shuffle(order)
            ids = {f"P{k + 1}": pi for k, pi in enumerate(order)}
            lines = []
            for pid, pi in ids.items():
                y, o = pairs[pi]
                act = "stop and decide now" if o == STOP else f"run check ({o2d[o]}) and then decide"
                lines.append(f"{pid}: state: {states[y][1]}; action: {act}")
            u = (f"{C.evidence_text(st)}\n\n{opts}\n\nUtility elicitation. Each pair below combines one possible state of the "
                 f"world with one option above. Rank ALL pairs from the most preferred to the least preferred outcome for you, "
                 f"considering the payoffs, the costs of the checks, and what you would learn and decide after a check.\n"
                 + "\n".join(lines) +
                 f"\n\nThink briefly, then end with STRICT JSON {{\"ranking\": [\"P..\", ...]}} listing every pair id once, "
                 f"most preferred first.")

            def parse(t, ids=ids):
                o = C.parse_json(t)
                if not o or not isinstance(o.get("ranking"), list):
                    return None
                seen = []
                for x in o["ranking"]:
                    k = str(x).strip().upper().strip("\"' ")
                    if k in ids and ids[k] not in seen:
                        seen.append(ids[k])
                return seen if len(seen) >= max(2, len(ids) // 2) else None

            calls.append(_retry_gen(llm, C.system_prompt(scn), u, parse, max_tokens=self.max_tokens,
                                    tag="dellma_rank", ctx=self.ctx(st, r=r)))
        for rk, raws, ok in await asyncio.gather(*calls):
            if not ok:
                continue
            info["rankings"].append(rk)
            for i in range(len(rk)):
                for j in range(i + 1, len(rk)):
                    prefs.append((rk[i], rk[j]))
        if not info["rankings"]:
            return _stop_out(dict(info, fail="rank_parse"))
        util = bradley_terry(len(pairs), prefs)
        scores = {}
        for k, o in enumerate(options):
            ua, ub = util[pairs.index((0, o))], util[pairs.index((1, o))]
            scores[o] = float(p_adv * ua + (1 - p_adv) * ub)
        info["util"] = {f"{'adv' if y == 0 else 'ben'}:{o}": float(util[i]) for i, (y, o) in enumerate(pairs)}
        best = max(options, key=lambda o: (scores[o], o == STOP))  # ties -> stop
        return {"choice": best, "scores": scores, "info": info}


class UoT(Policy):
    name = "uot"

    def __init__(self, n_hyp=8, tau=0.05, max_tokens=1024):
        self.n_hyp, self.tau, self.max_tokens = n_hyp, tau, max_tokens

    async def _hypotheses(self, llm, st):
        d = C.domain_info(st.scn)
        n = self.n_hyp
        u = (f"{C.evidence_text(st)}\n\nList {n} distinct, plausible, complete descriptions of this case (possible "
             f"explanations of the situation) that are consistent with everything known so far. Each description is one "
             f"short sentence and is labelled \"adverse\" if {d['adverse']}, otherwise \"benign\". Return STRICT JSON "
             f"{{\"possibilities\": [{{\"description\": \"...\", \"label\": \"adverse\" or \"benign\"}}, ...]}} with "
             f"exactly {n} entries.")

        def parse(t):
            o = C.parse_json(t)
            if not o or not isinstance(o.get("possibilities"), list):
                return None
            out = []
            for x in o["possibilities"]:
                if not isinstance(x, dict):
                    continue
                desc = str(x.get("description", "")).strip()
                lab = str(x.get("label", "")).strip().lower()
                if desc and lab in ("adverse", "benign"):
                    out.append((desc[:300], int(lab == "adverse")))
            return out[:n] if len(out) >= 2 else None

        r, _, ok = await _retry_gen(llm, C.system_prompt(st.scn), u, parse, max_tokens=self.max_tokens,
                                    tag="uot_hyp", ctx=self.ctx(st))
        return r

    async def _simulate(self, llm, st, lab, hyps):
        scn = st.scn
        c = st.checks[lab]
        if C.is_binary_finding(scn, lab):
            outs = ["no finding", "finding"]
            q = (f"For each possibility, would {c['name']} find that {c['finding_text']}? Answer \"finding\" or "
                 f"\"no finding\" for every possibility.")
        else:
            outs = C.outcomes(scn, lab)
            q = (f"For each possibility, which result would {c['name']} report? Choose one of: "
                 + "; ".join(f'"{o}"' for o in outs) + ".")
        hl = "\n".join(f"H{k + 1}: {h[0]}" for k, h in enumerate(hyps))
        u = (f"{C.evidence_text(st)}\n\nPossibilities:\n{hl}\n\n{q} Return STRICT JSON "
             f"{{\"H1\": \"<result>\", ...}} with one entry per possibility. Output only the JSON.")
        low = [o.lower() for o in outs]

        def parse(t):
            o = C.parse_json(t)
            if not o:
                return None
            res = []
            for k in range(len(hyps)):
                v = o.get(f"H{k + 1}", o.get(f"h{k + 1}"))
                if v is None:
                    return None
                s = str(v).strip().lower().strip("\"'. ")
                if s in low:
                    res.append(low.index(s)); continue
                m = [i for i, x in enumerate(low) if x in s or s in x]
                if len(outs) == 2 and outs[0] == "no finding":
                    m = [0] if ("no" in s.split() or "not" in s or s.startswith("no")) else [1]
                if not m:
                    return None
                res.append(max(m, key=lambda i: len(low[i])))
            return res

        r, _, ok = await _retry_gen(llm, C.system_prompt(scn), u, parse, max_tokens=self.max_tokens,
                                    tag="uot_sim", ctx=self.ctx(st, e=lab))
        return r

    @staticmethod
    def _ig(idx, ans):
        """entropy (bits) of the split of possibility indices idx by the simulated answers"""
        if not idx:
            return 0.0
        cnt = {}
        for i in idx:
            cnt[ans[i]] = cnt.get(ans[i], 0) + 1
        return H([v / len(idx) for v in cnt.values()])

    async def step(self, llm, st):
        rem = st.remaining
        if not rem:
            return _stop_out()
        hyps = await self._hypotheses(llm, st)
        if hyps is None:
            return _stop_out({"fail": "hyp_parse"})
        labels = [h[1] for h in hyps]
        info = {"hyps": hyps}
        if len(set(labels)) == 1:  # the possibilities agree on the answer: UoT answers now
            sc = {k: 0.0 for k in rem}; sc[STOP] = 1.0
            return {"choice": STOP, "scores": sc, "info": dict(info, resolved=True)}
        sims = await asyncio.gather(*[self._simulate(llm, st, lab, hyps) for lab in rem])
        A = {lab: s for lab, s in zip(rem, sims) if s is not None}
        info["sim"] = A
        if not A:
            return _stop_out(dict(info, fail="sim_parse"))
        allidx = list(range(len(hyps)))
        scores = {}
        for q, ans in A.items():
            r1 = self._ig(allidx, ans)
            r2 = 0.0
            for o in set(ans):
                sub = [i for i in allidx if ans[i] == o]
                if len(set(labels[i] for i in sub)) <= 1:  # resolved branch: UoT answers there
                    continue
                best2 = max((self._ig(sub, a2) for q2, a2 in A.items() if q2 != q), default=0.0)
                r2 += len(sub) / len(allidx) * best2
            scores[q] = r1 + r2
        scores[STOP] = self.tau
        best = _argmax(scores, rem)
        choice = best if best is not None and scores[best] >= self.tau else STOP
        return {"choice": choice, "scores": scores, "info": info}


PRIOR = {"dellma": DeLLMa, "uot": UoT}
