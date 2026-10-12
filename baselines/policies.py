"""Acquisition-policy baselines behind one interface (all three tracks: binary, real, ms).

    policy.step(llm, st)   -> {"choice": label | "S", "scores": {label or "S": float}, "info": {...}}
    policy.decide(llm, st) -> {"action": action name, "probs": {name: p}, "prob": p(chosen), "belief": P(adverse)|None}

`st` is a common.State (verbalized evidence, remaining candidate checks with costs, the decision from the row).
Higher score = more preferred; scores feed the ranking metrics (Spearman with VOI, top-1).
The the IO-VOI method is intentionally NOT implemented here.
"""
import asyncio, hashlib, json, math, os, random, re

import numpy as np

from . import common as C
from .common import STOP, H, norm

_SYN = os.path.join(C.INFOSEEK, "envs_synth")
core = C._load(os.path.join(_SYN, "core.py"), "infoseek_synth_core")  # exact Model (VOI / EIG / DP) for every track
TOL = 1e-7
_MODELS = {}


def get_model(scn):
    k = scn["id"]
    if k not in _MODELS:
        if len(_MODELS) > 512:
            _MODELS.clear()
        _MODELS[k] = core.Model(scn["model"])
    return _MODELS[k]


def exact_obs(st):
    M = get_model(st.scn)
    return list(M.observed0) + [(M.lab2i[a["label"]], int(a["value"])) for a in st.acquired]


def exact_belief(st):
    """Exact posterior over latent states (vector) at the current history."""
    return get_model(st.scn).belief(exact_obs(st))


def _argmax(d, keys=None):
    keys = [k for k in (keys or d) if k in d and d[k] is not None]
    return max(keys, key=lambda k: d[k]) if keys else None


def _seed(*parts):
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _stop_out(info=None):
    return {"choice": STOP, "scores": {STOP: 1.0}, "info": info or {}}


async def _retry_gen(llm, system, user, parse, tries=3, max_tokens=1024, tag="", ctx=None, n=1):
    """Generate, parse; on failure retry with temperature 0.7 and new seeds. Returns (parsed, raw_list, ok)."""
    raws = []
    for t in range(tries):
        outs = await llm.generate(system, user, max_tokens=max_tokens, temperature=0.0 if t == 0 else 0.7,
                                  seed=None if t == 0 else 1000 + t, tag=f"{tag}#try{t}", ctx=ctx, n=n)
        raws += outs
        for o in outs:
            r = parse(o)
            if r is not None:
                return r, raws, True
    return None, raws, False


class Policy:
    name = "base"
    uses_llm = True

    async def step(self, llm, st):
        raise NotImplementedError

    async def decide(self, llm, st):
        return await C.final_decision(llm, st, ctx={"m": self.name, "id": st.scn.get("id"), "phase": "decide"})

    def ctx(self, st, **kw):
        d = {"m": self.name, "id": st.scn.get("id"), "k": len(st.acquired)}
        d.update(kw)
        return d


# ------------------------------------------------------------------ trivial policies
class RandomPolicy(Policy):
    """Uniform over remaining checks and STOP (seeded by scenario id + step, so reruns are identical)."""
    name = "random"

    def __init__(self, seed=0, include_stop=True):
        self.seed, self.include_stop = seed, include_stop

    async def step(self, llm, st):
        rng = random.Random(_seed(self.seed, st.scn.get("id"), len(st.acquired)))
        opts = st.remaining + ([STOP] if self.include_stop else [])
        scores = {k: rng.random() for k in opts}
        return {"choice": _argmax(scores), "scores": scores, "info": {}}


class AlwaysAll(Policy):
    """Acquire every check (label order) up to the budget, then decide."""
    name = "always_all"

    async def step(self, llm, st):
        rem = st.remaining
        scores = {k: float(len(rem) - i) for i, k in enumerate(rem)}
        scores[STOP] = 0.0 if rem else 1.0
        return {"choice": rem[0] if rem else STOP, "scores": scores, "info": {}}


class Never(Policy):
    """Never acquire; decide from the initial evidence."""
    name = "never"

    async def step(self, llm, st):
        scores = {k: 0.0 for k in st.remaining}
        scores[STOP] = 1.0
        return {"choice": STOP, "scores": scores, "info": {}}


# ------------------------------------------------------------------ letter-choice policies
class DirectChoice(Policy):
    """Direct choice over (A..F, S) from letter logprobs, two label permutations averaged."""
    name = "direct"
    INSTR = ("Your goal is to maximize your final payoff (the payoff of your decision minus the costs of the requests you "
             "make). Which check should you run next, or should you stop and decide now (S)?")

    async def step(self, llm, st):
        sc = await C.choice_scores(llm, st, self.INSTR, include_stop=True, tag="direct", ctx=self.ctx(st))
        if sc is None:
            return _stop_out({"fail": "no_logprobs"})
        return {"choice": _argmax(sc), "scores": sc, "info": {}}


class Salience(Policy):
    """'Most important item': letter logprobs over checks only (no STOP). Never stops on its own: it stops when the
    episode budget (run.py --budget, or the row's budget) is exhausted or checks run out (use budget sweeps)."""
    name = "salience"
    INSTR = "Which of these checks is the single most important piece of information for this decision?"

    def __init__(self, budget=None):
        self.budget = budget

    async def step(self, llm, st):
        if not st.remaining or (self.budget is not None and len(st.acquired) >= self.budget):
            return _stop_out({"budget_stop": True})
        sc = await C.choice_scores(llm, st, self.INSTR, include_stop=False, k_perm=3, tag="salience", ctx=self.ctx(st))
        if sc is None:
            return {"choice": st.remaining[0], "scores": {}, "info": {"fail": "no_logprobs"}}
        sc = dict(sc)
        sc[STOP] = 0.0
        return {"choice": _argmax(sc, st.remaining), "scores": sc, "info": {}}


async def _all_factor_beliefs(llm, st, tag, ctx):
    fs = C.factors(st.scn)
    bs = await asyncio.gather(*[C.factor_belief(llm, st, f["key"], tag=tag, ctx=ctx) for f in fs])
    return {f["key"]: b for f, b in zip(fs, bs)}


class UncertaintyOnly(Policy):
    """Stop iff the model's belief entropy is below tau (binary latent: H(P(adverse)) bits; ms: mean over factors of
    H(factor belief)/log2(#levels)). Otherwise run the check the model says would most reduce its uncertainty
    (letter logprobs, two permutations). selector='salience' uses the salience prompt instead."""
    name = "uncertainty"

    def __init__(self, tau=0.5, selector="uncertainty"):
        self.tau, self.selector = tau, selector

    async def step(self, llm, st):
        bs = await _all_factor_beliefs(llm, st, "unc_belief", self.ctx(st))
        if any(b is None for b in bs.values()):
            return _stop_out({"fail": "no_belief"})
        levels = {f["key"]: len(f["levels"]) for f in C.factors(st.scn)}
        h = float(np.mean([H(b) / math.log2(levels[k]) for k, b in bs.items()]))
        info = {"beliefs": bs, "entropy": h}
        if h < self.tau or not st.remaining:
            return _stop_out(info)
        if self.selector == "salience":
            instr = Salience.INSTR
        elif C.track(st.scn) == "ms":
            instr = "Which of these requests would most reduce your uncertainty about the unknown factors?"
        else:
            instr = f"Which of these checks would most reduce your uncertainty about whether {C.domain_info(st.scn)['adverse']}?"
        sc = await C.choice_scores(llm, st, instr, include_stop=False, k_perm=3, tag="unc_select", ctx=self.ctx(st))
        if sc is None:
            return {"choice": st.remaining[0], "scores": {}, "info": dict(info, fail="no_logprobs")}
        sc = dict(sc)
        sc[STOP] = 0.0
        return {"choice": _argmax(sc, st.remaining), "scores": sc, "info": info}


class FinRiskAsk(Policy):
    """FinRisk-Ask style: (1) PROCEED vs ASK (letter logprobs, both orders); (2) if ASK, which single check to request.
    scores[label] = P(ask) * P(label | ask); scores[S] = P(proceed)."""
    name = "finrisk_ask"

    async def step(self, llm, st):
        if not st.remaining:
            return _stop_out()
        ev = C.evidence_text(st)
        opts, _ = C.options_text(st, include_stop=False)
        desc = {"PROCEED": "PROCEED: the information is sufficient; make the decision now.",
                "ASK": "ASK: request one more item of information before deciding."}
        orders = (("PROCEED", "ASK"), ("ASK", "PROCEED"))
        calls = []
        for order in orders:
            u = (f"{ev}\n\n{opts}\n\nIs the information sufficient to make the decision now, or should you ask for "
                 f"more information first? Remember that each request has a cost.\n(A) {desc[order[0]]}\n(B) {desc[order[1]]}\n\n"
                 f"Answer with the letter of one option only.")
            calls.append(llm.letter_probs(C.system_prompt(st.scn), u, ["A", "B"], tag="frask_gate", ctx=self.ctx(st)))
        pa = [(p["B"] if order[0] == "PROCEED" else p["A"]) for order, p in zip(orders, await asyncio.gather(*calls))
              if p is not None]
        if not pa:
            return _stop_out({"fail": "no_logprobs"})
        p_ask = sum(pa) / len(pa)
        sc = await C.choice_scores(llm, st, "You decided to ask for more information. Which single item do you request?",
                                   include_stop=False, k_perm=3, tag="frask_target", ctx=self.ctx(st))
        scores = {k: p_ask * v for k, v in (sc or {}).items()}
        scores[STOP] = 1 - p_ask
        choice = STOP if p_ask < 0.5 else (_argmax(sc, st.remaining) if sc else st.remaining[0])
        return {"choice": choice, "scores": scores, "info": {"p_ask": p_ask}}


# ------------------------------------------------------------------ information-gain policies
def _martingale_p1(b0, b1, bz):
    d = b1 - bz
    return 0.5 if abs(d) < 1e-6 else min(1.0, max(0.0, (b0 - bz) / d))


class EIGImagined(Policy):
    """EIG with imagined-observation beliefs (DESIGN.md s4: same LLM beliefs as the the method, objective = IG).
    For check e on factor f: b0 = LLM belief over f, b_o = LLM belief over f after an imagined result o;
    p(o): binary latent + 2 outcomes -> Bayes-consistency (martingale, as DESIGN.md); otherwise the LLM predictive
    distribution over the outcomes. EIG(e) = H(b0) - sum_o p(o) H(b_o) (bits). Stop iff max EIG < tau."""
    name = "eig"

    def __init__(self, tau=0.05):
        self.tau = tau

    async def step(self, llm, st):
        rem = st.remaining
        if not rem:
            return _stop_out()
        scn = st.scn
        fk = {lab: C.check_factor(scn, lab) for lab in rem}
        fs = sorted(set(fk.values()))
        b0s = await asyncio.gather(*[C.factor_belief(llm, st, f, tag="eig_b0", ctx=self.ctx(st, f=f)) for f in fs])
        b0 = dict(zip(fs, b0s))
        jobs, keys = [], []
        for lab in rem:
            outs = C.outcomes(scn, lab)
            for o in range(len(outs)):
                jobs.append(C.factor_belief(llm, st, fk[lab], hypothetical=(lab, o), tag="eig_bo", ctx=self.ctx(st, e=lab, o=o)))
                keys.append((lab, o))
            mart = C.track(scn) != "ms" and len(outs) == 2
            if not mart:
                jobs.append(C.outcome_dist(llm, st, lab, tag="eig_pred", ctx=self.ctx(st, e=lab)))
                keys.append((lab, "p"))
        rs = dict(zip(keys, await asyncio.gather(*jobs)))
        scores, info = {}, {"b0": b0, "bo": {}, "p": {}}
        for lab in rem:
            f = fk[lab]
            if b0[f] is None:
                continue
            n_o = len(C.outcomes(scn, lab))
            bo = [rs[(lab, o)] for o in range(n_o)]
            if any(b is None for b in bo):
                continue
            if (lab, "p") in rs:
                p = rs[(lab, "p")]
                if p is None:
                    continue
            else:
                p1 = _martingale_p1(b0[f][1], bo[1][1], bo[0][1])
                p = [1 - p1, p1]
            scores[lab] = H(b0[f]) - sum(po * H(b) for po, b in zip(p, bo))
            info["bo"][lab], info["p"][lab] = bo, p
        if not scores:
            return _stop_out(dict(info, fail="no_belief"))
        scores[STOP] = self.tau
        best = _argmax(scores, rem)
        choice = best if best is not None and scores[best] >= self.tau else STOP
        return {"choice": choice, "scores": scores, "info": info}


class BEDLLM(Policy):
    """BED-LLM style (arXiv 2508.21184): hypotheses theta = levels of the check's latent factor, weighted by the LLM
    belief (weights='belief') or uniformly (weights='uniform', as BED-LLM after filtering); likelihoods
    p_LLM(y | theta, check) elicited per hypothesis; Rao-Blackwellized EIG
        EIG(x) = H( sum_theta w_theta p(.|theta,x) ) - sum_theta w_theta H(p(.|theta,x)).
    Stop iff max EIG < tau bits."""
    name = "bed_llm"

    def __init__(self, tau=0.05, weights="belief"):
        self.tau, self.weights = tau, weights

    def _conds(self, scn, f):
        if C.track(scn) != "ms":
            d = C.domain_info(scn)
            return [f"Assume it is known that it is NOT the case that {d['adverse']}.",
                    f"Assume it is known that {d['adverse']}."]
        fac = [x for x in C.factors(scn) if x["key"] == f][0]
        return [f"Assume it is known that {fac['name']} is '{lv}'." for lv in fac["levels"]]

    async def step(self, llm, st):
        rem = st.remaining
        if not rem:
            return _stop_out()
        scn = st.scn
        fk = {lab: C.check_factor(scn, lab) for lab in rem}
        fs = sorted(set(fk.values()))
        b0 = dict(zip(fs, await asyncio.gather(*[C.factor_belief(llm, st, f, tag="bed_b0", ctx=self.ctx(st, f=f))
                                                 for f in fs])))
        jobs, keys = [], []
        for lab in rem:
            for th, cond in enumerate(self._conds(scn, fk[lab])):
                jobs.append(C.outcome_dist(llm, st, lab, condition=cond, tag="bed_lik", ctx=self.ctx(st, e=lab, th=th)))
                keys.append((lab, th))
        rs = dict(zip(keys, await asyncio.gather(*jobs)))
        scores, info = {}, {"b0": b0, "lik": {}}
        for lab in rem:
            f = fk[lab]
            if b0[f] is None:
                continue
            nl = len(self._conds(scn, f))
            lik = [rs[(lab, th)] for th in range(nl)]
            if any(x is None for x in lik):
                continue
            w = b0[f] if self.weights == "belief" else [1.0 / nl] * nl
            pm = [sum(w[t] * lik[t][o] for t in range(nl)) for o in range(len(lik[0]))]
            scores[lab] = H(pm) - sum(w[t] * H(lik[t]) for t in range(nl))
            info["lik"][lab] = lik
        if not scores:
            return _stop_out(dict(info, fail="no_belief"))
        scores[STOP] = self.tau
        best = _argmax(scores, rem)
        choice = best if best is not None and scores[best] >= self.tau else STOP
        return {"choice": choice, "scores": scores, "info": info}


# ------------------------------------------------------------------ value-of-information prompting
class VoIPrompt(Policy):
    """DESIGN.md s4 'VoI-prompting': the model verbally estimates, per check, the expected gain in payoff from running
    it minus its cost (STRICT JSON). Run argmax net if > 0, else stop."""
    name = "voi_prompt"

    async def step(self, llm, st):
        rem = st.remaining
        if not rem:
            return _stop_out()
        opts, d2o = C.options_text(st, include_stop=False)
        keys = list(d2o)
        ex = ", ".join(f'"{k}": <number>' for k in keys)
        u = (f"{C.evidence_text(st)}\n\n{opts}\n\nFor each check, estimate its value of information: how much running it "
             f"is expected to add to your expected payoff (because its result might change which action you take), minus "
             f"its cost, in the same units as the payoffs. A check whose result could not change your decision has value 0 "
             f"before cost. Think briefly, then end with STRICT JSON mapping every check letter to its net value: {{{ex}}}.")
        if _BLP == 1:
            u = (f"{C.evidence_text(st)}\n\n{opts}\n\nFor every check, estimate its net value of information: the increase in "
                 f"your expected payoff from running it (as its result may change the action you take), minus its cost, in "
                 f"payoff units. If no result of a check could change your decision, its value before cost is 0. Think briefly, "
                 f"then end with STRICT JSON that maps each check letter to its net value: {{{ex}}}.")
        elif _BLP == 2:
            u = (f"{C.evidence_text(st)}\n\n{opts}\n\nAssess each check: by how much would running it raise your expected "
                 f"payoff, through possibly changing your action, after subtracting its cost (same units as the payoffs)? A "
                 f"check that cannot change your decision is worth 0 before its cost. Think briefly and finish with STRICT JSON "
                 f"giving every check letter its net value: {{{ex}}}.")

        def parse(txt):
            o = C.parse_json(txt)
            if not o:
                return None
            out = {}
            for k, v in o.items():
                kk = str(k).strip().strip("()").upper()
                if kk in d2o and C.to_float(v) is not None:
                    out[d2o[kk]] = C.to_float(v)
            return out if len(out) >= max(1, len(keys) // 2) else None

        sc, raws, ok = await _retry_gen(llm, C.system_prompt(st.scn), u, parse, max_tokens=1024, tag="voi_prompt",
                                        ctx=self.ctx(st))
        if not ok:
            return _stop_out({"fail": "parse"})
        sc = dict(sc)
        sc[STOP] = 0.0
        best = _argmax(sc, rem)
        return {"choice": best if sc[best] > 0 else STOP, "scores": sc, "info": {}}


class VoIPaper(Policy):
    """Value of Information for Human-Agent Communication (arXiv 2601.06407) style pipeline:
    (1) belief b(theta) per latent factor as STRICT JSON; (2) per check, LLM-simulated answer distribution p(y|q,b)
    (JSON); (3) one-step look-ahead: belief over the check's factor re-estimated (JSON) given each simulated answer;
    (4) EU from the stated payoff model (row.model.utility, joint belief = product of factor beliefs),
    VoI(q) = sum_y p(y) V(b_y) - V(b); run argmax VoI - cost if > 0, else stop.
    Differences from IO-VOI: verbal JSON probabilities (not letter-logprob scale) and LLM-predicted p(y)."""
    name = "voi_paper"

    async def _json_dist(self, llm, st, keys, question, hypothetical=None, tag="", ctx=None):
        ev = C.evidence_text(st, hypothetical=hypothetical)
        kl = ", ".join(json.dumps(k) for k in keys)
        u = (f"{ev}\n\n{question}\nReturn STRICT JSON with keys exactly [{kl}] whose values are probabilities that sum "
             f"to 1. Output only the JSON.")

        def parse(txt):
            o = C.parse_json(txt)
            if not o:
                return None
            lo = {str(k).strip().lower().strip("'\""): C.to_float(v) for k, v in o.items()}
            vals = [lo.get(k.lower()) for k in keys]
            if sum(v is None for v in vals) > 1 or all(v is None for v in vals):
                return None
            if any(v is None for v in vals):  # one missing -> remainder
                rest = 1 - sum(v for v in vals if v is not None)
                vals = [max(rest, 0.0) if v is None else v for v in vals]
            vals = [max(v, 0.0) for v in vals]
            return norm(vals) if sum(vals) > 0 else None

        r, _, ok = await _retry_gen(llm, C.system_prompt(st.scn), u, parse, max_tokens=256, tag=tag, ctx=ctx)
        return r

    def _state_question(self, scn, f):
        if C.track(scn) != "ms":
            d = C.domain_info(scn)
            return (f"Based only on this history, estimate the probability of each state: \"{d['adverse_short']}\" "
                    f"({d['adverse']}) or \"{d['benign_short']}\" (it does not).")
        fac = [x for x in C.factors(scn) if x["key"] == f][0]
        return f"Based only on this history, estimate the probability of each possible value of {fac['name']}."

    def _value(self, scn, fbel):
        """max_a sum_s prod_f b_f[s_f] U[a][s]."""
        if "model" not in scn:  # knowledge-required records: stated payoffs only (act/pass x benign/adverse)
            U = np.array([[float(scn["g"]), -float(scn["l"])], [0.0, 0.0]])
        else:
            U = np.asarray(get_model(scn).U)
        if C.track(scn) != "ms":
            b = np.asarray(fbel["Y"])
        else:
            fs = [f["key"] for f in C.factors(scn)]
            states = scn["model"]["states"]
            b = np.array([np.prod([fbel[fs[j]][lv] for j, lv in enumerate(s)]) for s in states])
            b = b / b.sum() if b.sum() > 0 else b
        return float((U @ b).max())

    async def step(self, llm, st):
        rem = st.remaining
        if not rem:
            return _stop_out()
        scn = st.scn
        facs = C.factors(scn)
        lv = {f["key"]: f["levels"] for f in facs}
        b0 = dict(zip(lv, await asyncio.gather(*[
            self._json_dist(llm, st, lv[f], self._state_question(scn, f), tag="vp_prior", ctx=self.ctx(st, f=f))
            for f in lv])))
        if any(v is None for v in b0.values()):
            return _stop_out({"fail": "prior_parse", "b0": b0})
        jobs, keys = [], []
        for lab in rem:
            c = st.checks[lab]
            f = C.check_factor(scn, lab)
            if C.is_binary_finding(scn, lab):
                okeys = ["no finding", "finding"]
                q = (f"If {c['name']} were performed now, estimate the probability of each result: \"finding\" = it finds "
                     f"that {c['finding_text']}; \"no finding\" = it does not.")
            else:
                okeys = C.outcomes(scn, lab)
                q = f"If {c['name']} were requested now, estimate the probability of each possible result."
            jobs.append(self._json_dist(llm, st, okeys, q, tag="vp_answer", ctx=self.ctx(st, e=lab)))
            keys.append((lab, "p"))
            for o in range(len(okeys)):
                jobs.append(self._json_dist(llm, st, lv[f], self._state_question(scn, f), hypothetical=(lab, o),
                                            tag="vp_post", ctx=self.ctx(st, e=lab, o=o)))
                keys.append((lab, o))
        rs = dict(zip(keys, await asyncio.gather(*jobs)))
        V0 = self._value(scn, b0)
        scores, info = {}, {"b0": b0, "sim": {}}
        for lab in rem:
            p = rs[(lab, "p")]
            n_o = len(p) if p else 0
            bo = [rs.get((lab, o)) for o in range(n_o)]
            if p is None or any(b is None for b in bo):
                continue
            f = C.check_factor(scn, lab)
            voi = sum(p[o] * self._value(scn, dict(b0, **{f: bo[o]})) for o in range(n_o)) - V0
            scores[lab] = voi - st.cost(lab)
            info["sim"][lab] = {"p": p, "b": bo, "voi": voi}
        scores[STOP] = 0.0
        best = _argmax(scores, rem)
        choice = best if best is not None and scores[best] > 0 else STOP
        return {"choice": choice, "scores": scores, "info": info}


# ------------------------------------------------------------------ free-generation policies
COT_INSTR = ("Think step by step about what you already know and what else you need to know to make the best decision, "
             "keeping in mind that every request has a cost. What else do you need? Finish with a final line of the form "
             "'ANSWER: X', where X is the letter of the single check to run next, or S if you need nothing more and "
             "should decide now.")


_BLP = int(os.environ.get("BL_PARA", "0"))  # F4 (paraphrased instructions for CoT / VoI-P / PAL; 0 = original
COT_INSTR_P = {1: ("Reason step by step about what is already known and what further information the decision requires, "
                   "remembering that each request costs something. What do you still need? End with a final line 'ANSWER: X', "
                   "where X is the letter of the one check to run next, or S if nothing more is needed and you should decide now."),
               2: ("Work through the problem step by step: what do you know, and what is still missing for the best decision, "
                   "given that every request has a cost? Conclude with one last line 'ANSWER: X', with X the letter of the single "
                   "check to run next, or S to stop and decide now.")}
if _BLP:
    COT_INSTR = COT_INSTR_P[_BLP]
DIRECT_INSTR_P = {1: ("Aim for the highest final payoff, i.e., the payoff of your decision minus the costs of the requests you make. "
                      "Which check do you run next, or do you stop and decide now (S)?"),
                  2: ("You want to maximize your final payoff (your decision's payoff net of the costs of your requests). Should you "
                      "run another check next, and if so which, or stop and decide now (S)?")}
if _BLP:
    DirectChoice.INSTR = DIRECT_INSTR_P[_BLP]


class CoTNeed(Policy):
    """CoT 'what else do you need?' (greedy). Robust parse with up to 2 resamples; parse failure -> STOP (logged)."""
    name = "cot_need"

    def __init__(self, max_tokens=1024):
        self.max_tokens = max_tokens

    def _prompt(self, st):
        opts, d2o = C.options_text(st, include_stop=True)
        return f"{C.evidence_text(st)}\n\n{opts}\n\n{COT_INSTR}", d2o

    async def step(self, llm, st):
        u, d2o = self._prompt(st)
        names = {k: v["name"] for k, v in st.checks.items()}
        how = {}

        def parse(t):
            lab, h = C.parse_choice(t, d2o, names)
            how["h"] = h
            return lab

        lab, raws, ok = await _retry_gen(llm, C.system_prompt(st.scn), u, parse, max_tokens=self.max_tokens,
                                         tag=self.name, ctx=self.ctx(st))
        if not ok:
            lab = STOP
        sc = {k: 0.0 for k in st.remaining + [STOP]}
        sc[lab] = 1.0
        return {"choice": lab, "scores": sc, "info": {"parse": how.get("h") if ok else "fail_default_stop"}}


class SelfConsistencyCoT(CoTNeed):
    """Self-consistency over k sampled CoT answers (T=0.7); scores = vote shares; ties broken by label order, S last."""
    name = "sc_cot"

    def __init__(self, k=5, temperature=0.7, max_tokens=1024):
        super().__init__(max_tokens)
        self.k, self.temperature = k, temperature

    async def step(self, llm, st):
        u, d2o = self._prompt(st)
        names = {k: v["name"] for k, v in st.checks.items()}
        outs = await llm.generate(C.system_prompt(st.scn), u, max_tokens=self.max_tokens, temperature=self.temperature,
                                  n=self.k, seed=_seed(st.scn.get("id"), len(st.acquired)), tag=self.name, ctx=self.ctx(st))
        votes = {k: 0.0 for k in st.remaining + [STOP]}
        nfail = 0
        for o in outs:
            lab, _ = C.parse_choice(o, d2o, names)
            if lab is None:
                nfail += 1
                continue
            votes[lab] += 1
        tot = sum(votes.values())
        if tot == 0:
            return {"choice": STOP, "scores": votes, "info": {"parse_fail": nfail, "fail": "all_parse_fail"}}
        sc = {k: v / tot for k, v in votes.items()}
        order = st.remaining + [STOP]
        best = max(order, key=lambda k: (sc[k], -order.index(k)))
        return {"choice": best, "scores": sc, "info": {"parse_fail": nfail}}


class ReAct(Policy):
    """ReAct-style thought/action loop with the simulator. The transcript lives in st.memory['react'];
    tools: check[X] (returns the result as an Observation) and decide[ACTION]. Labels are the original labels."""
    name = "react"

    def __init__(self, max_tokens=512):
        self.max_tokens = max_tokens

    def _header(self, st):
        acts = C.actions(st.scn)
        st0 = C.State(scn=st.scn)  # initial evidence only; results come back as Observations
        opts, _ = C.options_text(st0, include_stop=False)
        dec = " or ".join(f"decide[{a}]" for a in acts)
        return (f"{C.evidence_text(st0)}\n\n{opts}\n\nSolve this by interleaving Thought, Action and Observation steps. "
                f"Actions:\n  check[X]  -- run check X (one letter) and observe its result\n"
                f"  {dec}  -- stop checking and make the final decision\n"
                f"Each check can be run at most once. Write exactly one 'Thought:' line and one 'Action:' line per step.")

    def _parse(self, t, st):
        t = C.strip_reasoning(t)
        m = list(re.finditer(r"Action\s*:?\s*\**\s*(check|decide|finish|request)\s*[\[\(]\s*([^\]\)]*)[\]\)]", t, flags=re.I))
        if not m:
            m = list(re.finditer(r"(check|decide)\s*\[\s*([^\]]*)\]", t, flags=re.I))
        if not m:
            return None
        verb, arg = m[0].group(1).lower(), m[0].group(2).strip().strip("()'\"").strip()
        if verb in ("check", "request"):
            lab = arg[:1].upper() if arg else ""
            if len(arg) > 1:  # name given instead of letter
                lab2, _ = C.parse_choice(arg, {k: k for k in st.remaining}, {k: v["name"] for k, v in st.checks.items()})
                lab = lab2 or lab
            if lab in st.remaining and st.budget_left > 0:
                return ("check", lab, t[:m[0].end()])
            return ("invalid", arg, t[:m[0].end()])
        return ("decide", C.parse_action(arg, C.actions(st.scn)), t[:m[0].end()])

    async def step(self, llm, st):
        mem = st.memory.setdefault("react", {"transcript": "", "n_obs": 0, "react_action": None})
        while mem["n_obs"] < len(st.acquired):
            a = st.acquired[mem["n_obs"]]
            mem["transcript"] += f"\nObservation: {C.result_line(st.scn, a['label'], a['value'])[2:]}\n"
            mem["n_obs"] += 1
        if not st.remaining or st.budget_left <= 0:
            mem["transcript"] += "\nObservation: No more checks are possible; you must decide now.\n"
        sysm = C.system_prompt(st.scn)
        u = self._header(st)
        info = {}
        sc = {k: 0.0 for k in st.remaining + [STOP]}
        for t in range(3):
            prefill = mem["transcript"] + "Thought:"
            outs = await llm.generate(sysm, u, prefill=prefill, max_tokens=self.max_tokens,
                                      temperature=0.0 if t == 0 else 0.7, seed=None if t == 0 else 77 + t,
                                      stop=["Observation:", "\nObservation"], tag=f"react#try{t}", ctx=self.ctx(st))
            gen = outs[0][len(prefill):] if outs else ""
            r = self._parse(gen, st)
            if r is None or r[0] == "invalid":
                info.setdefault("bad", []).append(gen[-200:])
                continue
            mem["transcript"] = prefill + r[2] + "\n"
            if r[0] == "check":
                sc[r[1]] = 1.0
                return {"choice": r[1], "scores": sc, "info": info}
            mem["react_action"] = r[1]
            sc[STOP] = 1.0
            return {"choice": STOP, "scores": sc, "info": dict(info, react_action=r[1])}
        sc[STOP] = 1.0
        return {"choice": STOP, "scores": sc, "info": dict(info, fail="parse_default_stop")}

    async def decide(self, llm, st):
        out = await super().decide(llm, st)
        ra = st.memory.get("react", {}).get("react_action")
        out["react_action"] = ra
        out["shared_action"] = out["action"]
        if ra is not None and self.use_react_action:
            out["action"] = ra
            out["prob"] = (out.get("probs") or {}).get(ra)
        return out

    use_react_action = True  # the agent's own decide[...] is its decision; the shared probe gives the probability


# ------------------------------------------------------------------ oracle upper bounds
class OracleVOI(Policy):
    """Exact myopic decision-VOI from the row's model (envs_synth core.Model): run argmax VOI - cost if > TOL, else stop
    (ties -> highest net, then label order; same rule as exact.gold_choice). mode='dp': exact DP-optimal move
    (non-myopic upper bound; for envs_real this is the naive-Bayes approximation disclosed there).
    decide_with='bayes' -> Bayes action at the exact posterior; 'llm' -> the shared LLM decision (isolates acquisition)."""
    uses_llm = False

    def __init__(self, mode="myopic", decide_with="bayes"):
        self.mode, self.decide_with = mode, decide_with
        self.name = "oracle_voi" if mode == "myopic" else "oracle_dp"
        if decide_with == "llm":
            self.name += "_llmdec"
            self.uses_llm = True

    async def step(self, llm, st):
        M = get_model(st.scn)
        obs = exact_obs(st)
        t = M.node_table(obs)
        labs = M.labels
        rem = [l for l in st.remaining]
        voi = {labs[i]: v for i, v in t["voi"].items() if labs[i] in rem}
        net = {labs[i]: v for i, v in t["net"].items() if labs[i] in rem}
        scores = dict(net)
        scores[STOP] = 0.0
        info = {"voi": voi, "v0": t["v0"], "v_star": t["v_star"]}
        if st.budget_left <= 0 or not rem:
            return {"choice": STOP, "scores": scores, "info": info}
        if self.mode == "dp":
            f = M.dp_first(obs, tol=TOL)
            choice = STOP if f["move"] == "S" else labs[f["item"]]
            return {"choice": choice, "scores": scores, "info": dict(info, dp_ties=[labs[i] for i in f["ties"]])}
        mx = max(net.values())
        if mx <= TOL:
            return {"choice": STOP, "scores": scores, "info": info}
        cands = [l for l, v in net.items() if v >= mx - TOL]
        choice = max(cands, key=lambda l: (net[l], -labs.index(l)))
        return {"choice": choice, "scores": scores, "info": info}

    async def decide(self, llm, st):
        if self.decide_with == "llm":
            return await super().decide(llm, st)
        M = get_model(st.scn)
        eu = M.eu(exact_obs(st))
        b = M.belief(exact_obs(st))
        acts = C.actions(st.scn)
        ai = int(np.argmax(eu))
        out = {"action": acts[ai], "eu": [float(x) for x in eu], "probs": None,
               "prob": float(sum(b[s] for s in range(len(b)) if int(np.argmax(M.U[:, s])) == ai))}  # P(chosen is ex-post best)
        if C.track(st.scn) != "ms":
            out["belief"] = float(b[1])
        return out


# ------------------------------------------------------------------ registry

class PALVoI(Policy):
    """Program-aided acquisition (PAL; Gao et al. 2023). The LLM writes a Python program from the scenario that prints the
    option letter to choose next (a check letter, or S to stop); the objective (expected payoff minus check costs) is stated.
    Executed in a restricted subprocess (python -I, 10 s timeout, empty env, temp cwd, whitelisted modules, no open/exec).
    One retry with the error message on failure; remaining failures fall back to stop (logged as info.fail)."""
    name = "pal"
    ALLOWED = "{'math','statistics','itertools','functools','fractions','decimal','collections','operator','random','re','json','heapq','bisect'}"

    def _run(self, code):
        import subprocess, tempfile
        wrapper = ("import builtins\n"
                   "ALLOWED=" + self.ALLOWED + "\n"
                   "_imp=builtins.__import__\n"
                   "def _safe(name,*a,**k):\n"
                   "    if name.split('.')[0] not in ALLOWED: raise ImportError('blocked: '+name)\n"
                   "    return _imp(name,*a,**k)\n"
                   "b=dict(vars(builtins)); b['__import__']=_safe\n"
                   "for x in ('open','exec','eval','compile','input','breakpoint'): b.pop(x,None)\n"
                   "exec(compile(CODE,'<pal>','exec'),{'__builtins__':b,'__name__':'__main__'})\n")
        try:
            with tempfile.TemporaryDirectory() as td:
                r = subprocess.run(["python3", "-I", "-c", "CODE=" + repr(code) + "\n" + wrapper], capture_output=True,
                                   text=True, timeout=10, cwd=td, env={})
                return r.stdout.strip(), r.stderr.strip()[-400:]
        except Exception as e:
            return "", repr(e)[:200]

    async def step(self, llm, st):
        import re as _re
        rem = st.remaining
        if not rem:
            return _stop_out()
        opts, d2o = C.options_text(st, include_stop=True)
        u = (f"{C.evidence_text(st)}\n\n{opts}\n\nYour goal is to maximize your expected payoff minus the total cost of the "
             f"checks you run. You may run checks one at a time, see each result, and stop whenever you want. Write a short, "
             f"self-contained Python 3 program (standard library only) that uses the numbers above to decide the best next "
             f"option: one check letter, or S to stop and decide now. The program must print exactly one option letter as its "
             f"last line. Output only the program in a ```python code block.")
        if _BLP == 1:
            u = (f"{C.evidence_text(st)}\n\n{opts}\n\nAim for the highest expected payoff after subtracting the total cost of "
                 f"the checks you run. Checks can be run one after another, each result is shown, and you may stop at any point. "
                 f"Write a short, self-contained Python 3 program (standard library only) that computes the best next option from "
                 f"the numbers above: a check letter, or S to stop and decide now. Its last printed line must be exactly one "
                 f"option letter. Output only the program in a ```python code block.")
        elif _BLP == 2:
            u = (f"{C.evidence_text(st)}\n\n{opts}\n\nYou want to maximize expected payoff net of the costs of the checks you "
                 f"run, requesting checks sequentially, seeing each outcome, and deciding whenever you like. Produce a brief, "
                 f"self-contained Python 3 program (standard library only) that uses the numbers above to pick the next option, "
                 f"either one check letter or S to stop and decide now, and prints that single letter as its final line. Output "
                 f"only the program in a ```python code block.")
        out, err, info = "", "", {}
        for attempt in range(2):
            raws = await llm.generate(C.system_prompt(st.scn), u, max_tokens=1500, tag="pal", ctx=self.ctx(st))
            txt = raws[0] if isinstance(raws, list) and raws else (raws or "")
            m = _re.search(r"```(?:python)?\s*\n(.*?)```", txt, _re.S)
            code = m.group(1) if m else txt
            out, err = self._run(code)
            last = out.strip().splitlines()[-1].strip().strip("().'\"").upper() if out.strip() else ""
            ch = None
            if last in d2o:
                ch = d2o[last]
            elif last in ("STOP", "S"):
                ch = STOP
            else:
                lets = _re.findall(r"\b([A-Z])\b", out.upper())
                for L in reversed(lets):
                    if L in d2o:
                        ch = d2o[L]; break
            if ch is not None and (ch == STOP or ch in rem):
                return {"choice": ch, "scores": {k: float(k == ch) for k in rem + [STOP]}, "info": {"attempt": attempt, "out": out[-80:]}}
            u = u + f"\n\nYour previous program failed (output: {out[-150:]!r}; error: {err[-200:]!r}). Fix it."
        return _stop_out({"fail": "exec", "out": out[-150:], "err": err[-150:]})


REGISTRY = {
    "random": RandomPolicy,
    "always_all": AlwaysAll,
    "never": Never,
    "direct": DirectChoice,
    "salience": Salience,
    "uncertainty": UncertaintyOnly,
    "finrisk_ask": FinRiskAsk,
    "eig": EIGImagined,
    "bed_llm": BEDLLM,
    "voi_prompt": VoIPrompt,
    "voi_paper": VoIPaper,
    "cot_need": CoTNeed,
    "sc_cot": SelfConsistencyCoT,
    "react": ReAct,
    "pal": PALVoI,
    "oracle_voi": lambda **kw: OracleVOI(mode="myopic", **kw),
    "oracle_dp": lambda **kw: OracleVOI(mode="dp", **kw),
    "oracle_voi_llmdec": lambda **kw: OracleVOI(mode="myopic", decide_with="llm", **kw),
}


def make_policy(name, **kw):
    return REGISTRY[name](**kw)

