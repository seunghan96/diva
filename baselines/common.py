"""Shared pieces for all baselines: track adapters (binary / ms / real), scenario state + rendering, robust parsers,
belief elicitation, final decision.

Scenario rows follow envs_synth / envs_real (DESIGN.md s2-3): id, track, actions, checks, observed, label_order,
text_parts, model{prior, utility, items[lik, outcomes, cost], observed, budget, states?}, exact, episodes.
Three tracks:
  binary : latent Y in {benign, adverse}, binary checks with stated reliabilities (s1, s0)
  real   : latent Y binary, checks with 3-5 value bins (outcome = bin; result text = the ACTUAL value)
  ms     : several latent factors (row.factors), checks inform one factor, 3 actions (BUY/HOLD/SELL)
Every track is viewed as a list of latent *factors* (binary/real: one factor Y with levels = row.states).
"""
import asyncio, difflib, json, math, os, re, sys
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
INFOSEEK = os.path.dirname(HERE)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
STOP = "S"


# ------------------------------------------------------------------ domain info
def _load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_DOM = None


def _domains():
    global _DOM
    if _DOM is None:
        _DOM = {}
        for p in (os.path.join(INFOSEEK, "envs_synth", "domains_bin.py"), os.path.join(INFOSEEK, "scripts", "domains.py")):
            if os.path.exists(p):
                try:
                    _DOM = getattr(_load(p, "infoseek_domains_" + os.path.basename(p)[:-3]), "DOMAINS", {})
                    if _DOM:
                        break
                except Exception:
                    continue
    return _DOM


def track(scn):
    return scn.get("track", "binary")


def domain_info(scn):
    if "_dinfo" in scn:
        return scn["_dinfo"]
    d = dict(scn.get("domain_info") or _domains().get(scn.get("domain"), {}))
    for k in ("role", "context", "adverse", "adverse_short", "benign_short"):
        if k in scn:
            d[k] = scn[k]
    acts = scn.get("actions")
    if acts:
        d["act"], d["pass"] = acts[0], acts[-1]
        d["actions"] = list(acts)
    else:
        d.setdefault("act", "ACT")
        d.setdefault("pass", "PASS")
        d["actions"] = [d["act"], d["pass"]]
    st = scn.get("states")
    if st and len(st) == 2:
        d.setdefault("benign_short", st[0])
        d.setdefault("adverse_short", st[1])
    if "adverse" not in d:
        d["adverse"] = (f"the adverse case occurs (the stock {st[1]})" if track(scn) == "real" and st
                        else "the adverse outcome occurs")
    d.setdefault("adverse_short", "adverse")
    d.setdefault("benign_short", "not adverse")
    d.setdefault("role", "a careful decision maker")
    scn["_dinfo"] = d
    return d


def actions(scn):
    return domain_info(scn)["actions"]


def pct(x):
    return f"{round(100 * x)}%"


# ------------------------------------------------------------------ factors and outcomes
def factors(scn):
    """[{key, name, levels}] ; binary/real: one factor 'Y' with levels [benign, adverse]."""
    if track(scn) == "ms":
        fs = scn["factors"]
        return [{"key": f["key"], "name": f["name"], "levels": list(f["levels"])} for f in fs]
    d = domain_info(scn)
    return [{"key": "Y", "name": "the outcome", "levels": [d["benign_short"], d["adverse_short"]]}]


def check_factor(scn, label):
    c = {x["label"]: x for x in scn["checks"]}[label]
    return c.get("factor", "Y") if track(scn) == "ms" else "Y"


def outcomes(scn, label):
    """Outcome names of a check (index = value)."""
    c = {x["label"]: x for x in scn["checks"]}[label]
    if c.get("outcomes"):
        return list(c["outcomes"])
    if c.get("bins"):
        return [b.get("text", b.get("level", str(i))) for i, b in enumerate(c["bins"])]
    m = scn.get("model")
    if m:
        for it in m["items"]:
            if it["label"] == label and it.get("outcomes"):
                return list(it["outcomes"])
    return ["no finding", "finding"]


def is_binary_finding(scn, label):
    c = {x["label"]: x for x in scn["checks"]}[label]
    return track(scn) == "binary" and "finding_text" in c and not c.get("bins")


# ------------------------------------------------------------------ episode state
@dataclass
class State:
    scn: dict
    acquired: list = field(default_factory=list)  # [{"label","value"}] in acquisition order
    memory: dict = field(default_factory=dict)  # per-episode scratch for stateful policies (ReAct)

    @property
    def checks(self):
        return {c["label"]: c for c in self.scn["checks"]}

    @property
    def remaining(self):
        got = {a["label"] for a in self.acquired}
        return [c["label"] for c in self.scn["checks"] if c["label"] not in got]

    @property
    def budget_left(self):
        b = self.scn.get("budget") or len(self.scn["checks"])
        return max(0, int(b) - len(self.acquired))

    def spent(self):
        ch = self.checks
        return sum(float(ch[a["label"]].get("cost", 0)) for a in self.acquired)

    def cost(self, label):
        return float(self.checks[label].get("cost", 0))


def option_body(scn, label):
    tp = scn.get("text_parts") or {}
    ol = tp.get("option_lines") or {}
    if label in ol:
        return re.sub(r"^\s*\([A-Z]\)\s*", "", ol[label]).strip()
    if "_bodies" not in scn:  # parse a plain options block
        out, cur = {}, None
        for line in (tp.get("options") or "").splitlines():
            m = re.match(r"^\s*\(([A-Z])\)\s*(.*)$", line)
            if m:
                cur = m.group(1)
                out[cur] = m.group(2).strip()
            elif cur and line.strip():
                out[cur] += " " + line.strip()
        scn["_bodies"] = out
    if label in scn["_bodies"]:
        return scn["_bodies"][label]
    c = {x["label"]: x for x in scn["checks"]}[label]
    d = domain_info(scn)
    return (f"{c['name']} (cost {c.get('cost', 0):g} points): shows whether {c['finding_text']}. When {d['adverse']}, "
            f"this finding is present in about {pct(c['s1'])} of cases; otherwise in about {pct(c['s0'])}.")


def result_line(scn, label, value):
    c = {x["label"]: x for x in scn["checks"]}[label]
    if track(scn) == "real":
        txt = c.get("actual_text") if value == c.get("actual_bin", value) else outcomes(scn, label)[value]
        return f"- Result of ({label}) {c['name']}: {txt or outcomes(scn, label)[value]}."
    if is_binary_finding(scn, label):
        if value:
            return f"- Result of ({label}) {c['name']}: it found that {c['finding_text']}."
        return f"- Result of ({label}) {c['name']}: it did NOT find that {c['finding_text']}."
    return f"- Result of ({label}) {c['name']}: '{outcomes(scn, label)[value]}'."


def hypo_line(scn, label, value):
    c = {x["label"]: x for x in scn["checks"]}[label]
    if is_binary_finding(scn, label):
        if value:
            return f"- (Hypothetical) Suppose {c['name']} were performed and found that {c['finding_text']}."
        return f"- (Hypothetical) Suppose {c['name']} were performed and did NOT find that {c['finding_text']}."
    o = outcomes(scn, label)[value]
    if track(scn) == "real":
        return f"- (Hypothetical) Suppose {c['name']} were requested and reported: {o}."
    return f"- (Hypothetical) Suppose {c['name']} were requested and returned '{o}'."


def evidence_text(st, hypothetical=None):
    """Situation + payoffs + beliefs/base rate + everything known so far (+ optional hypothetical result)."""
    scn = st.scn
    tp = scn.get("text_parts") or {}
    d = domain_info(scn)
    res = [result_line(scn, a["label"], a["value"]) for a in st.acquired]
    if hypothetical is not None:
        res.append(hypo_line(scn, *hypothetical))
    if track(scn) == "ms":
        mode = scn.get("_mode", "numeric")
        bel = tp.get("beliefs_qualitative" if mode == "qualitative" else "beliefs_numeric", "")
        parts = [tp.get("header", ""), tp.get("situation", ""), bel, tp.get("payoff", "")]
        if res:
            parts.append("Results of your requests so far:\n" + "\n".join(res))
        return "\n\n".join(p for p in parts if p)
    parts = [tp.get("context") or (f"You are {d['role']}. " + d.get("context", ""))]
    if tp.get("payoff"):
        parts.append(tp["payoff"])
    elif "gain_text" in d:
        parts.append("Payoffs: " + d["gain_text"].format(g=scn["g"], l=scn["l"]))
    if tp.get("base_rate"):
        parts.append(tp["base_rate"])
    known = (tp.get("known") or "").strip()
    if known:
        parts.append(known)
    if res:
        parts.append("Results of the checks you ran:\n" + "\n".join(res))
    elif not known:
        parts.append("Already known: nothing beyond the base rate.")
    return "\n\n".join(p for p in parts if p)


def options_text(st, perm=None, include_stop=True):
    """Returns (text, disp2orig). perm: original labels in display order (relabelled A, B, ...)."""
    rem = st.remaining
    order = [x for x in (perm or rem) if x in rem]
    tp = st.scn.get("text_parts") or {}
    disp2orig, lines = {}, []
    for i, lab in enumerate(order):
        dl = LETTERS[i]
        disp2orig[dl] = lab
        lines.append(f"({dl}) {option_body(st.scn, lab)}")
    if include_stop:
        acts = actions(st.scn)
        lines.append(f"({STOP}) Stop and decide now ({', '.join(acts[:-1])} or {acts[-1]}).")
        disp2orig[STOP] = STOP
    unit = "thousand dollars" if track(st.scn) == "ms" else "points"
    head = (f"Available requests (each at most once; costs are in {unit} and are subtracted from your payoff)"
            if track(st.scn) == "ms" else
            "Available checks (each at most once; the cost of each check you run is subtracted from your score)")
    bl = st.budget_left
    if bl < len(rem):
        head += f". You may run at most {bl} more"
    return head + ":\n" + "\n".join(lines), disp2orig


def system_prompt(scn):
    return f"You are {domain_info(scn)['role']}. Reason carefully and follow the answer format exactly."


def permutations(st, k=2):
    """Display permutations used to average out letter-position bias: identity and reversed."""
    rem = st.remaining
    ps = [rem, rem[::-1]]
    if k >= 3 and len(rem) > 2:  # third display order: rotation by half (3-order protocol shared with the direct policy)
        h = len(rem) // 2
        ps.append(rem[h:] + rem[:h])
    return ps[:k] if len(rem) > 1 else ps[:1]


# ------------------------------------------------------------------ robust parsing
def strip_reasoning(text):
    if text is None:
        return ""
    if "<|channel|>final<|message|>" in text:
        text = text.split("<|channel|>final<|message|>")[-1]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    for tok in ("<|end|>", "<|return|>", "<|im_end|>", "<|eot_id|>", "<end_of_turn>", "</s>"):
        text = text.replace(tok, "")
    return text.strip()


def parse_json(text):
    """Last parseable top-level {...} object; tolerates code fences, single quotes, trailing commas, % signs."""
    t = strip_reasoning(text)
    t = re.sub(r"```(?:json)?", "", t)
    starts = [m.start() for m in re.finditer(r"\{", t)]
    top, depth = [], 0
    for j, ch in enumerate(t):
        if ch == "{":
            if depth == 0:
                top.append(j)
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
    starts = list(reversed(top)) + [s for s in reversed(starts) if s not in top]
    for s in starts:
        depth = 0
        for j in range(s, len(t)):
            if t[j] == "{":
                depth += 1
            elif t[j] == "}":
                depth -= 1
                if depth == 0:
                    cand = t[s:j + 1]
                    for fix in (lambda x: x,
                                lambda x: re.sub(r",\s*([}\]])", r"\1", x),
                                lambda x: re.sub(r",\s*([}\]])", r"\1", x.replace("'", '"')),
                                lambda x: re.sub(r"(\d+(?:\.\d+)?)\s*%", lambda m: str(float(m.group(1)) / 100),
                                                 re.sub(r",\s*([}\]])", r"\1", x.replace("'", '"')))):
                        try:
                            obj = json.loads(fix(cand))
                            if isinstance(obj, dict):
                                return obj
                        except Exception:
                            pass
                    break
    return None


def to_float(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", x.replace(",", ""))
        if m:
            v = float(m.group(0))
            return v / 100 if "%" in x[m.end():m.end() + 2] else v
    return None


_STOP_WORDS = r"(?:stop|none|nothing|no (?:further|more|additional) (?:check|information|checks|requests?)|proceed|decide now|sufficient)"


def parse_choice(text, disp2orig, names=None):
    """Map free text to an original label (or 'S'). Returns (label or None, how).
    Priority: explicit 'ANSWER: X' / 'Final answer: (X)' / 'request[X]'; then the last '(X)'; then the check name
    (exact, then fuzzy); then stop words. names: {orig_label: check name}."""
    t = strip_reasoning(text)
    valid = set(disp2orig)
    pats = [r"(?:final\s+)?answer\s*\**\s*[:\-]\s*\**\s*\(?\s*([A-Z])\s*\)?(?![a-z])",
            r"(?:request|check|choose|choice|select|acquire)\s*[\[\(:]\s*\(?([A-Z])\)?\s*[\]\)]?",
            r"\boption\s*\(?([A-Z])\)?(?![a-z])"]
    for p in pats:
        ms = [m.group(1) for m in re.finditer(p, t, flags=re.I) if m.group(1).upper() in valid]
        if ms:
            return disp2orig[ms[-1].upper()], "explicit"
    ms = [m for m in re.findall(r"\(([A-Z])\)", t) if m in valid]
    if ms:
        return disp2orig[ms[-1]], "paren"
    tail = t[-300:]
    if names:
        best, bl = 0.0, None
        for lab, nm in names.items():
            if lab not in disp2orig.values():
                continue
            if nm.lower() in tail.lower():
                return lab, "name"
            r = difflib.SequenceMatcher(None, nm.lower(), tail.lower()[-len(nm) - 40:]).ratio()
            if r > best:
                best, bl = r, lab
        if best > 0.75:
            return bl, "fuzzy"
    if STOP in valid and re.search(_STOP_WORDS, tail, flags=re.I):
        return STOP, "stopword"
    return None, "fail"


def parse_action(text, acts):
    """Map free text to one of the action names (last mention wins)."""
    t = strip_reasoning(text).upper()
    best, pos = None, -1
    for a in acts:
        for m in re.finditer(r"\b" + re.escape(a.upper()) + r"\b", t):
            if m.start() > pos:
                best, pos = a, m.start()
    return best


def H(p):
    """Entropy in bits of a probability vector."""
    s = 0.0
    for x in p:
        if x > 1e-12:
            s -= x * math.log2(x)
    return s


def norm(v):
    z = sum(v)
    return [x / z for x in v] if z > 0 else [1.0 / len(v)] * len(v)


# ------------------------------------------------------------------ belief elicitation
BINS = [i / 10 for i in range(11)]


async def scale_prob(llm, st, question, hypothetical=None, condition=None, tag="prob", ctx=None):
    """A probability on the 11-bin letter scale (0%..100%), averaged over ascending and descending orders."""
    scn = st.scn
    ev = evidence_text(st, hypothetical=hypothetical)
    if condition:
        ev += "\n\n" + condition
    outs = []
    for asc in (True, False):
        vals = BINS if asc else BINS[::-1]
        opts = "\n".join(f"({L}) {round(v * 100)}%" for L, v in zip(LETTERS[:11], vals))
        u = f"{ev}\n\n{question}\n{opts}\n\nAnswer with the letter of the closest option only."
        outs.append(llm.letter_probs(system_prompt(scn), u, list(LETTERS[:11]), tag=tag, ctx=ctx))
    ests = []
    for asc, p in zip((True, False), await asyncio.gather(*outs)):
        if p is None:
            continue
        vals = BINS if asc else BINS[::-1]
        ests.append(sum(p[L] * v for L, v in zip(LETTERS[:11], vals)))
    return sum(ests) / len(ests) if ests else None


async def belief(llm, st, hypothetical=None, condition=None, question=None, tag="belief", ctx=None):
    """P(adverse | evidence) for binary-latent tracks (11-bin scale, two orders)."""
    d = domain_info(st.scn)
    q = question or f"Based only on this information, what is the probability that {d['adverse']}?"
    return await scale_prob(llm, st, q, hypothetical=hypothetical, condition=condition, tag=tag, ctx=ctx)


async def factor_belief(llm, st, fkey, hypothetical=None, condition=None, tag="fbelief", ctx=None):
    """Distribution over the levels of one latent factor. Binary-latent tracks: [1-b, b] from `belief`.
    ms: one 11-bin probability per level, normalized."""
    if track(st.scn) != "ms":
        b = await belief(llm, st, hypothetical=hypothetical, condition=condition, tag=tag, ctx=ctx)
        return None if b is None else [1 - b, b]
    f = [x for x in factors(st.scn) if x["key"] == fkey][0]
    qs = [scale_prob(llm, st, f"Based only on this information, what is the probability that {f['name']} is '{lv}'?",
                     hypothetical=hypothetical, condition=condition, tag=tag, ctx=dict(ctx or {}, f=fkey, lv=i))
          for i, lv in enumerate(f["levels"])]
    ps = await asyncio.gather(*qs)
    if any(p is None for p in ps):
        return None
    return norm([max(p, 1e-6) for p in ps])


async def outcome_dist(llm, st, label, condition=None, tag="pred", ctx=None):
    """LLM predictive distribution over a check's outcomes (letter logprobs, two orders averaged)."""
    scn = st.scn
    c = st.checks[label]
    outs = outcomes(scn, label)
    if is_binary_finding(scn, label):
        q = f"If {c['name']} were performed now, what is the probability that it would find that {c['finding_text']}?"
        p = await scale_prob(llm, st, q, condition=condition, tag=tag, ctx=ctx)
        return None if p is None else [1 - p, p]
    ev = evidence_text(st) + ("\n\n" + condition if condition else "")
    calls = []
    orders = [list(range(len(outs))), list(range(len(outs)))[::-1]]
    for order in orders:
        opts = "\n".join(f"({LETTERS[i]}) {outs[j]}" for i, j in enumerate(order))
        u = (f"{ev}\n\nIf {c['name']} were requested now, which result would it most likely report?\n{opts}\n\n"
             f"Answer with the letter of one option only.")
        calls.append(llm.letter_probs(system_prompt(scn), u, list(LETTERS[:len(outs)]), tag=tag, ctx=ctx))
    acc, n = [0.0] * len(outs), 0
    for order, p in zip(orders, await asyncio.gather(*calls)):
        if p is None:
            continue
        n += 1
        for i, j in enumerate(order):
            acc[j] += p[LETTERS[i]]
    return None if n == 0 else norm(acc)


# ------------------------------------------------------------------ letter choice over checks (+S), permutation averaged
async def choice_scores(llm, st, instruction, include_stop=True, k_perm=2, tag="choice", ctx=None):
    """Probability per original label (and 'S') from letter logprobs, averaged over display permutations."""
    scn = st.scn
    acc, n, calls, maps = {}, 0, [], []
    for perm in permutations(st, k_perm):
        opts, d2o = options_text(st, perm, include_stop=include_stop)
        u = f"{evidence_text(st)}\n\n{opts}\n\n{instruction}\nAnswer with the letter of one option only."
        calls.append(llm.letter_probs(system_prompt(scn), u, list(d2o), tag=tag, ctx=ctx))
        maps.append(d2o)
    for p, d2o in zip(await asyncio.gather(*calls), maps):
        if p is None:
            continue
        n += 1
        for dl, pr in p.items():
            acc[d2o[dl]] = acc.get(d2o[dl], 0.0) + pr
    if n == 0:
        return None
    return {k: v / n for k, v in acc.items()}


# ------------------------------------------------------------------ final decision with a probability
async def final_decision(llm, st, with_belief=True, ctx=None):
    """Decision over the row's actions from letter logprobs (identity and reversed order averaged).
    Returns {action (name), probs {name: p}, prob (of the chosen action), belief (binary-latent tracks: P(adverse))}."""
    scn = st.scn
    acts = actions(scn)
    ev = evidence_text(st)
    orders = [acts, acts[::-1]]
    calls = []
    for order in orders:
        opts = "\n".join(f"({LETTERS[i]}) {a}" for i, a in enumerate(order))
        u = (f"{ev}\n\nNo more requests are possible. Choose the action that maximizes your expected payoff.\n{opts}\n\n"
             f"Answer with the letter of one option only.")
        calls.append(llm.letter_probs(system_prompt(scn), u, list(LETTERS[:len(acts)]), tag="decide", ctx=ctx))
    if with_belief and track(scn) != "ms":
        calls.append(belief(llm, st, tag="final_belief", ctx=ctx))
    rs = await asyncio.gather(*calls)
    acc, n = {a: 0.0 for a in acts}, 0
    for order, p in zip(orders, rs[:2]):
        if p is None:
            continue
        n += 1
        for i, a in enumerate(order):
            acc[a] += p[LETTERS[i]]
    out = {"action": None, "probs": None, "prob": None,
           "belief": rs[2] if (with_belief and track(scn) != "ms") else None}
    if n:
        probs = {a: v / n for a, v in acc.items()}
        a = max(acts, key=lambda x: probs[x])
        out.update(action=a, probs=probs, prob=probs[a])
    return out
