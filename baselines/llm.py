"""LLM adapter used by every baseline.

All calls go through /v1/completions on locally rendered chat prompts (DESIGN.md section 0), so letter-logprob
scoring works the same way across families (incl. gpt-oss and R1 distills).

The adapter needs a *transport* object with the dcfr_lib.Client signature
    async complete(prompt, max_tokens=1, temperature=0.0, logprobs=20, seed=None, stop=None, n=1) -> OpenAI JSON
and a *render* function render(model, system, user, prefill="", think=False) -> str.
`make_llm(model)` prefers infoseek/serving/client.py (when it exists) and falls back to naacl_dcfr/scripts/dcfr_lib.py.
Tests use MockTransport (no network).
"""
import asyncio, importlib, importlib.util, json, math, os, re, sys, time, hashlib
from functools import lru_cache

INFOSEEK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DCFR = os.path.dirname(INFOSEEK)


def _h(s):
    return hashlib.md5(s.encode()).hexdigest()[:12]


# ------------------------------------------------------------------ rendering (cached tokenizers)
@lru_cache(maxsize=None)
def _tok(model):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def default_render(model, system, user, prefill="", think=False, effort="low"):
    """Same logic as dcfr_lib.render (Qwen3 thinking off, gpt-oss -> final channel, R1 -> closed think), cached tokenizer."""
    t = _tok(model)
    m = model.lower()
    msgs = []
    if system:
        if "gemma" in m:
            user = system + "\n\n" + user
        else:
            msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    kw = {}
    if "qwen3" in m:
        kw["enable_thinking"] = bool(think)
    if "gpt-oss" in m:
        kw["reasoning_effort"] = effort
    s = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    if "gpt-oss" in m and not think:
        s += "<|channel|>final<|message|>"
    if ("deepseek-r1" in m or "fin-r1" in m) and not think:
        if not s.rstrip().endswith("<think>"):
            s += "<think>\n"
        s += "\n</think>\n\n"
    return s + prefill


def plain_render(model, system, user, prefill="", think=False, effort="low"):
    """Template-free rendering (tests / mock)."""
    return f"[SYSTEM]{system}\n[USER]{user}\n[ASSISTANT]{prefill}"


# ------------------------------------------------------------------ call logger
class CallLog:
    """Collects every LLM call (prompt hash + full user prompt for free-text calls, raw output, parse result)."""

    def __init__(self, keep_prompts="free"):
        self.rows = []
        self.keep_prompts = keep_prompts  # "all" | "free" | "none"

    def add(self, **kw):
        kw.setdefault("t", round(time.time(), 2))
        self.rows.append(kw)

    def drain(self):
        r, self.rows = self.rows, []
        return r


# ------------------------------------------------------------------ adapter
class LLM:
    def __init__(self, model, transport, render=default_render, think=False, log=None):
        self.model = model
        self.tp = transport
        self.render = render
        self.think = think
        self.log = log if log is not None else CallLog()

    # letter probabilities from the first generated token's top-20 logprobs
    async def letter_probs(self, system, user, letters, prefill="Answer: (", tag="", ctx=None):
        prompt = self.render(self.model, system, user, prefill=prefill, think=False)
        resp = await self.tp.complete(prompt, max_tokens=1, temperature=0.0, logprobs=20)
        probs, mass, top = None, 0.0, None
        try:
            top = resp["choices"][0]["logprobs"]["top_logprobs"][0]
            p = {L: 0.0 for L in letters}
            for t, lp in top.items():
                k = t.strip().strip("(").strip(")").strip()
                if k in p:
                    p[k] += math.exp(lp)
            mass = sum(p.values())
            if mass > 0:
                probs = {k: v / mass for k, v in p.items()}
        except Exception:
            pass
        row = dict(kind="letters", tag=tag, ph=_h(prompt), letters="".join(letters), mass=round(mass, 4),
                   top=({k: round(v, 3) for k, v in top.items()} if isinstance(top, dict) else None),
                   err=(resp.get("error") if isinstance(resp, dict) else None), ok=probs is not None, **(ctx or {}))
        if self.log.keep_prompts == "all":
            row["user"] = user
        self.log.add(**row)
        return probs

    async def generate(self, system, user, prefill="", max_tokens=768, temperature=0.0, n=1, stop=None, seed=None,
                       tag="", ctx=None, think=None):
        th = self.think if think is None else think
        prompt = self.render(self.model, system, user, prefill=prefill, think=th)
        resp = await self.tp.complete(prompt, max_tokens=max_tokens, temperature=temperature, logprobs=None,
                                      n=n, stop=stop, seed=seed)
        outs = []
        try:
            outs = [prefill + c.get("text", "") for c in resp["choices"]]
        except Exception:
            outs = []
        row = dict(kind="gen", tag=tag, ph=_h(prompt), raw=outs, n=n, temp=temperature,
                   err=(resp.get("error") if isinstance(resp, dict) else None), **(ctx or {}))
        if self.log.keep_prompts in ("all", "free"):
            row["user"] = user
        self.log.add(**row)
        return outs


# ------------------------------------------------------------------ factories
def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _roster():
    p = os.path.join(INFOSEEK, "serving", "roster.json")
    try:
        return json.load(open(p)).get("models", [])
    except Exception:
        return []


def resolve_model(tag_or_model):
    """Accepts a roster short name (qwen3-14b), a lead tag (q32, oss, ...) or a full HF id; returns the HF id."""
    for m in _roster():
        if tag_or_model in (m.get("short"), m.get("id")):
            return m["id"]
    tsv = os.path.join(DCFR, "logs", "servers_launched.tsv")
    if os.path.exists(tsv):
        for line in open(tsv):
            f = line.rstrip("\n").split("\t")
            if len(f) >= 4 and f[0] == tag_or_model:
                return f[3]
    return tag_or_model


def short_name(model):
    for m in _roster():
        if m.get("id") == model:
            return m.get("short", model.split("/")[-1])
    return model.split("/")[-1]


class FleetTransport:
    """Adapter: serving/client.py FleetClient (load-balanced over endpoints.json, retries, disk cache) bound to one
    model, exposing the dcfr_lib.Client.complete signature used by LLM."""

    def __init__(self, model, use_cache=True):
        sys.path.insert(0, os.path.join(INFOSEEK, "serving"))
        import client as _fleet  # noqa
        self.fc = _fleet.FleetClient(use_cache=use_cache)
        self.model = model

    async def __aenter__(self):
        await self.fc.__aenter__()
        return self

    async def __aexit__(self, *a):
        await self.fc.__aexit__(*a)

    async def complete(self, prompt, max_tokens=1, temperature=0.0, logprobs=20, seed=None, stop=None, n=1):
        params = {"max_tokens": max_tokens, "temperature": temperature, "n": n, "skip_special_tokens": False}
        if logprobs:
            params["logprobs"] = logprobs
        if seed is not None:
            params["seed"] = seed
        if stop:
            params["stop"] = stop
        return await self.fc.complete(self.model, prompt, **params)


def make_transport(model, conc=64, prefer="fleet"):
    """FleetTransport over infoseek/serving (preferred), else dcfr_lib.Client over the lead's registry."""
    if prefer == "fleet" and os.path.exists(os.path.join(INFOSEEK, "serving", "client.py")):
        try:
            return FleetTransport(model)
        except Exception as e:
            print(f"[baselines.llm] FleetClient unusable ({e!r}); falling back to dcfr_lib", file=sys.stderr)
    sys.path.insert(0, os.path.join(DCFR, "scripts"))
    import dcfr_lib
    return dcfr_lib.Client(model, conc=conc)


def make_render():
    return default_render


# ------------------------------------------------------------------ mock transport (tests)
class MockTransport:
    """fn(prompt, kind, **kw) -> text (for generation) or dict letter->prob (for logprob calls)."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        pass

    async def complete(self, prompt, max_tokens=1, temperature=0.0, logprobs=20, seed=None, stop=None, n=1):
        self.calls.append(prompt)
        if logprobs:
            d = self.fn(prompt, "letters")
            if d is None:
                return {"error": "mock: no logprobs"}
            return {"choices": [{"text": max(d, key=d.get), "logprobs": {"top_logprobs": [
                {k: math.log(max(v, 1e-12)) for k, v in d.items()}]}}]}
        outs = []
        for i in range(n):
            t = self.fn(prompt, "gen", i=i, temperature=temperature)
            if stop:
                for s in stop:
                    if s in t:
                        t = t[:t.index(s)]
            outs.append({"text": t})
        return {"choices": outs}
