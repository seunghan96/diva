#!/usr/bin/env python3
"""Async, load-balanced, cached client for the InfoSeek vLLM fleet.

    from client import FleetClient
    async with FleetClient() as fc:                                   # reads serving/endpoints.json (re-read every 30 s)
        rs = await fc.complete_batch("Qwen/Qwen3-14B", prompts, max_tokens=1, logprobs=20, temperature=0)
        rs = await fc.chat_batch("google/gemma-2-9b-it", [[{"role": "user", "content": "hi"}]], max_tokens=64)
    # sync helper:  results = run_complete("Qwen/Qwen3-8B", prompts, max_tokens=1, logprobs=20)

Features
- Load balancing: least-outstanding-requests over all healthy replicas of the model (ties broken randomly); per-replica
  concurrency cap (default 48). Replicas added/removed by the watchdog are picked up automatically (registry refresh).
- Retries: up to `retries` attempts on connection errors / 5xx / 429 / timeouts, exponential backoff, failing replica
  is cooled down for 30 s and the next attempt goes to another replica. HTTP 400 is not retried (returns {"error": ...}).
- Logprobs: pass `logprobs=k` (completions, k<=20) or `logprobs=True, top_logprobs=k` (chat). Helpers
  `first_token_top_logprobs(resp)` and `option_probs(resp, letters)` extract the first generated token's distribution.
- Cache: every successful response is stored in JSONL.gz shards under cache_dir/<model>/, keyed by sha256 of the
  canonical request (endpoint, model, prompt/messages, all sampling params, plus an optional `cache_tag`). An identical
  request returns the stored response without a call; to draw *new* samples at temperature > 0 pass a different `seed`
  or `cache_tag`. Shards are per process (no locking) and tolerate a truncated tail after a crash.
- Each result is the raw OpenAI-format response dict plus "_url" (replica that served it) and "_cached" (bool).
"""
import asyncio, glob, gzip, hashlib, json, os, random, socket, time, zlib

import aiohttp

D = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REGISTRY = f"{D}/endpoints.json"
DEFAULT_CACHE = os.environ.get("FLEET_CACHE_DIR", f"{D}/cache")


def _canon(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def request_key(kind, payload, cache_tag=None):
    p = dict(payload)
    p.pop("stream", None)
    return hashlib.sha256(_canon({"kind": kind, "p": p, "tag": cache_tag}).encode()).hexdigest()


def _iter_gz_lines(path):
    """Yield complete JSON lines from a (possibly multi-member, possibly truncated) gzip file."""
    with open(path, "rb") as f:
        data = f.read()
    out = b""
    while data:
        d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            out += d.decompress(data)
        except zlib.error:
            break
        if not d.eof:  # truncated member: keep only what decompressed
            break
        data = d.unused_data
    for line in out.split(b"\n"):
        if line.strip():
            try:
                yield json.loads(line)
            except Exception:
                continue  # partial last line


class DiskCache:
    def __init__(self, cache_dir, model, flush_every=200):
        self.dir = os.path.join(cache_dir, model.replace("/", "__"))
        os.makedirs(self.dir, exist_ok=True)
        self.mem = {}
        for p in sorted(glob.glob(os.path.join(self.dir, "*.jsonl.gz"))):
            for rec in _iter_gz_lines(p):
                if "k" in rec and "r" in rec:
                    self.mem[rec["k"]] = rec["r"]
        self.shard = os.path.join(self.dir, f"{socket.gethostname()}_{os.getpid()}_{int(time.time())}.jsonl.gz")
        self.buf = []
        self.flush_every = flush_every

    def get(self, k):
        return self.mem.get(k)

    def put(self, k, r, req=None):
        self.mem[k] = r
        self.buf.append({"k": k, "r": r, "q": req, "t": round(time.time(), 2)})
        if len(self.buf) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        data = "".join(_canon(x) + "\n" for x in self.buf).encode()
        with open(self.shard, "ab") as f:  # one gzip member per flush (appendable, crash-tolerant)
            f.write(gzip.compress(data))
        self.buf = []


class FleetClient:
    def __init__(self, registry=DEFAULT_REGISTRY, cache_dir=DEFAULT_CACHE, use_cache=True, per_replica=48,
                 timeout=900, retries=6, store_request=False):
        self.registry = registry
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.per_replica = per_replica
        self.timeout = timeout
        self.retries = retries
        self.store_request = store_request
        self.eps = {}
        self.eps_t = 0
        self.inflight = {}
        self.cool = {}
        self.caches = {}
        self.session = None
        self.stats = {"calls": 0, "cached": 0, "errors": 0, "retries": 0}

    # ---------------------------------------------------------------- lifecycle
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout),
                                             connector=aiohttp.TCPConnector(limit=0, ttl_dns_cache=600))
        self._refresh(force=True)
        return self

    async def __aexit__(self, *a):
        self.flush()
        await self.session.close()

    def flush(self):
        for c in self.caches.values():
            c.flush()

    # ---------------------------------------------------------------- registry / balancing
    def _refresh(self, force=False):
        if force or time.time() - self.eps_t > 30:
            try:
                with open(self.registry) as f:
                    eps = json.load(f)
                self.eps = {m: [u.rstrip("/") for u in us] for m, us in eps.items() if not m.startswith("_")}
            except Exception:
                pass
            self.eps_t = time.time()

    def urls(self, model):
        self._refresh()
        us = self.eps.get(model)
        if not us:  # allow short names / substring match ("qwen3-14b")
            for m, v in self.eps.items():
                if model.lower() in (m.lower(), m.split("/")[-1].lower()):
                    us = v
                    break
        return us or []

    async def _pick(self, model, exclude=()):
        while True:
            us = self.urls(model)
            if not us:
                raise RuntimeError(f"no endpoint for {model} in {self.registry}")
            t = time.time()
            cand = [u for u in us if u not in exclude and self.cool.get(u, 0) < t] or [u for u in us if self.cool.get(u, 0) < t] or us
            free = [u for u in cand if self.inflight.get(u, 0) < self.per_replica]
            if free:
                m = min(self.inflight.get(u, 0) for u in free)
                u = random.choice([x for x in free if self.inflight.get(x, 0) == m])
                self.inflight[u] = self.inflight.get(u, 0) + 1
                return u
            await asyncio.sleep(0.02)

    def _cache(self, model):
        if model not in self.caches:
            self.caches[model] = DiskCache(self.cache_dir, model)
        return self.caches[model]

    # ---------------------------------------------------------------- core request
    async def request(self, kind, payload, cache_tag=None, use_cache=None):
        """kind in {"completions", "chat/completions"}; payload = OpenAI request body (must include "model")."""
        model = payload["model"]
        uc = self.use_cache if use_cache is None else use_cache
        k = request_key(kind, payload, cache_tag)
        if uc:
            hit = self._cache(model).get(k)
            if hit is not None:
                self.stats["cached"] += 1
                return dict(hit, _cached=True)
        tried, err = [], ""
        for attempt in range(self.retries):
            u = await self._pick(model, exclude=tried)
            try:
                self.stats["calls"] += 1
                async with self.session.post(f"{u}/v1/{kind}", json=payload) as r:
                    txt = await r.text()
                    if r.status == 200:
                        resp = json.loads(txt)
                        if uc:
                            self._cache(model).put(k, resp, payload if self.store_request else None)
                        resp["_url"] = u
                        resp["_cached"] = False
                        return resp
                    err = f"HTTP {r.status}: {txt[:300]}"
                    if r.status == 400 or r.status == 404 and "does not exist" in txt:
                        self.stats["errors"] += 1
                        return {"error": err, "_url": u}
            except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError) as e:
                err = f"{type(e).__name__}: {e}"
            finally:
                self.inflight[u] -= 1
            self.cool[u] = time.time() + 30
            tried.append(u)
            self.stats["retries"] += 1
            await asyncio.sleep(min(30, 1.5 * 2 ** attempt) * (0.5 + random.random()))
            self._refresh(force=True)
        self.stats["errors"] += 1
        return {"error": f"failed after {self.retries} attempts: {err}"}

    async def complete(self, model, prompt, cache_tag=None, use_cache=None, **params):
        return await self.request("completions", {"model": model, "prompt": prompt, **params}, cache_tag, use_cache)

    async def chat(self, model, messages, cache_tag=None, use_cache=None, **params):
        return await self.request("chat/completions", {"model": model, "messages": messages, **params}, cache_tag, use_cache)

    async def complete_batch(self, model, prompts, progress=False, **params):
        return await self._batch([self.complete(model, p, **params) for p in prompts], progress)

    async def chat_batch(self, model, conversations, progress=False, **params):
        return await self._batch([self.chat(model, m, **params) for m in conversations], progress)

    async def _batch(self, coros, progress):
        n = len(coros)
        done = 0
        t0 = time.time()

        async def wrap(i, c):
            nonlocal done
            r = await c
            done += 1
            if progress and (done % max(1, n // 20) == 0 or done == n):
                print(f"[fleet] {done}/{n} {time.time() - t0:.0f}s {self.stats}", flush=True)
            return i, r

        out = [None] * n
        for i, r in await asyncio.gather(*(wrap(i, c) for i, c in enumerate(coros))):
            out[i] = r
        self.flush()
        return out


# -------------------------------------------------------------------- helpers
def text_of(resp):
    c = (resp.get("choices") or [{}])[0]
    return c.get("text") if "text" in c else (c.get("message") or {}).get("content")


def first_token_top_logprobs(resp):
    """{token: logprob} for the first generated token (completions or chat format); {} on error."""
    try:
        c = resp["choices"][0]
        lp = c["logprobs"]
        if "top_logprobs" in lp and isinstance(lp["top_logprobs"], list):  # completions
            return dict(lp["top_logprobs"][0])
        return {t["token"]: t["logprob"] for t in lp["content"][0]["top_logprobs"]}  # chat
    except Exception:
        return {}


def option_probs(resp, letters=("A", "B", "C", "D", "E", "F", "S")):
    """Renormalized probability over option letters from the first token's top logprobs (strips spaces/parens)."""
    import math
    acc = {L: 0.0 for L in letters}
    for tok, lp in first_token_top_logprobs(resp).items():
        t = tok.replace("\u2581", " ").replace("\u0120", " ").strip().strip("()").strip()
        if t in acc:
            acc[t] += math.exp(lp)
    z = sum(acc.values())
    return {L: (v / z if z > 0 else float("nan")) for L, v in acc.items()}


def run_complete(model, prompts, **params):
    async def go():
        async with FleetClient() as fc:
            return await fc.complete_batch(model, prompts, **params)
    return asyncio.run(go())


def run_chat(model, conversations, **params):
    async def go():
        async with FleetClient() as fc:
            return await fc.chat_batch(model, conversations, **params)
    return asyncio.run(go())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="client self-test: one completion + one chat per model in endpoints.json")
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=8, help="parallel copies per model (exercises balancing)")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()

    async def main():
        async with FleetClient(use_cache=not a.no_cache) as fc:
            models = [a.model] if a.model else sorted(fc.eps)
            for m in models:
                t0 = time.time()
                rs = await fc.complete_batch(m, ["The capital of France is"] * a.n, max_tokens=1, logprobs=20, temperature=0,
                                             cache_tag=None)
                ch = await fc.chat(m, [{"role": "user", "content": "Answer with one letter: is 2+2=4? (A) yes (B) no"}],
                                   max_tokens=1, logprobs=True, top_logprobs=5, temperature=0)
                urls = sorted({r.get("_url") for r in rs})
                top = sorted(first_token_top_logprobs(rs[0]).items(), key=lambda x: -x[1])[:3]
                print(f"{m:55s} ok={sum('error' not in r for r in rs)}/{a.n} replicas_used={len(urls)} "
                      f"top={[(t, round(v, 2)) for t, v in top]} chat_AB={ {k: round(v, 2) for k, v in option_probs(ch, ('A', 'B')).items()} } "
                      f"{time.time() - t0:.1f}s err={rs[0].get('error', '')[:80]}")
            print("stats", fc.stats)

    asyncio.run(main())
