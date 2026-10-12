"""CLI: run baselines on scenario files (first-step scoring and/or full episodes) against one served model.

python -m baselines.run --model q32 --methods all --env ../envs_synth/data/finance_dev.jsonl.gz --mode both
(run from naacl_dcfr/infoseek). Outputs (resumable, append-only, gzip jsonl):
  baselines/outputs/{model_tag}/{method}/{env}.jsonl.gz            first-step rows (scores per label, choice)
  baselines/outputs/{model_tag}/{method}/{env}.episode.jsonl.gz    episode rows (trajectory, decision, utility)
  ... .calls.jsonl.gz next to each: every raw LLM output / letter top-logprobs with parse status.
"""
import argparse, asyncio, glob, gzip, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from baselines import policies as P, episode as E, llm as L  # noqa: E402
from baselines.toy_env import make_scenario  # noqa: E402

LLM_FREE = {"oracle_voi", "oracle_dp"}  # the others need an LLM for the final decision


def read_jsonl_gz(p):
    out = []
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def _jd(o):
    return o.tolist() if hasattr(o, "tolist") else (o.item() if hasattr(o, "item") else str(o))


def append(p, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with gzip.open(p, "at") as f:
        for r in rows:
            f.write(json.dumps(r, default=_jd) + "\n")


def done_keys(p, key):
    if not os.path.exists(p):
        return set()
    return {key(r) for r in read_jsonl_gz(p)}


def parse_kw(items):
    kw = {}
    for it in items or []:
        k, v = it.split("=", 1)
        m, a = k.split(".", 1)
        try:
            v = json.loads(v)
        except Exception:
            pass
        kw.setdefault(m, {})[a] = v
    return kw


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="tag (q32, oss, ...) or full model id; 'none' for LLM-free methods")
    ap.add_argument("--tag", default=None, help="output folder name (default: --model)")
    ap.add_argument("--methods", default="all", help="comma list, or all")
    ap.add_argument("--env", nargs="*", default=[], help="scenario jsonl(.gz) files/globs")
    ap.add_argument("--toy", type=int, default=0, help="use N toy scenarios instead of --env (smoke tests only)")
    ap.add_argument("--mode", default="both", choices=["first", "episode", "both"])
    ap.add_argument("--reps", type=int, default=1, help="episode worlds per scenario (common random numbers)")
    ap.add_argument("--budget", type=int, nargs="*", default=[None], help="episode budgets (max acquisitions); sweep ok")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conc", type=int, default=48, help="concurrent scenarios")
    ap.add_argument("--kw", nargs="*", help="method.arg=value (e.g. eig.tau=0.1 uncertainty.tau=0.3)")
    ap.add_argument("--out", default=os.path.join(HERE, "outputs"))
    ap.add_argument("--transport", default="fleet", choices=["fleet", "dcfr"])
    ap.add_argument("--think", action="store_true", help="enable thinking for free-generation calls")
    ap.add_argument("--ms_mode", default="numeric", choices=["numeric", "qualitative"], help="ms track belief wording")
    a = ap.parse_args()

    names = list(P.REGISTRY) if a.methods == "all" else a.methods.split(",")
    if a.model == "none":
        names = [n for n in names if n in LLM_FREE]
    kws = parse_kw(a.kw)
    envs = []
    if a.toy:
        envs.append(("toy", [make_scenario(i) for i in range(a.toy)]))
    for pat in a.env:
        for p in sorted(glob.glob(pat)):
            nm = os.path.basename(p).replace(".jsonl.gz", "").replace(".jsonl", "")
            rows = read_jsonl_gz(p)
            for r in rows:
                r["_mode"] = a.ms_mode
            if any(r.get("track") == "ms" for r in rows) and a.ms_mode != "numeric":
                nm += "." + a.ms_mode
            envs.append((nm, rows))
    if a.limit:
        envs = [(n, s[:a.limit]) for n, s in envs]
    tag = a.tag or (L.short_name(L.resolve_model(a.model)) if a.model != "none" else "none")

    transport = None
    model = None
    if a.model != "none":
        model = L.resolve_model(a.model)
        transport = L.make_transport(model, conc=64, prefer=a.transport)
        await transport.__aenter__()
    render = L.make_render()
    sem = asyncio.Semaphore(a.conc)

    async def one(fn, pol, scn, out_p, calls_p, extra):
        async with sem:
            log = L.CallLog()
            llm = L.LLM(model, transport, render=render, think=a.think, log=log) if transport else L.LLM("none", None, log=log)
            try:
                row = await fn(pol, llm, scn, **extra)
            except Exception as e:
                row = {"id": scn["id"], "method": pol.name, "error": repr(e)[:500], **extra}
            row["model"] = model
            append(out_p, [row])
            calls = log.drain()
            if calls:
                append(calls_p, [{"id": scn["id"], **extra, "calls": calls}])
            return row

    t0 = time.time()
    jobs = []  # all methods x envs at once, so the server batch stays full
    for env_name, scns in envs:
        for nm in names:
            pol = P.make_policy(nm, **kws.get(nm, {}))
            if pol.uses_llm and transport is None:
                continue
            d = os.path.join(a.out, tag, pol.name)
            if a.mode in ("first", "both"):
                op = os.path.join(d, f"{env_name}.jsonl.gz")
                dn = done_keys(op, lambda r: r["id"] if "error" not in r else None)
                jobs += [one(E.first_step, pol, s, op, op.replace(".jsonl.gz", ".calls.jsonl.gz"), {})
                         for s in scns if s["id"] not in dn]
            if a.mode in ("episode", "both"):
                for b in a.budget:
                    suf = "" if b is None else f".b{b}"
                    op = os.path.join(d, f"{env_name}.episode{suf}.jsonl.gz")
                    dn = done_keys(op, lambda r: (r["id"], r.get("rep")) if "error" not in r else None)
                    jobs += [one(E.episode, pol, s, op, op.replace(".jsonl.gz", ".calls.jsonl.gz"), {"rep": r, "budget": b})
                             for s in scns for r in range(a.reps) if (s["id"], r) not in dn]
    print(f"[{time.strftime('%H:%M:%S')}] {tag}: {len(jobs)} jobs", flush=True)
    nerr = ndone = 0
    for fut in asyncio.as_completed(jobs):
        r = await fut
        ndone += 1
        nerr += "error" in r
        if ndone % 200 == 0 or ndone == len(jobs):
            print(f"[{time.strftime('%H:%M:%S')}] {tag}: {ndone}/{len(jobs)} done, {nerr} errors, {time.time() - t0:.0f}s",
                  flush=True)
    if transport:
        await transport.__aexit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(main())
