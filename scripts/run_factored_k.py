"""Factored elicitation for KNOWLEDGE-REQUIRED scenarios (likelihoods not stated). Same model structure as run_factored.py
(base rate + per-check P(finding|adverse), P(finding|not adverse)); wording asks for the model's knowledge.
Forms: lik (likelihood questions), cond (P(adverse | finding present/absent) -> converted to likelihoods with Bayes and
the elicited P(finding)), both (stored separately; combined at evaluation time).
Samples: greedy + K samples at T=0.7 (stored raw; aggregation at evaluation time).
usage: run_factored_k.py MODEL ENV OUT [--form lik|cond|both] [--k 0] [--limit N]"""
import sys, os, json, gzip, asyncio, argparse, re
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "<path>")
from dcfr_lib import Client, render, read_jsonl_gz
from domains import DOMAINS

SYS = "You are a careful decision analyst. Answer exactly in the requested format."
PARA = 0


def dom(sc):
    if sc.get("dataset") or sc.get("track") == "real":  # env-specific wording: refer to the definition in the text
        return {"adverse": "the adverse case (as defined above) occurs", "population": "such cases"}
    d = dict(DOMAINS.get(sc.get("domain"), {}))
    d.update(sc.get("domain_info") or {})
    for k in ("adverse", "population"):
        if sc.get(k):
            d[k] = sc[k]
    d.setdefault("adverse", "the adverse outcome occurs")
    d.setdefault("population", "comparable cases")
    return d


def parse_pct(t):
    m = re.search(r"(\d+(?:\.\d+)?)", t or "")
    return None if not m else min(1.0, max(0.0, float(m.group(1)) / 100))


async def ask(cli, body, q, k):
    u = body + "\n\n" + q + "\nAnswer with a single percentage between 0% and 100% (e.g. 37%), nothing else."
    pr = render(cli.model, SYS, u, prefill="Answer: ")
    tasks = [cli.complete(pr, max_tokens=6, logprobs=0)]
    if k:
        tasks.append(cli.complete(pr, max_tokens=6, logprobs=0, temperature=0.7, n=k, seed=1))
    rs = await asyncio.gather(*tasks)
    try:
        g = parse_pct(rs[0]["choices"][0]["text"])
    except Exception:
        g = None
    samp = []
    if k:
        try:
            samp = [parse_pct(c["text"]) for c in rs[1]["choices"]]
        except Exception:
            samp = []
    return [g] + samp


async def elicit_joint(cli, sc, k):
    """Contrastive joint elicitation: the model states, for ALL checks in one answer, the share of adverse cases among
    cases where the check shows its finding and among cases where it does not (conditional form), plus prevalence."""
    from run_factored_ms import ask_list
    D = dom(sc)
    body = sc["text"]
    items = [("obs", i, o) for i, o in enumerate(sc.get("observed", []))] + [("chk", c["label"], c) for c in sc["checks"]]
    lines = "\n".join(f"{n+1}. {c['name']}: {c['finding_text']}" for n, (_, _, c) in enumerate(items))
    T = {"prior": ask(cli, body, f"Question: Based on the information above, what share of {D['population']} end up in the case where {D['adverse']}?", k)}
    base = f"Consider these findings:\n{lines}\n"
    T["c1"] = ask_list(cli, body, base + f"Question: Based on your general knowledge, comparing the findings with each other, for each finding (in order) estimate the share of cases WITH that finding that end up in the case where {D['adverse']}.", len(items), "percentages")
    T["c0"] = ask_list(cli, body, base + f"Question: Based on your general knowledge, comparing the findings with each other, for each finding (in order) estimate the share of cases WITHOUT that finding that end up in the case where {D['adverse']}.", len(items), "percentages")
    T["pf"] = ask_list(cli, body, base + f"Question: Based on your general knowledge, for each finding (in order) estimate the share of {D['population']} that show it.", len(items), "percentages")
    keys = list(T); rs = await asyncio.gather(*[T[x] for x in keys]); R = dict(zip(keys, rs))
    out = {"id": sc["id"], "form": "cond", "k": k, "joint": True, "prior": R["prior"], "obs": [], "chk": {}}
    for n, (kind, key, c) in enumerate(items):
        rec = {}
        for q in ("c1", "c0", "pf"):
            v = R[q]
            rec[q] = [min(1.0, max(0.0, v[n] / 100))] if v is not None and n < len(v) else [None]
        if kind == "obs":
            out["obs"].append(rec)
        else:
            out["chk"][key] = rec
    return out


async def elicit(cli, sc, form, k):
    if form == "joint":
        return await elicit_joint(cli, sc, k)
    D = dom(sc)
    body = sc["text"]
    out = {"id": sc["id"], "form": form, "k": k}
    tasks = {"prior": ask(cli, body, f"Question: Based on the information above, what share of {D['population']} end up in the case where {D['adverse']}?", k)}
    items = [("obs", i, o) for i, o in enumerate(sc.get("observed", []))] + [("chk", c["label"], c) for c in sc["checks"]]
    for kind, key, c in items:
        f = c["finding_text"]
        if form in ("lik", "both"):
            tasks[(kind, key, "l1")] = ask(cli, body, f"Question: Based on the information above and your general knowledge, among cases where {D['adverse']}, what share would show this finding for the check '{c['name']}': {f}?", k)
            tasks[(kind, key, "l0")] = ask(cli, body, f"Question: Based on the information above and your general knowledge, among cases where it is NOT the case that {D['adverse']}, what share would show this finding for the check '{c['name']}': {f}?", k)
        if form in ("cond", "both") and PARA == 1:
            tasks[(kind, key, "c1")] = ask(cli, body, f"Question: Using what you know about such cases, if the check '{c['name']}' finds that {f}, how likely is it that {D['adverse']}? Give a percentage.", k)
            tasks[(kind, key, "c0")] = ask(cli, body, f"Question: Using what you know about such cases, if the check '{c['name']}' does not find that {f}, how likely is it that {D['adverse']}? Give a percentage.", k)
            tasks[(kind, key, "pf")] = ask(cli, body, f"Question: Using what you know, how common is it among {D['population']} that {f}? Give a percentage.", k)
        elif form in ("cond", "both"):
            tasks[(kind, key, "c1")] = ask(cli, body, f"Question: Based on the information above and your general knowledge, among cases where the check '{c['name']}' shows this finding ({f}), what share end up in the case where {D['adverse']}?", k)
            tasks[(kind, key, "c0")] = ask(cli, body, f"Question: Based on the information above and your general knowledge, among cases where the check '{c['name']}' does NOT show this finding (it is not the case that {f}), what share end up in the case where {D['adverse']}?", k)
            tasks[(kind, key, "pf")] = ask(cli, body, f"Question: Based on the information above and your general knowledge, among {D['population']}, what share would show this finding for the check '{c['name']}': {f}?", k)
    keys = list(tasks)
    rs = await asyncio.gather(*[tasks[x] for x in keys])
    R = dict(zip(keys, rs))
    out["prior"] = R["prior"]
    out["obs"] = []
    out["chk"] = {}
    for kind, key, c in items:
        rec = {x: R[(kind, key, x)] for x in ("l1", "l0", "c1", "c0", "pf") if (kind, key, x) in R}
        if kind == "obs":
            out["obs"].append(rec)
        else:
            out["chk"][key] = rec
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("env"); ap.add_argument("out")
    ap.add_argument("--form", default="both"); ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--conc", type=int, default=48)
    ap.add_argument("--para", type=int, default=0)
    args = ap.parse_args()
    global PARA
    PARA = args.para
    scs = read_jsonl_gz(args.env)[: args.limit or None]
    done = {r["id"] for r in read_jsonl_gz(args.out)}
    todo = [s for s in scs if s["id"] not in done]
    print(f"{args.model}: {len(todo)} to do", flush=True)
    async with Client(args.model, conc=args.conc) as cli:
        sem = asyncio.Semaphore(16); buf = []

        async def one(sc):
            async with sem:
                try:
                    r = await elicit(cli, sc, args.form, args.k)
                except Exception as e:
                    print("ERR", sc["id"], repr(e), flush=True); return
                buf.append(r)
                if len(buf) >= 20:
                    rows = buf[:]; buf.clear()
                    with gzip.open(args.out, "at") as f:
                        f.write("".join(json.dumps(x) + "\n" for x in rows))
        await asyncio.gather(*[one(s) for s in todo])
        if buf:
            with gzip.open(args.out, "at") as f:
                f.write("".join(json.dumps(x) + "\n" for x in buf))
    print("done", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
