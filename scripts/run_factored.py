"""Factored elicitation for IO-VOI: the LLM specifies the decision model (base rate + per-check likelihoods), exact Bayes
does the inference. Elicits, via 11-bin letter logprobs (two orders): prior P(Y=1); for every check (candidate and
already observed) P(finding | Y=1) and P(finding | Y=0). Resumable. usage: run_factored.py MODEL ENV OUT"""
import sys, os, json, gzip, asyncio, argparse, math
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "<path>")
from dcfr_lib import Client, render, read_jsonl_gz
from verbalize import scenario_text
from domains import DOMAINS
from run_probe import dist, SYS, BL

VALS = [i / 10 for i in range(11)]


NUMERIC = False


async def ask_num(cli, body, q):
    u = body + "\n\n" + q + "\nAnswer with a single percentage between 0% and 100% (e.g. 37%), nothing else."
    r = await cli.complete(render(cli.model, SYS, u, prefill="Answer: "), max_tokens=6, logprobs=0)
    try:
        t = r["choices"][0]["text"]
    except Exception:
        return None
    import re
    m = re.search(r"(\d+(?:\.\d+)?)", t)
    if not m:
        return None
    return min(1.0, max(0.0, float(m.group(1)) / 100))


async def ask_bins(cli, body, q):
    if NUMERIC:
        return await ask_num(cli, body, q)
    outs = []
    for desc in (False, True):
        vals = VALS[::-1] if desc else VALS
        ol = "\n".join(f"({BL[k]}) {int(round(100*v))}%" for k, v in enumerate(vals))
        u = body + "\n\n" + q + "\nOptions:\n" + ol + "\n\nAnswer with the letter of the closest option only."
        outs.append((cli.complete(render(cli.model, SYS, u, prefill="Answer: ("), max_tokens=1, logprobs=20), vals))
    rs = await asyncio.gather(*[o[0] for o in outs])
    ps = []
    for r, (_, vals) in zip(rs, outs):
        d = dist(r, BL)
        if d is None:
            return None
        ps.append(sum(d[0][BL[k]] * vals[k] for k in range(11)))
    return sum(ps) / 2


PARA = 0


async def elicit(cli, sc):
    D = DOMAINS[sc["domain"]]
    body = sc["text"] if sc.get("text") else scenario_text(sc)
    if PARA == 1:
        tasks = [ask_bins(cli, body, f"Question: From the text, how common is it that {D['adverse']} among {D['population']}? Give it as a percentage.")]
    elif PARA == 2:
        tasks = [ask_bins(cli, body, f"Question: Before running any further check, what fraction of {D['population']} end up with {D['adverse']}, as stated above?")]
    else:
        tasks = [ask_bins(cli, body, f"Question: According to the information above, what is the base rate, i.e. the share of {D['population']} in which {D['adverse']}?")]
    items = [("obs", i, o) for i, o in enumerate(sc.get("observed", []))] + [("chk", c["label"], c) for c in sc["checks"]]
    for kind, key, c in items:
        for y, cond in ((1, f"cases where {D['adverse']}"), (0, f"the other cases (where it is not the case that {D['adverse']})")):
            if PARA == 1:
                q = (f"Question: From the text, if we restrict attention to {cond}, how often does the check '{c['name']}' report "
                     f"that {c['finding_text']}? Give it as a percentage.")
            elif PARA == 2:
                q = (f"Question: Consider only {cond}. In what fraction of them would the check '{c['name']}' come back with the "
                     f"finding '{c['finding_text']}', as stated above?")
            else:
                q = (f"Question: According to the information above, among {cond}, what share show this finding for the check "
                     f"'{c['name']}': {c['finding_text']}?")
            tasks.append(ask_bins(cli, body, q))
    rs = await asyncio.gather(*tasks)
    out = {"id": sc["id"], "prior": rs[0], "obs": [], "chk": {}}
    k = 1
    for kind, key, c in items:
        l1, l0 = rs[k], rs[k + 1]; k += 2
        if kind == "obs":
            out["obs"].append([l1, l0])
        else:
            out["chk"][key] = [l1, l0]
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("env"); ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--conc", type=int, default=48)
    ap.add_argument("--numeric", action="store_true"); ap.add_argument("--para", type=int, default=0)
    args = ap.parse_args()
    global NUMERIC, PARA
    NUMERIC = args.numeric; PARA = args.para
    scs = read_jsonl_gz(args.env)[: args.limit or None]
    done = {r["id"] for r in read_jsonl_gz(args.out)}
    todo = [s for s in scs if s["id"] not in done]
    print(f"{args.model}: {len(todo)} to do", flush=True)
    async with Client(args.model, conc=args.conc) as cli:
        sem = asyncio.Semaphore(24); buf = []

        async def one(sc):
            async with sem:
                try:
                    r = await elicit(cli, sc)
                except Exception as e:
                    print("ERR", sc["id"], e, flush=True); return
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
