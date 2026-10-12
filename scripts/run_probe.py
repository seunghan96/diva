"""Probe an LLM on controlled scenarios: elicits everything needed for IO-VOI, EIG-from-beliefs, and direct choice.
Per scenario stores: b0 (2 orders), b(check=1), b(check=0) for each check, predictive P(finding) per check (optional),
direct-choice letter distribution (2 label orders). Resumable; output jsonl.gz.
usage: run_probe.py MODEL ENV_FILE OUT [--pred] [--limit N]"""
import sys, os, json, gzip, asyncio, argparse, random, math
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "<path>")
from dcfr_lib import Client, render, read_jsonl_gz
from verbalize import belief_prompt, choice_prompt, scenario_text, LETTERS
from domains import DOMAINS

SYS = "You are a careful decision analyst. Answer exactly in the requested format."
BL = "ABCDEFGHIJK"


def dist(resp, letters):
    try:
        top = resp["choices"][0]["logprobs"]["top_logprobs"][0]
    except Exception:
        return None
    p = {L: 0.0 for L in letters}
    for t, lp in top.items():
        k = t.strip().strip("(").strip(")")
        if k in p:
            p[k] += math.exp(lp)
    z = sum(p.values())
    if z <= 0:
        return None
    return {k: v / z for k, v in p.items()}, z


async def belief(cli, sc, extra=None):
    outs = []
    for desc in (False, True):
        txt, vals = belief_prompt(sc, extra_observed=extra, descending=desc)
        outs.append((cli.complete(render(cli.model, SYS, txt, prefill="Answer: ("), max_tokens=1, logprobs=20), vals))
    rs = await asyncio.gather(*[o[0] for o in outs])
    bs, mass = [], []
    for r, (_, vals) in zip(rs, outs):
        d = dist(r, BL)
        if d is None:
            return None
        dd, z = d
        bs.append(sum(dd[BL[k]] * vals[k] for k in range(11)))
        mass.append(z)
    return {"b": (bs[0] + bs[1]) / 2, "b_asc": bs[0], "b_desc": bs[1], "mass": min(mass)}


async def predictive(cli, sc, c):
    """LLM predictive probability that check c shows its finding (two orders of a yes/no question)."""
    txt = scenario_text(sc, include_checks=False, include_payoff=False)
    q = (f"\n\nQuestion: If you ran the check '{c['name']}', how likely is it to show this finding: {c['finding_text']}?"
         f" Reliability: among cases where {DOMAINS[sc['domain']]['adverse']}, {int(round(100*c['s1']))}% show this finding;"
         f" among the other cases, {int(round(100*c['s0']))}% do.")
    outs = []
    for desc in (False, True):
        vals = [i / 10 for i in range(11)]
        if desc:
            vals = vals[::-1]
        ol = "\n".join(f"({BL[k]}) {int(round(100*v))}%" for k, v in enumerate(vals))
        u = txt + q + "\nOptions:\n" + ol + "\n\nAnswer with the letter of the closest option only."
        outs.append((cli.complete(render(cli.model, SYS, u, prefill="Answer: ("), max_tokens=1, logprobs=20), vals))
    rs = await asyncio.gather(*[o[0] for o in outs])
    ps = []
    for r, (_, vals) in zip(rs, outs):
        d = dist(r, BL)
        if d is None:
            return None
        ps.append(sum(d[0][BL[k]] * vals[k] for k in range(11)))
    return sum(ps) / 2


async def direct_choice(cli, sc, rng):
    m = len(sc["checks"])
    orders = [list(range(m)), list(range(m))[::-1]]
    res = {}
    for order in orders:
        txt, mapping = choice_prompt(sc, order=order)
        r = await cli.complete(render(cli.model, SYS, txt, prefill="Answer: ("), max_tokens=1, logprobs=20)
        d = dist(r, list(mapping.keys()))
        if d is None:
            return None
        for k, v in d[0].items():
            res[mapping[k]] = res.get(mapping[k], 0) + v / 2
    return res


async def probe(cli, sc, args, rng):
    out = {"id": sc["id"]}
    tasks = [belief(cli, sc)]
    for c in sc["checks"]:
        for v in (1, 0):
            o = dict(name=c["name"], finding_text=c["finding_text"], value=v, s1=c["s1"], s0=c["s0"])
            tasks.append(belief(cli, sc, extra=[o]))
    if args.pred:
        for c in sc["checks"]:
            tasks.append(predictive(cli, sc, c))
    tasks.append(direct_choice(cli, sc, rng))
    rs = await asyncio.gather(*tasks)
    out["b0"] = rs[0]
    k = 1
    out["bv"] = {}
    for c in sc["checks"]:
        out["bv"][c["label"]] = {"1": rs[k], "0": rs[k + 1]}
        k += 2
    if args.pred:
        out["pred"] = {}
        for c in sc["checks"]:
            out["pred"][c["label"]] = rs[k]; k += 1
    out["direct"] = rs[k]
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("env"); ap.add_argument("out")
    ap.add_argument("--pred", action="store_true"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conc", type=int, default=48)
    args = ap.parse_args()
    scs = read_jsonl_gz(args.env)
    if args.limit:
        scs = scs[:args.limit]
    done = {r["id"] for r in read_jsonl_gz(args.out)}
    todo = [s for s in scs if s["id"] not in done]
    print(f"{args.model}: {len(todo)} to do ({len(done)} done)", flush=True)
    rng = random.Random(0)
    async with Client(args.model, conc=args.conc) as cli:
        sem = asyncio.Semaphore(32)
        buf = []

        async def one(sc):
            async with sem:
                try:
                    r = await probe(cli, sc, args, rng)
                except Exception as e:
                    print("ERR", sc["id"], e, flush=True); return
                buf.append(r)
                if len(buf) >= 20:
                    rows = buf[:]; buf.clear()
                    with gzip.open(args.out, "at") as f:
                        for x in rows:
                            f.write(json.dumps(x) + "\n")
        await asyncio.gather(*[one(s) for s in todo])
        if buf:
            with gzip.open(args.out, "at") as f:
                for x in buf:
                    f.write(json.dumps(x) + "\n")
    print("done", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
