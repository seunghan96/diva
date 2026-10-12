"""DIVA elicitation for the multi-state track (factored hidden state, 3 actions, multi-outcome checks).
Question templates are instantiated from the scenario schema (factor names/levels, check outcomes, actions); every NUMBER
comes from the LLM: current factor probabilities, P(outcome | level of the factor the check reports on), and additive payoff
contributions per action x factor level plus a per-action constant. Exact computation happens at evaluation time.
usage: run_factored_ms.py MODEL ENV OUT [--qual] [--limit N]"""
import sys, os, json, gzip, asyncio, argparse, re
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "<path>")
from dcfr_lib import Client, render, read_jsonl_gz

SYS = "You are a careful decision analyst. Answer exactly in the requested format."


def nums(t, n):
    xs = re.findall(r"-?\d+(?:\.\d+)?", (t or "").replace(",", " , ").replace("$", ""))
    xs = [float(x) for x in xs]
    return xs[:n] if len(xs) >= n else None


async def ask_list(cli, body, q, n, fmt):
    u = body + "\n\n" + q + f"\nAnswer with exactly {n} numbers separated by commas ({fmt}), nothing else."
    r = await cli.complete(render(cli.model, SYS, u, prefill="Answer: "), max_tokens=12 * n, logprobs=0, stop=["\n"])
    try:
        return nums(r["choices"][0]["text"], n)
    except Exception:
        return None


async def elicit(cli, sc, qual):
    body = sc["text_qualitative"] if qual and sc.get("text_qualitative") else sc["text"]
    fac = {f["key"]: f for f in sc["factors"]}
    T = {}
    for f in sc["factors"]:
        lv = f["levels"]
        T[("prior", f["key"])] = ask_list(cli, body, f"Question: Based on the information above, what are the current probabilities (in %) that {f['name']} is " + ", ".join(f"'{x}'" for x in lv) + ", respectively?", len(lv), "percentages")
    for c in sc["checks"]:
        f = fac.get(c.get("factor"))
        if f is None:
            continue
        for j, lev in enumerate(f["levels"]):
            T[("lik", c["label"], j)] = ask_list(cli, body, f"Question: Based on the information above, if {f['name']} is '{lev}', how often (in %) does request ({c['label']}) {c['name']} give the results " + ", ".join(f"'{o}'" for o in c["outcomes"]) + ", respectively?", len(c["outcomes"]), "percentages")
    for a in sc["actions"]:
        for f in sc["factors"]:
            T[("u", a, f["key"])] = ask_list(cli, body, f"Question: Based on the information above, if you choose {a}, what does {f['name']} contribute to the payoff (in thousand dollars or the stated payoff units; 0 if it does not matter) when it is " + ", ".join(f"'{x}'" for x in f["levels"]) + ", respectively?", len(f["levels"]), "signed numbers")
        T[("c", a)] = ask_list(cli, body, f"Question: Based on the information above, what fixed amount (in the stated payoff units, negative for costs such as trading costs; 0 if none) is added to the payoff of {a} regardless of the factors?", 1, "a signed number")
    keys = list(T)
    rs = await asyncio.gather(*[T[k] for k in keys])
    R = dict(zip(keys, rs))
    out = {"id": sc["id"], "qual": qual, "prior": {}, "lik": {}, "u": {}, "c": {}}
    for k, v in R.items():
        if k[0] == "prior":
            out["prior"][k[1]] = v
        elif k[0] == "lik":
            out["lik"].setdefault(k[1], {})[str(k[2])] = v
        elif k[0] == "u":
            out["u"].setdefault(k[1], {})[k[2]] = v
        else:
            out["c"][k[1]] = v
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("env"); ap.add_argument("out")
    ap.add_argument("--qual", action="store_true"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    a = ap.parse_args()
    scs = read_jsonl_gz(a.env)[a.offset:]
    scs = scs[: a.limit or None]
    done = {r["id"] for r in read_jsonl_gz(a.out)}
    todo = [s for s in scs if s["id"] not in done]
    print(f"{a.model}: {len(todo)} to do", flush=True)
    async with Client(a.model, conc=48) as cli:
        sem = asyncio.Semaphore(16); buf = []

        async def go(sc):
            async with sem:
                try:
                    r = await elicit(cli, sc, a.qual)
                except Exception as e:
                    print("ERR", sc["id"], repr(e), flush=True); return
                buf.append(r)
                if len(buf) >= 20:
                    rows = buf[:]; buf.clear()
                    with gzip.open(a.out, "at") as f:
                        f.write("".join(json.dumps(x) + "\n" for x in rows))
        await asyncio.gather(*[go(s) for s in todo])
        if buf:
            with gzip.open(a.out, "at") as f:
                f.write("".join(json.dumps(x) + "\n" for x in buf))
    print("done", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
