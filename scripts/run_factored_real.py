"""FIO-VOI elicitation for multi-valued (binned) checks with UNSTATED reliabilities (envs_real; also usable for any
scenario whose checks carry `bins`). The LLM supplies single-feature conditionals:
  prior  = share of adverse cases (base rate),
  c_obs  = P(adverse | observed fact) for each already-known fact,
  c_bin  = P(adverse | check falls in bin b) for each candidate check and bin,
which exact Bayes turns into likelihood ratios (naive-Bayes combination; bins are terciles so P(bin) = p_bins or 1/3).
usage: run_factored_real.py MODEL ENV OUT [--k 0] [--limit N]"""
import sys, os, json, gzip, asyncio, argparse
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "<path>")
from dcfr_lib import Client, read_jsonl_gz
from run_factored_k import ask
from run_factored_ms import ask_list

ADV = "the adverse case (as defined above)"


def obs_text(o):
    for k in ("text", "value_text", "finding_text"):
        if o.get(k):
            return o[k]
    return f"{o.get('name', 'the item')}: {o.get('value')}"


JOINT = False


async def elicit_joint_check(cli, body, c, k):
    """Elicit the adverse share for ALL outcome bins of one check in a single answer (contrastive, joint)."""
    bins = c.get("bins", [])
    lines = "; ".join(f"({chr(97+j)}) {b['text']}" for j, b in enumerate(bins))
    q = (f"Question: Ignore the other facts. Based on your general knowledge, consider the request '{c['name']}' "
         f"({c.get('ask', '')}). For each of its possible results, estimate the share of such cases that end up in {ADV}: {lines}.")
    outs = [ask_list(cli, body, q, len(bins), "percentages, one per result in order")]
    v = await outs[0]
    if v is None:
        return [[None] for _ in bins]
    return [[min(1.0, max(0.0, x / 100))] for x in v]


CONTRAST = False


async def elicit_ratio_check(cli, body, c, k, pri_text):
    """Contrastive elicitation: relative risk of the adverse case between the top and bottom bins of a check."""
    bins = c.get("bins", [])
    if len(bins) < 2:
        return [[None] for _ in bins]
    q = (f"Question: Ignore the other facts. Based on your general knowledge, compare cases like this where {bins[-1]['text']} "
         f"with cases where {bins[0]['text']}. How many times as likely is {ADV} in the first group as in the second group? "
         f"Answer with a single positive number (for example 1.8 if it is more likely, 0.6 if it is less likely), nothing else.")
    u = body + "\n\n" + q
    from dcfr_lib import render
    r = await cli.complete(render(cli.model, "You are a careful decision analyst. Answer exactly in the requested format.", u, prefill="Answer: "), max_tokens=8, logprobs=0)
    import re as _re
    try:
        m = _re.search(r"(\d+(?:\.\d+)?)", r["choices"][0]["text"])
        rho = max(0.05, min(20.0, float(m.group(1))))
    except Exception:
        return [[None] for _ in bins]
    return [["RATIO", rho]] + [[None]] * (len(bins) - 2) + [["RATIO", rho]]


async def elicit(cli, sc, k):
    body = sc["text"]
    T = {"prior": ask(cli, body, f"Question: Based on the information above, what share of such cases end up in {ADV}?", k)}
    for i, o in enumerate(sc.get("observed", [])):
        T[("obs", i)] = ask(cli, body, f"Question: Ignore the other facts. Based on your general knowledge, among cases like this where {obs_text(o)}, what share end up in {ADV}?", k)
    for c in sc["checks"]:
        if JOINT:
            T[("chkj", c["label"])] = elicit_joint_check(cli, body, c, k)
            continue
        if CONTRAST:
            T[("chkr", c["label"])] = elicit_ratio_check(cli, body, c, k, None)
            continue
        for b in c.get("bins", []):
            T[("chk", c["label"], b["bin"])] = ask(cli, body, f"Question: Ignore the other facts. Based on your general knowledge, among cases like this where {b['text']}, what share end up in {ADV}?", k)
    keys = list(T)
    rs = await asyncio.gather(*[T[x] for x in keys])
    R = dict(zip(keys, rs))
    out = {"id": sc["id"], "k": k, "prior": R["prior"], "obs": [R[("obs", i)] for i in range(len(sc.get("observed", [])))],
           "chk": {c["label"]: (R[("chkj", c["label"])] if JOINT else (R[("chkr", c["label"])] if CONTRAST else [R[("chk", c["label"], b["bin"])] for b in c.get("bins", [])])) for c in sc["checks"]}}
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("env"); ap.add_argument("out")
    ap.add_argument("--k", type=int, default=0); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conc", type=int, default=48); ap.add_argument("--joint", action="store_true")
    ap.add_argument("--offset", type=int, default=0); ap.add_argument("--contrast", action="store_true")
    a = ap.parse_args()
    global JOINT, CONTRAST
    JOINT = a.joint; CONTRAST = a.contrast
    scs = read_jsonl_gz(a.env)[a.offset:]
    scs = scs[: a.limit or None]
    done = {r["id"] for r in read_jsonl_gz(a.out)}
    todo = [s for s in scs if s["id"] not in done]
    print(f"{a.model}: {len(todo)} to do", flush=True)
    async with Client(a.model, conc=a.conc) as cli:
        sem = asyncio.Semaphore(16); buf = []

        async def one(sc):
            async with sem:
                try:
                    r = await elicit(cli, sc, a.k)
                except Exception as e:
                    print("ERR", sc["id"], repr(e), flush=True); return
                buf.append(r)
                if len(buf) >= 20:
                    rows = buf[:]; buf.clear()
                    with gzip.open(a.out, "at") as f:
                        f.write("".join(json.dumps(x) + "\n" for x in rows))
        await asyncio.gather(*[one(s) for s in todo])
        if buf:
            with gzip.open(a.out, "at") as f:
                f.write("".join(json.dumps(x) + "\n" for x in buf))
    print("done", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
