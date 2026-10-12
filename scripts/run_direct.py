"""Direct LLM acquisition policy (first step), letter-logprob over the next action with the STOP option permuted
(first / middle / last) and checks rotated across 3 permutations (fixes R1-M6). Sequential wording.
Stores the averaged distribution over original labels + 'S' and the captured letter mass.
usage: run_direct.py MODEL ENV OUT [--limit N] [--offset N]"""
import sys, os, json, gzip, asyncio, argparse, math, re
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "<path>")
from dcfr_lib import Client, render, read_jsonl_gz

SYS = "You are a careful decision analyst. Answer exactly in the requested format."
LET = "ABCDEFGHIJKLMNOP"


def strip_label(line):
    return re.sub(r"^\([A-Z]\)\s*", "", line)


QUAL = False


def build(sc, perm_idx):
    tp = sc["text_parts"]
    labs = [c["label"] for c in sc["checks"]]
    m = len(labs)
    rot = labs[perm_idx % m:] + labs[: perm_idx % m] if perm_idx else labs
    pos = [m, 0, m // 2][perm_idx % 3]  # stop last / first / middle
    seq = rot[:pos] + ["S"] + rot[pos:]
    stop_body = strip_label(tp.get("stop_line", "(S) Stop and decide now."))
    lines, mapping = [], {}
    for i, k in enumerate(seq):
        L = LET[i]
        mapping[L] = k
        body = stop_body if k == "S" else strip_label(tp["option_lines"][k])
        lines.append(f"({L}) {body}")
    skip = {"options", "option_lines", "stop_line", "budget", "note_optional"}
    skip.add("beliefs_numeric" if (QUAL and tp.get("beliefs_qualitative")) else "beliefs_qualitative")
    head = "\n\n".join(v for k, v in tp.items() if k not in skip and isinstance(v, str) and v.strip())
    intro = tp.get("options", "").split("\n")[0]
    txt = (head + "\n\n" + intro + "\n" + "\n".join(lines) +
           "\n\nYou may run checks one at a time, see each result, and stop whenever you want."
           "\nQuestion: What do you do next? Answer with the letter of one option only.")
    if PARA:  # F4 (reviewer S6): paraphrased closing instruction; default (0) unchanged
        txt = txt.replace("\n\nYou may run checks one at a time, see each result, and stop whenever you want."
                          "\nQuestion: What do you do next? Answer with the letter of one option only.", PARA_TXT[PARA])
    return txt, mapping


PARA = 0
PARA_TXT = {1: "\n\nChecks can be run one after another; after each one you see its result, and you can stop at any point."
               "\nQuestion: Which option do you choose now? Reply with a single option letter.",
            2: "\n\nYou can request checks sequentially, observe each outcome, and make your decision at any time."
               "\nQuestion: What is your next move? Respond only with the letter of the option you pick."}


THINK = False


async def one_think(cli, sc, idx):
    """Native reasoning: one permutation per scenario (rotated over scenarios), reasoning enabled, answer parsed."""
    pi = idx % 3
    txt, mp = build(sc, pi)
    txt = txt.replace("Answer with the letter of one option only.", "Think it through, then end with 'Answer: (X)' where X is the letter of one option.")
    pr = render(cli.model, SYS, txt, think=True, effort="medium")
    r = await cli.complete(pr, max_tokens=6000, logprobs=0)
    try:
        t = r["choices"][0]["text"]
    except Exception:
        t = ""
    import re as _re
    final = t.split("</think>")[-1].split("<|channel|>final<|message|>")[-1]
    m = _re.findall(r"Answer:\s*\(?([A-Z])\)?", final) or _re.findall(r"\(([A-Z])\)", final[-200:])
    ch = mp.get(m[-1]) if m else None
    forced = False
    if ch is None:  # reasoning did not finish or no parsable answer: force the answer after the reasoning so far
        mdl = cli.model.lower()
        if "gpt-oss" in mdl:
            cont = pr + t + "<|end|><|start|>assistant<|channel|>final<|message|>Answer: ("
        elif "</think>" in t:
            cont = pr + t + "\n\nAnswer: ("
        else:
            cont = pr + t + "\n</think>\n\nAnswer: ("
        r2 = await cli.complete(cont, max_tokens=1, logprobs=20)
        try:
            top = r2["choices"][0]["logprobs"]["top_logprobs"][0]
            best = max(((k.strip().strip("()"), v) for k, v in top.items() if k.strip().strip("()") in mp), key=lambda x: x[1], default=None)
            ch = mp[best[0]] if best else None; forced = ch is not None
        except Exception:
            ch = None
    return {"id": sc["id"], "dist": ({ch: 1.0} if ch else {}), "mass": [1.0 if ch else 0.0], "think_tokens": len(t) // 4, "forced": forced}


async def one(cli, sc):
    outs = []
    for pi in range(3):
        txt, mp = build(sc, pi)
        outs.append((cli.complete(render(cli.model, SYS, txt, prefill="Answer: ("), max_tokens=1, logprobs=20), mp))
    rs = await asyncio.gather(*[o[0] for o in outs])
    agg, masses = {}, []
    for r, (_, mp) in zip(rs, outs):
        try:
            top = r["choices"][0]["logprobs"]["top_logprobs"][0]
        except Exception:
            continue
        p = {L: 0.0 for L in mp}
        for t, lp in top.items():
            k = t.strip().strip("(").strip(")")
            if k in p:
                p[k] += math.exp(lp)
        z = sum(p.values()); masses.append(z)
        if z <= 0:
            continue
        for L, v in p.items():
            agg[mp[L]] = agg.get(mp[L], 0) + v / z / 3
    return {"id": sc["id"], "dist": agg, "mass": masses}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("env"); ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--think", action="store_true"); ap.add_argument("--qual", action="store_true")
    ap.add_argument("--para", type=int, default=0)
    a = ap.parse_args()
    global THINK, QUAL, PARA
    PARA = a.para
    QUAL = a.qual
    THINK = a.think
    scs = read_jsonl_gz(a.env)[a.offset:]
    scs = scs[: a.limit or None]
    done = {r["id"] for r in read_jsonl_gz(a.out)}
    todo = [s for s in scs if s["id"] not in done]
    print(f"{a.model}: {len(todo)} to do", flush=True)
    async with Client(a.model, conc=48) as cli:
        sem = asyncio.Semaphore(32); buf = []

        idx_of = {s["id"]: k for k, s in enumerate(scs)}

        async def go(sc):
            async with sem:
                try:
                    r = await (one_think(cli, sc, idx_of[sc["id"]]) if THINK else one(cli, sc))
                except Exception as e:
                    print("ERR", sc["id"], repr(e), flush=True); return
                buf.append(r)
                if len(buf) >= 40:
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
