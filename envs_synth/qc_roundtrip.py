"""QC round-trip: can a strong served LLM read back the numbers that define each scenario from its text?
This is a FILTER on the verbalization, never a label source (all labels come from core.py enumeration).

For every row the model is asked (temperature 0, JSON output) to extract:
  binary : base-rate percent, gain/loss points, and per check: cost, P(finding | adverse) %, P(finding | not) %
  ms     : per request cost; per factor-level prior percent (numeric text); one probed likelihood cell per request;
           one probed payoff cell; and from the qualitative text the verbal phrase given for each factor level.
A row passes if every extracted number equals the generator's value (tolerance 0.51 percentage points / 1e-6 cost)
and (ms) every phrase matches. Output: data/qc/{track}_{domain}_{split}.jsonl.gz (id, model, pass, errors, raw).

usage: python qc_roundtrip.py --files data/ms/finance_test.jsonl.gz [...] --model Qwen/Qwen3-32B-AWQ [--conc 48] [--limit N]
Endpoints: infoseek/serving/endpoints.json merged with naacl_dcfr/logs/servers_launched.tsv (health-checked).
"""
import argparse, asyncio, gzip, json, os, random, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "<path>"
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, HERE)
import dcfr_lib as L

SYS = "You extract structured facts from documents exactly as stated. Output only one JSON object, no commentary."


def endpoints(model):
    urls = []
    p = f"{ROOT}/infoseek/serving/endpoints.json"
    if os.path.exists(p):
        try:
            urls += json.load(open(p)).get(model, [])
        except Exception:
            pass
    urls += L.endpoints().get(model, [])
    return list(dict.fromkeys(urls))


def probes(row):
    rng = random.Random(sum(map(ord, row["id"])))
    if row["track"] == "binary":
        return {}
    out = {"lik": [], "pay": None}
    for c in row["checks"]:
        f = [x for x in row["factors"] if x["key"] == c["factor"]][0]
        li = rng.randrange(len(f["levels"]))
        oi = rng.randrange(len(c["outcomes"]))
        out["lik"].append({"label": c["label"], "level": f["levels"][li], "factor": f["key"], "result": c["outcomes"][oi],
                           "value": round(c["lik_by_level"][li][oi] * 100)})
    return out


def prompt_binary(row):
    labs = row["label_order"]
    schema = ('{"base_rate_percent": number, "gain_points": number, "loss_points": number, "checks": {' +
              ", ".join(f'"{l}": {{"cost": number, "if_adverse_percent": number, "otherwise_percent": number}}' for l in labs) + "}}")
    return (f"DOCUMENT:\n{row['text']}\n\nExtract: the base-rate percentage of the adverse case; the points gained and lost "
            "(as positive numbers) for the action with uncertain payoff; and for each available check (A, B, ...) its cost in points "
            "(0 if free) and how often (in percent) it reports its finding in the adverse case and otherwise.\n"
            f"Return JSON exactly in this form: {schema}")


def prompt_ms(row, pr, qual=False):
    labs = row["label_order"]
    if qual:
        fac = "; ".join(f"{f['name']}: levels " + ", ".join(f"'{lv}'" for lv in f["levels"]) for f in row["factors"])
        return (f"DOCUMENT:\n{row['text_qualitative']}\n\nFor each factor in the current assessment, give the exact verbal "
                f"likelihood phrase the document uses for each level (e.g. 'likely'). Factors and levels: {fac}.\n"
                'Return JSON: {"phrases": {"<factor key>": {"<level>": "<phrase>"}}} using factor keys ' +
                ", ".join(f"'{f['key']}' ({f['name']})" for f in row["factors"]) + ".")
    fac = ", ".join(f"'{f['key']}' ({f['name']}; levels " + ", ".join(f"'{lv}'" for lv in f["levels"]) + ")" for f in row["factors"])
    q = "; ".join(f"request {x['label']}: probability (percent) of result '{x['result']}' when the {x['factor']} factor is '{x['level']}'"
                  for x in pr["lik"])
    return (f"DOCUMENT:\n{row['text_numeric']}\n\nExtract: (1) the cost of each request {', '.join(labs)} (0 if free, in the "
            f"document's units, as a number); (2) the current probability in percent of each level of each factor {fac}; "
            f"(3) these likelihoods: {q}; (4) the maximum number of requests allowed (null if no limit is stated).\n"
            'Return JSON: {"costs": {"A": number, ...}, "priors": {"<factor key>": {"<level>": number}}, '
            '"lik": {"A": number, ...}, "budget": number or null}')


def close(a, b, tol):
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def check_binary(row, js):
    err = []
    if not isinstance(js, dict):
        return ["no_json"]
    if not close(js.get("base_rate_percent"), row["prior"] * 100, 0.51): err.append("base_rate")
    if not close(js.get("gain_points"), row["g"], 1e-6): err.append("gain")
    if not close(abs(float(js.get("loss_points") or 1e9)), row["l"], 1e-6): err.append("loss")
    ch = js.get("checks") or {}
    for c in row["checks"]:
        x = ch.get(c["label"]) or {}
        if not close(x.get("cost"), c["cost"], 1e-6): err.append(f"cost_{c['label']}")
        if not close(x.get("if_adverse_percent"), c["s1"] * 100, 0.51): err.append(f"s1_{c['label']}")
        if not close(x.get("otherwise_percent"), c["s0"] * 100, 0.51): err.append(f"s0_{c['label']}")
    return err


def check_ms(row, js, pr):
    err = []
    if not isinstance(js, dict):
        return ["no_json"]
    costs = js.get("costs") or {}
    for c in row["checks"]:
        if not close(costs.get(c["label"]), c["cost"], 1e-6): err.append(f"cost_{c['label']}")
    pri = js.get("priors") or {}
    for f in row["factors"]:
        d = pri.get(f["key"]) or {}
        dl = {str(k).strip().lower(): v for k, v in d.items()} if isinstance(d, dict) else {}
        for lv, p in zip(f["levels"], f["prior"]):
            if not close(dl.get(lv.lower()), p * 100, 0.51): err.append(f"prior_{f['key']}")
    lk = js.get("lik") or {}
    for x in pr["lik"]:
        if not close(lk.get(x["label"]), x["value"], 0.51): err.append(f"lik_{x['label']}")
    b = js.get("budget")
    want = row["budget"] if row["budget"] < len(row["label_order"]) else None
    if (want is None and b not in (None, "null", len(row["label_order"]))) or (want is not None and not close(b, want, 1e-6)):
        err.append("budget")
    return sorted(set(err))


def check_qual(row, js):
    from gen_ms import verbal_bin
    if not isinstance(js, dict):
        return ["no_json"]
    ph = js.get("phrases") or {}
    err = []
    for f in row["factors"]:
        d = ph.get(f["key"]) or {}
        dl = {str(k).strip().lower(): str(v).strip().lower() for k, v in d.items()} if isinstance(d, dict) else {}
        for lv, p in zip(f["levels"], f["prior"]):
            if dl.get(lv.lower()) != verbal_bin(p)[0]:
                err.append(f"phrase_{f['key']}")
    return sorted(set(err))


async def run(rows, model, conc, out_path, max_tokens):
    done = set()
    import glob
    want = {r["id"] for r in rows}
    for p in glob.glob(os.path.join(os.path.dirname(out_path), "*." + out_path.split(".")[-3] + ".jsonl.gz")):
        try:
            for l in gzip.open(p, "rt"):
                try:
                    done.add(json.loads(l)["id"])
                except Exception:
                    pass
        except Exception:
            pass
    done &= want
    todo = [r for r in rows if r["id"] not in done]
    print(f"{out_path}: {len(done)} done, {len(todo)} to do", flush=True)
    think_kw = {}
    async with L.Client(model, urls=endpoints(model), conc=conc) as cli:
        buf, t0, n_ok = [], time.time(), 0

        async def one(r):
            pr = probes(r)
            if r["track"] == "binary":
                ps = [("num", prompt_binary(r))]
            else:
                ps = [("num", prompt_ms(r, pr)), ("qual", prompt_ms(r, pr, qual=True))]
            res = {"id": r["id"], "model": model, "errors": [], "raw": {}}
            for tag, p in ps:
                s = L.render(model, SYS, p, think=False)
                o = await cli.complete(s, max_tokens=max_tokens, temperature=0.0, logprobs=None)
                txt = (o.get("choices") or [{}])[0].get("text") if isinstance(o, dict) else None
                js = L.parse_json_obj(txt)
                if r["track"] == "binary":
                    e = check_binary(r, js)
                elif tag == "num":
                    e = check_ms(r, js, pr)
                else:
                    e = check_qual(r, js)
                res["errors"] += [f"{tag}:{x}" for x in e]
                res["raw"][tag] = (txt or "")[-1500:]
            res["pass"] = not res["errors"]
            res["pass_numeric"] = not any(x.startswith("num:") for x in res["errors"])
            return res

        tasks = [asyncio.ensure_future(one(r)) for r in todo]
        for k, fut in enumerate(asyncio.as_completed(tasks)):
            res = await fut
            n_ok += res["pass"]
            buf.append(res)
            if len(buf) >= 100 or k == len(tasks) - 1:
                L.write_jsonl_gz(out_path, buf)
                buf = []
                print(f"{k + 1}/{len(tasks)} pass={n_ok / (k + 1):.3f} {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-32B-AWQ")
    ap.add_argument("--conc", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=700)
    a = ap.parse_args()
    os.makedirs(f"{HERE}/data/qc", exist_ok=True)
    for p in a.files:
        rows = [json.loads(l) for l in gzip.open(p, "rt")]
        if a.limit:
            rows = rows[:a.limit]
        track = rows[0]["track"]
        name = os.path.basename(p).replace(".jsonl.gz", "")
        tag = a.model.split("/")[-1]
        out = f"{HERE}/data/qc/{track}_{name}.{tag}.jsonl.gz"
        asyncio.run(run(rows, a.model, a.conc, out, a.max_tokens))


if __name__ == "__main__":
    main()
