"""Realized-replay scenario files (reviewer S3, 2026-10-11): test (2,000) + fresh (1,000) + big (2,000) scenarios of
Taiwan / CDC, each with one 'episode' world equal to the actual record (state = y_true, outcomes = record values), so that
baselines.run --mode episode replays LLM policies on the real records. Writes envs_tab/data/{ds}_rz.jsonl.gz."""
import gzip, json, os
IS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for ds in ("taiwan", "cdc"):
    rows = []
    for part in ("test", "fresh", "big"):
        for l in gzip.open(f"{IS}/envs_tab/data/{ds}_{part}.jsonl.gz", "rt"):
            r = json.loads(l)
            vals = {c["label"]: c.get("record_value") for c in r["checks"]}
            if r.get("y_true") is None or any(v is None for v in vals.values()):
                continue
            r["episodes"] = [{"state": int(r["y_true"]), "y": int(r["y_true"]), "outcomes": {k: int(v) for k, v in vals.items()}, "source": "record"}]
            rows.append(r)
    with gzip.open(f"{IS}/envs_tab/data/{ds}_rz.jsonl.gz", "wt") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    print(ds, len(rows))
