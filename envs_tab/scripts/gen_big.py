"""Additional knowledge-required tabular scenarios for a higher-powered realized-utility test (reviewer S3, 2026-10-11):
new scenario draws from TEST records used by neither the original test scenarios nor the fresh scenarios, with an
independent seed. Taiwan / CDC only (German / Heart have ~200 test records). Writes data/{ds}_big.jsonl.gz
(ids '<ds>-test-big-<k>')."""
import sys, json, zlib, gzip
import numpy as np, pandas as pd
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import gen

N = {"taiwan": 2000, "cdc": 2000}
for ds in sys.argv[1:] or list(N):
    meta = json.load(open(f"{gen.B}/clean/{ds}_meta.json"))
    df = pd.read_csv(f"{gen.B}/clean/{ds}.csv.gz")
    feats = list(meta["findings"].keys()); placebos = set(meta["placebos"]); D = gen.DATASETS[ds]
    ref = gen.Ref(ds, feats)
    rng = np.random.default_rng(zlib.crc32(f"{ds}-big-2026-10-11".encode()))
    recs = df[df.split == "test"]
    used = set()
    for part in ("test", "fresh"):
        for l in gzip.open(f"{gen.B}/data/{ds}_{part}.jsonl.gz", "rt"):
            used.add(json.loads(l)["record_id"])
    pool = [r for r in recs.rid.values if r not in used]
    n = min(N[ds], len(pool))
    rids = list(rng.choice(pool, n, replace=False))
    seeds = [int(s) for s in rng.integers(0, 2 ** 31 - 1, size=len(rids))]
    rows = gen.make_rows(ds, "test", "big", recs, rids, seeds, ref, meta, feats, placebos, D)
    gen.write(f"{gen.B}/data/{ds}_big.jsonl.gz", rows)
    print(ds, len(rows), "unused-record pool", len(pool), flush=True)
