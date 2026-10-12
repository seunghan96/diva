"""Fresh knowledge-required tabular scenarios for a clean evaluation of DIVA-U (not used in its design):
new scenario draws (checks, costs, payoffs, known findings) from the TEST records with an independent seed.
Taiwan/CDC: 1,000 scenarios from test records not sampled for the original test scenarios where possible;
German/Heart: the test split has only ~200 records, so records are reused with fresh scenario draws (disclosed).
Writes data/{ds}_fresh.jsonl.gz (ids '<ds>-test-fresh-<k>')."""
import sys, json, zlib, gzip
import numpy as np, pandas as pd
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import gen

N = {"taiwan": 1000, "cdc": 1000, "german": 1000, "heart": 1000}
for ds in sys.argv[1:] or list(N):
    meta = json.load(open(f"{gen.B}/clean/{ds}_meta.json"))
    df = pd.read_csv(f"{gen.B}/clean/{ds}.csv.gz")
    feats = list(meta["findings"].keys()); placebos = set(meta["placebos"]); D = gen.DATASETS[ds]
    ref = gen.Ref(ds, feats)
    rng = np.random.default_rng(zlib.crc32(f"{ds}-fresh-2026-10-10".encode()))
    recs = df[df.split == "test"]
    used = set()
    for l in gzip.open(f"{gen.B}/data/{ds}_test.jsonl.gz", "rt"):
        used.add(json.loads(l)["record_id"])
    pool = [r for r in recs.rid.values if r not in used]
    n = N[ds]
    if len(pool) >= n:
        rids = list(rng.choice(pool, n, replace=False))
    else:
        allr = recs.rid.values
        rids = list(np.concatenate([rng.permutation(allr) for _ in range(int(np.ceil(n / len(allr))))])[:n])
    seeds = [int(s) for s in rng.integers(0, 2 ** 31 - 1, size=len(rids))]
    rows = gen.make_rows(ds, "test", "fresh", recs, rids, seeds, ref, meta, feats, placebos, D)
    gen.write(f"{gen.B}/data/{ds}_fresh.jsonl.gz", rows)
    print(ds, len(rows), "unused-record pool", len(pool), flush=True)
