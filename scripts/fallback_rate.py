"""Fallback rate: share of elicited numeric answers (greedy) that failed to parse (stored as null). Writes results/fallback.json."""
import glob, gzip, json, os, collections
IS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = True
def walk(x, c):
    if x is None: c[0] += 1; c[1] += 1
    elif isinstance(x, (int, float)): c[1] += 1
    elif isinstance(x, list):
        if SAMPLES and x and all(not isinstance(v, (list, dict)) for v in x): walk(x[0], c)   # greedy answer is element 0 of a sample list
        else:
            for v in x: walk(v, c)
    elif isinstance(x, dict):
        for k, v in x.items():
            if k not in ("id", "form", "k"): walk(v, c)
out = {}
for tr, pat in [("controlled", "outputs/test_bin/facnum_*.jsonl.gz"), ("records", "outputs/test_tab/fk_*.jsonl.gz"), ("sec", "outputs/test_real/fr_*.jsonl.gz")]:
    c = [0, 0]; SAMPLES = tr != "controlled"
    for f in glob.glob(f"{IS}/{pat}"):
        for line in gzip.open(f, "rt"):
            r = json.loads(line); walk({k: r[k] for k in ("prior", "obs", "chk") if k in r}, c)
    out[tr] = {"fail": c[0], "n": c[1], "rate": c[0] / max(c[1], 1)}
    print(tr, out[tr], flush=True)
json.dump(out, open(f"{IS}/results/fallback.json", "w"), indent=1)
