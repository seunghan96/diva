"""List gaps in the 20-LLM x 8-env knowledge grid (first 300 test scenarios): elicitation (fk/fr), direct, and baselines."""
import os, gzip, json, sys
IS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LL = ["q06b", "q17b", "q4b", "q8", "q14", "q32", "q25_7", "q25_14", "q72", "l8", "l70", "g9", "g27", "g12", "m7", "msm", "phi4", "oss", "r1_7", "r1_14"]
TAB = ["taiwan", "german", "cdc", "heart"]; SEC = ["ret_1", "ret_5", "vol_5", "vol_20"]
BL = ["cot_need", "bed_llm", "voi_prompt", "pal", "sc_cot", "react", "finrisk_ask", "voi_paper"]


def n(p, key="id", ok=None):
    if not os.path.exists(p):
        return 0
    ids = set()
    for l in gzip.open(p, "rt"):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if ok is None or ok(r):
            ids.add(r.get(key))
    return len(ids)


gaps = []
for t in LL:
    for e in TAB + SEC:
        tab = e in TAB
        spec = f"{IS}/outputs/test_tab/fk_{t}_{e}.jsonl.gz" if tab else f"{IS}/outputs/test_real/fr_{t}_{e}.jsonl.gz"
        dr = f"{IS}/outputs/test_tab/dir_{t}_{e}.jsonl.gz" if tab else f"{IS}/outputs/test_real/dir_{t}_{e}.jsonl.gz"
        if n(spec) < 300: gaps.append((t, e, "spec", n(spec)))
        if n(dr) < 300: gaps.append((t, e, "direct", n(dr)))
        for m in BL:
            c = n(f"{IS}/baselines/outputs/{t}/{m}/{e}_test.jsonl.gz", ok=lambda r: r.get("choice") is not None)
            if c < 285: gaps.append((t, e, m, c))
json.dump(gaps, open(f"{IS}/results/grid_gaps.json", "w"))
import collections
print(len(gaps)); print(collections.Counter(g[2] for g in gaps)); print(collections.Counter(g[0] for g in gaps))
