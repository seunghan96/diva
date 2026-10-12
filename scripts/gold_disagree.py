"""Property of the environments (no LLM): how often the myopic and the DP gold disagree at the first decision.
Writes results/gold_disagree.json."""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read
out = {}
for kind, envs in [("tab", ["taiwan", "german", "cdc", "heart"]), ("sec", ["ret_1", "ret_5", "vol_5", "vol_20"])]:
    for e in envs:
        p = f"{IS}/envs_tab/data/{e}_test.jsonl.gz" if kind == "tab" else f"{IS}/envs_real/data/{e}_test.jsonl.gz"
        rows = read(p); n = len(rows)
        ms = [r["exact"]["gold_choice"] == "S" for r in rows]
        dps = [("S" in (r["exact"].get("dp_first_ties") or [r["exact"].get("dp_first_action")])) for r in rows]
        both_acq = [(not a) and (not b) for a, b in zip(ms, dps)]
        same = sum(1 for r, a, b in zip(rows, ms, dps) if (not a) and (not b) and r["exact"]["gold_choice"] in (r["exact"].get("dp_first_ties") or []))
        out[e] = {"n": n, "myopic_stop": sum(ms) / n, "dp_stop": sum(dps) / n,
                  "myo_stop_dp_acq": sum(1 for a, b in zip(ms, dps) if a and not b) / n,
                  "dp_ties_mean": sum(len(r["exact"].get("dp_first_ties") or [1]) for r in rows) / n,
                  "same_check_when_both_acquire": same / max(sum(both_acq), 1)}
        print(e, {k: round(v, 3) for k, v in out[e].items()})
json.dump(out, open(f"{IS}/results/gold_disagree.json", "w"), indent=1)
