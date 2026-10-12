"""After generation: pilot subsets (first 150 rows of each dev file; rows are shuffled by group so prefixes are balanced),
test1000 subsets (first 1000 rows of each test file, the DESIGN.md size), and data/STATS.json with per-file counts."""
import gzip, json, os, glob
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))


def rd(p):
    return [json.loads(l) for l in gzip.open(p, "rt")]


def wr(p, rows):
    with gzip.open(p, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def prefix_groups(rows, n):
    """First n rows, extended so that a context/cost/payoff group is never cut in half."""
    out = rows[:n]
    if out and out[-1].get("group_id"):
        g = out[-1]["group_id"]
        k = n
        while k < len(rows) and rows[k].get("group_id") == g:
            out.append(rows[k]); k += 1
    return out


stats = {}
for track in ["binary", "ms"]:
    for p in sorted(glob.glob(f"{HERE}/data/{track}/*_dev.jsonl.gz")) + sorted(glob.glob(f"{HERE}/data/{track}/*_test.jsonl.gz")):
        rows = rd(p)
        name = os.path.basename(p).replace(".jsonl.gz", "")
        st = {"n": len(rows), "stratum": Counter(r["stratum"] for r in rows), "family": Counter(r["family"] for r in rows),
              "gold_stop": sum(r["exact"]["gold_choice"] == "S" for r in rows),
              "dp_stop": sum(r["exact"]["dp_first_action"] == "S" for r in rows),
              "myopic_dp_agree": sum((r["exact"]["gold_choice"] in r["exact"]["dp_first_ties"]) for r in rows),
              "has_voi0_eigpos_item": sum(any(r["exact"]["voi"][l] == 0 and r["exact"]["eig"][l] > 0.01 for l in r["label_order"]) for r in rows),
              "argmax_eig_not_argmax_voi": sum(max(r["exact"]["eig"], key=r["exact"]["eig"].get) != max(r["exact"]["voi"], key=r["exact"]["voi"].get) for r in rows),
              "n_items_mean": sum(len(r["label_order"]) for r in rows) / max(1, len(rows))}
        if track == "ms":
            st["qual_robust"] = sum(r["qual_robustness"]["robust"] for r in rows)
            st["context"] = Counter(r["context"] for r in rows)
        stats[f"{track}/{name}"] = st
        if name.endswith("_dev"):
            wr(p.replace("_dev.jsonl.gz", "_pilot.jsonl.gz"), prefix_groups(rows, 150))
        if name.endswith("_test"):
            wr(p.replace("_test.jsonl.gz", "_test1000.jsonl.gz"), prefix_groups(rows, 1000))
json.dump(stats, open(f"{HERE}/data/STATS.json", "w"), indent=1)
print(json.dumps(stats, indent=1))
