"""Completion marker for controlled-episode baseline jobs: writes outputs/ctl_ep_marks/{TAG}_{DOM}_{METHODS}.jsonl.gz iff every
listed method has an evaluated episode for each (scenario, world) of the first 100 scenarios x 2 worlds.
Usage: ctl_ep_mark.py TAG DOM m1,m2,..."""
import sys, os, gzip, json
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read
tag, dom, ms = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
need = {(s["id"], r) for s in read(f"{IS}/envs_synth/data/binary/{dom}_test.jsonl.gz")[:100] for r in range(2)}
miss = 0
for m in ms:
    f = f"{IS}/baselines/outputs/{tag}/{m}/{dom}_test.episode.jsonl.gz"
    have = {(r["id"], r.get("rep", 0)) for r in read(f) if r.get("eval")} if os.path.exists(f) else set()
    miss += len(need - have)
out = f"{IS}/outputs/ctl_ep_marks/{tag}_{dom}_{'-'.join(ms)}.jsonl.gz"
os.makedirs(os.path.dirname(out), exist_ok=True)
if miss == 0:
    with gzip.open(out, "wt") as h:
        h.write(json.dumps({"tag": tag, "dom": dom, "methods": ms}) + "\n")
print(tag, dom, "missing", miss)
