"""Completion marker for prompting-episode jobs (queue_daemon counts rows of the job's output file).
Usage: epk_mark.py TAG ENV NSCEN REPS -> writes outputs/epk_marks/{TAG}_{ENV}_n{NSCEN}_r{REPS}.jsonl.gz with one row iff every
prompting method has an episode (no error) for every (scenario, rep < REPS) among the first NSCEN scenarios of ENV_epk."""
import sys, os, gzip, json
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read
M = ["direct", "cot_need", "bed_llm", "voi_prompt", "pal", "sc_cot", "react", "finrisk_ask", "voi_paper"]
tag, env, n, reps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
ep = f"{IS}/envs_tab/data/{env}_epk.jsonl.gz" if env in ("taiwan", "german", "cdc", "heart") else f"{IS}/envs_real/data/{env}_epk.jsonl.gz"
need = {(s["id"], r) for s in read(ep)[:n] for r in range(reps)}
miss = 0
for m in M:
    f = f"{IS}/baselines/outputs/{tag}/{m}/{env}_epk.episode.jsonl.gz"
    have = set()
    if os.path.exists(f):
        for r in read(f):
            if "error" not in r and "requests" in r:
                have.add((r["id"], r.get("rep", 0)))
    miss += len(need - have)
out = f"{IS}/outputs/epk_marks/{tag}_{env}_n{n}_r{reps}.jsonl.gz"
os.makedirs(os.path.dirname(out), exist_ok=True)
if miss == 0:
    with gzip.open(out, "wt") as h:
        h.write(json.dumps({"tag": tag, "env": env, "n": n, "reps": reps}) + "\n")
print(tag, env, "missing", miss)
