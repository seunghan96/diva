"""Coverage marker for row-limited elicitation jobs (queue_daemon counts rows of the job's output file).
Usage: cov_mark.py OUT ENV N MARKER [--k] -> writes MARKER (one row) iff OUT has a row for each of the first N scenarios
of ENV (with a non-empty 'k' field when --k is given)."""
import sys, os, gzip, json
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read
out, env, n, mark = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
needk = "--k" in sys.argv
ids = [s["id"] for s in read(f"{IS}/{env}" if not env.startswith("/") else env)[:n]]
have = set()
p = f"{IS}/{out}" if not out.startswith("/") else out
if os.path.exists(p):
    for r in read(p):
        if not needk or r.get("k"):
            have.add(r["id"])
miss = sum(1 for i in ids if i not in have)
if miss == 0:
    mp = f"{IS}/{mark}" if not mark.startswith("/") else mark
    os.makedirs(os.path.dirname(mp), exist_ok=True)
    with gzip.open(mp, "wt") as h:
        h.write(json.dumps({"out": out, "env": env, "n": n}) + "\n")
print(out, "missing", miss)
