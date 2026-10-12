"""Coordinator auto-rebalancer (every 10 min): set fleet replica targets from the queue backlog.
- pending jobs per model = job lines in infoseek/jobs/*.tsv whose output has fewer rows than expected
- target = clamp(ceil(pending / 8), 1, 6) for 1-GPU models; 70B-class (TP2) models clamp to 1-2
- a model with 0 pending keeps 1 replica; extra watchdog-managed replicas are released (lead-launched servers untouched)
Logs to infoseek/logs/rebalance.log."""
import glob, gzip, json, math, os, subprocess, time
IS = "."
PY = "python"
FLEET = f"{IS}/serving/fleet.py"
BIG = {"casperhansen/llama-3.3-70b-instruct-awq", "Qwen/Qwen2.5-72B-Instruct-AWQ"}


def log(s):
    with open(f"{IS}/logs/rebalance.log", "a") as f:
        f.write(time.strftime("%m-%d %H:%M:%S ") + s + "\n")


def nrows(p):
    try:
        with gzip.open(p, "rt") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def fleet(*a):
    return subprocess.run([PY, FLEET, *a], capture_output=True, text=True, cwd=f"{IS}/serving").stdout


while True:
    try:
        pend = {}
        for f in glob.glob(f"{IS}/jobs/*.tsv"):
            for l in open(f):
                if not l.strip() or l.startswith("#"):
                    continue
                jid, m, n, out, cmd = l.rstrip("\n").split("\t", 4)
                pend.setdefault(m, 0)
                if nrows(f"{IS}/{out}" if not out.startswith("/") else out) < int(n):
                    pend[m] += 1
        roster = json.load(open(f"{IS}/serving/roster.json"))["models"]
        eps = json.load(open(f"{IS}/serving/endpoints.json"))
        state = json.load(open(f"{IS}/serving/state.json")) if os.path.exists(f"{IS}/serving/state.json") else {}
        servers = state.get("servers", {})
        for r in roster:
            m, cur = r["id"], r.get("target", 1)
            p = pend.get(m, 0)
            hi = 2 if m in BIG else 6
            want = max(1, min(hi, math.ceil(p / 4))) if p > 0 else 1
            if want != cur:
                fleet("target", "--model", m, "--n", str(want))
                log(f"target {m}: {cur} -> {want} (pending {p})")
            urls = eps.get(m, [])
            keep = want if p == 0 else 10**6  # release surplus replicas only when the model has no pending work (avoids thrash on GPUs only it can fit)
            if len(urls) > keep:
                for u in urls[keep:]:
                    hp = u.split("//")[1].replace(".onprem", "")
                    if hp in servers and servers[hp].get("owner") != "lead":  # only watchdog-managed servers
                        fleet("release", "--url", u)
                        log(f"release {u} ({m}, {len(urls)} replicas > target {want}, pending {p})")
    except Exception as e:
        log(f"ERR {e}")
    time.sleep(600)
