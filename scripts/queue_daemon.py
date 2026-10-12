"""Standing job queue (lead). Every 120 s: reads infoseek/jobs/*.tsv lines
    JOBID <TAB> MODEL <TAB> N_EXPECTED <TAB> OUTFILE <TAB> command...
launches a job when its model has >= 1 live endpoint, it is not running, its output has < N_EXPECTED rows, and the
model has fewer than (2 x live replicas) running jobs. Jobs are resumable, so relaunch is safe. Logs to logs/queue.log."""
import os, sys, time, glob, gzip, subprocess, json, urllib.request
sys.path.insert(0, "<path>")
import dcfr_lib

IS = "."
RUN = {}  # jobid -> Popen


def live(urls):
    out = []
    for u in urls:
        try:
            urllib.request.urlopen(u + "/v1/models", timeout=5); out.append(u)
        except Exception:
            pass
    return out


def nrows(p):
    if not os.path.exists(p):
        return 0
    try:
        with gzip.open(p, "rt") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def _epkey(cmd):
    """episode jobs on the same (tag, env) write the same files even with different --limit/--reps: treat as one job"""
    import re
    if "baselines.run" not in cmd or "--mode episode" not in cmd:
        return None
    a, b = re.search(r"--tag (\S+)", cmd), re.search(r"--env (\S+)", cmd)
    return (f"--tag {a.group(1)} ", b.group(1)) if a and b else None


def log(s):
    with open(f"{IS}/logs/queue.log", "a") as f:
        f.write(time.strftime("%m-%d %H:%M:%S ") + s + "\n")


while True:
    try:
        eps = dcfr_lib.endpoints()
        liv = {m: live(u) for m, u in eps.items()}
        for j, p in list(RUN.items()):
            if p.poll() is not None:
                del RUN[j]
        running_by_model = {}
        jobs = []
        for f in sorted(glob.glob(f"{IS}/jobs/*.tsv")):
            for line in open(f):
                if not line.strip() or line.startswith("#"):
                    continue
                jid, model, nexp, outf, cmd = line.rstrip("\n").split("\t", 4)
                jobs.append((jid, model, int(nexp), outf, cmd))
        ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
        external = set()
        for jid, model, nexp, outf, cmd in jobs:
            tail = cmd.split(" ", 1)[1] if " " in cmd else cmd
            ek = _epkey(cmd)
            if jid in RUN or (" " + outf) in ps or tail in ps or (ek and any(ek[0] in l and ek[1] in l and "baselines.run" in l for l in ps.splitlines())):
                running_by_model[model] = running_by_model.get(model, 0) + 1
                if jid not in RUN:
                    external.add(jid)
        for jid, model, nexp, outf, cmd in jobs:
            if jid in RUN or jid in external or not liv.get(model):
                continue
            if nrows(outf) >= nexp:
                continue
            if running_by_model.get(model, 0) >= (12 if any(k in model.lower() for k in ("70b", "72b", "r1-distill-qwen-14b", "qwen3-14b", "qwen3-32b")) else 6) * len(liv[model]):
                continue
            lf = open(f"{IS}/logs/job_{jid}.log", "a")
            RUN[jid] = subprocess.Popen(cmd, shell=True, cwd=IS, stdout=lf, stderr=subprocess.STDOUT)
            running_by_model[model] = running_by_model.get(model, 0) + 1
            log(f"launch {jid} ({model}, have {nrows(outf)}/{nexp})")
    except Exception as e:
        log(f"ERR {e}")
    time.sleep(30)
