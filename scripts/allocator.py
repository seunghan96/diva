"""Required-work GPU allocator (every 10 min; the watchdog's own fill is paused, its health checks/restarts continue).
Each pass:
  1. load per model = rows left over jobs/*.tsv (marker jobs weighted EPW rows; jobs_hold/ ignored);
     replica cap = max(1, ceil(load / CAP_ROWS)) for models with load, 0 without load;
  2. releases watchdog-managed ('serving') replicas of models with zero load and replicas beyond the cap, preferring
     replicas whose host would then offer a same-host GPU pair;
  3. TP2 models with load (by priority-weighted load per replica) get a same-host pair first: a host whose two GPUs are
     free or held only by releasable TP1 replicas (zero load or over cap); those replicas are released and the host is
     reserved for the TP2 launch so TP1 fills cannot take it;
  4. remaining free slots (free - 6 GB, free - 16 GB with a foreign process) go to TP1 models below their cap with the
     highest priority-weighted load per replica; alloc = min(pref, slot).
Logs to logs/allocator.log (and a summary line per action to ../STATUS.md). Stop: kill the allocator.py pid."""
import os, sys, glob, gzip, json, math, time, subprocess
IS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{IS}/serving")
import fleet  # noqa: E402

PY = "python"
EPW = 600  # rough rows-equivalent of one unfinished marker job (3000 over-provisioned q72, 10-11 08:50)
CAP_ROWS = 1500
PRIO = {"google/gemma-2-27b-it": 3.0, "Qwen/Qwen2.5-72B-Instruct-AWQ": 3.0, "casperhansen/llama-3.3-70b-instruct-awq": 2.0, "Qwen/Qwen3-14B": 3.0}
STATUS = f"{IS}/../STATUS.md"


def log(s, status=False):
    with open(f"{IS}/logs/allocator.log", "a") as f:
        f.write(time.strftime("%m-%d %H:%M:%S ") + s + "\n")
    if status:
        with open(STATUS, "a") as f:
            f.write(f"- {time.strftime('%m-%d %H:%M')} [allocator] {s}\n")


def nrows(p):
    p = p if p.startswith("/") else f"{IS}/{p}"
    try:
        with gzip.open(p, "rt") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def load():
    L = {}
    for f in glob.glob(f"{IS}/jobs/*.tsv"):
        for l in open(f):
            if not l.strip() or l.startswith("#"):
                continue
            jid, m, n, out, cmd = l.rstrip("\n").split("\t", 4)
            left = int(n) - nrows(out)
            if left > 0:
                L[m] = L.get(m, 0) + (EPW if ("_marks/" in out) else left)
    return L


def release(s, why):
    url = f"http://{s['host']}.onprem:{s['port']}"
    subprocess.run([PY, f"{IS}/serving/fleet.py", "release", "--url", url], capture_output=True, cwd=f"{IS}/serving")
    log(f"release {url} {s['model']} ({why})", status=True)


def servers():
    st = fleet.load_state()
    return st, [s for s in st["servers"].values() if s["status"] in ("up", "starting", "unhealthy")]


def gpus_of(s):
    return [int(g) for g in str(s.get("gpus", "")).split(",") if g.strip().isdigit()]


def one_pass():
    L = load()
    ros = {m["id"]: m for m in fleet.roster()}
    cap = {m: (max(1, math.ceil(L[m] / (CAP_ROWS * ros[m].get("tp", 1)))) if L.get(m, 0) > 0 else 0) for m in ros}  # a TP2 replica costs two GPUs
    st, S = servers()
    nrep = {}
    for s in S:
        nrep[s["model"]] = nrep.get(s["model"], 0) + 1
    scan = fleet.scan_all(st)
    ok_host = {h for h, r in scan.items() if r.get("ok") and not r["forbidden"] and not r["avoid"] and r["pit"] == 0}
    foreign = {(h, g): info["foreign"] for h, r in scan.items() if r.get("ok") for g, info in r.get("gpus", {}).items()}
    free = {(h, g): info["free"] for h, r in scan.items() if r.get("ok") for g, info in r.get("gpus", {}).items()}
    margin = lambda h, g: 16000 if foreign.get((h, g)) else 6000
    # releasable TP1 serving replicas: zero-load models, or replicas beyond the cap (most over-provisioned first)
    surplus = {m: nrep.get(m, 0) - cap.get(m, 0) for m in nrep}
    on_gpu = {}
    for s in S:
        if s.get("owner") == "serving" and ros.get(s["model"], {}).get("tp", 1) == 1:
            for g in gpus_of(s):
                on_gpu.setdefault((s["host"], g), []).append(s)
    est = lambda s: ros.get(s["model"], {}).get("pref_mib", 30000)
    released = []
    reserved = set()
    # 3. TP2 pairs first
    tp2 = sorted((PRIO.get(m, 1.0) * L[m] / (nrep.get(m, 0) + 1), m) for m in L if m in ros and ros[m]["tp"] == 2
                 and L[m] > 0 and nrep.get(m, 0) < cap[m])
    for score, m in sorted(tp2, reverse=True):
        need = ros[m]["min_mib"]
        best = None
        for h in ok_host:
            gs = [g for (hh, g) in free if hh == h]
            if len(gs) < 2 or h in reserved:
                continue
            gs = sorted(gs)[:2]
            rel, okp = [], True
            for g in gs:
                a = free[(h, g)] - margin(h, g)
                if a >= need:
                    continue
                cands = [s for s in on_gpu.get((h, g), []) if surplus.get(s["model"], 0) > 0 and s not in rel]
                gain = 0
                for s in sorted(cands, key=lambda s: -surplus[s["model"]]):
                    if a + gain >= need:
                        break
                    rel.append(s); gain += est(s)
                if a + gain < need:
                    okp = False
                    break
            if okp and (best is None or len(rel) < len(best[1])):
                best = (h, rel, gs)
        if best is None:
            log(f"no same-host pair for {m} (needs {need} MiB per GPU, load {L[m]}, replicas {nrep.get(m, 0)})")
            continue
        h, rel, gs = best
        for s in rel:
            release(s, f"frees a same-host pair on {h} for {m}")
            surplus[s["model"]] -= 1; nrep[s["model"]] -= 1
            released.append(s)
        reserved.add(h)
        if rel:
            time.sleep(150)
            st, _ = servers(); scan2 = fleet.scan_all(st)
            for g in gs:
                free[(h, g)] = scan2[h]["gpus"][g]["free"]
        a = min(free[(h, g)] - margin(h, g) for g in gs)
        if a < need:
            log(f"pair on {h} for {m} still short after release ({a} < {need}); retry next pass")
            continue
        mm = dict(ros[m]); mm["pref_mib"] = int(min(ros[m]["pref_mib"], a))
        log(f"launch {m} on {h} g{gs[0]},{gs[1]} alloc {mm['pref_mib']} (load {L[m]}, replicas {nrep.get(m, 0)}, cap {cap[m]})", status=True)
        try:
            fleet.do_launch(h, f"{gs[0]},{gs[1]}", mm, mm["pref_mib"])
        except Exception as e:
            log(f"launch error {e}")
        nrep[m] = nrep.get(m, 0) + 1
        for g in gs:
            free[(h, g)] = 0
    # 2. always release zero-load replicas; release over-cap replicas only when an under-cap model has no fitting slot
    avail = {k: v - margin(*k) for k, v in free.items() if k[0] in ok_host and k[0] not in reserved and v - margin(*k) > 0}
    unmet = 0
    for m in L:
        if m in ros and L[m] > 0 and ros[m]["tp"] == 1 and nrep.get(m, 0) < cap[m]:
            if not any(a >= ros[m]["min_mib"] for a in avail.values()):
                unmet += 1
    over = sorted((s for s in S if s not in released and s.get("owner") == "serving" and cap.get(s["model"], 0) > 0
                   and surplus.get(s["model"], 0) > 0), key=lambda s: -surplus[s["model"]])
    for s in S:
        if s in released or s.get("owner") != "serving":
            continue
        if cap.get(s["model"], 0) == 0:
            release(s, "no pending work"); nrep[s["model"]] -= 1; released.append(s)
    for s in over:
        if unmet <= 0:
            break
        if surplus.get(s["model"], 0) > 0:
            release(s, f"over cap {cap[s['model']]} (load {L.get(s['model'], 0)}); slot needed by an under-cap model")
            surplus[s["model"]] -= 1; nrep[s["model"]] -= 1; released.append(s); unmet -= ros.get(s["model"], {}).get("tp", 1)
    if released:
        time.sleep(150)
        st, _ = servers(); scan = fleet.scan_all(st)
        free = {(h, g): info["free"] for h, r in scan.items() if r.get("ok") for g, info in r.get("gpus", {}).items()}
    avail = {k: v - margin(*k) for k, v in free.items() if k[0] in ok_host and k[0] not in reserved and v - margin(*k) > 0}

    def launch1(m, h, g, a, tag):
        mm = dict(ros[m]); mm["pref_mib"] = int(min(ros[m]["pref_mib"], a))
        log(f"launch{tag} {m} on {h} g{g} alloc {mm['pref_mib']} (load {L[m]}, replicas {nrep.get(m, 0)}, cap {cap[m]})", status=True)
        try:
            fleet.do_launch(h, str(g), mm, mm["pref_mib"])
        except Exception as e:
            log(f"launch error {e}")
        nrep[m] = nrep.get(m, 0) + 1

    def fill_tp1(respect_cap):
        """best fit: slots from largest to smallest; each goes to the eligible model with the LARGEST memory need that fits
        (a >= 35 GB slot goes to a model that needs >= 35 GB first), ties broken by projected finish (load per replica)"""
        while True:
            elig = [m for m in L if m in ros and ros[m]["tp"] == 1 and L[m] > 0 and (not respect_cap or nrep.get(m, 0) < cap[m])]
            done = False
            for a, h, g in sorted(((a, h, g) for (h, g), a in avail.items()), reverse=True):
                fit = [m for m in elig if ros[m]["min_mib"] <= a]
                if not fit:
                    continue
                m = max(fit, key=lambda m: (ros[m]["min_mib"], PRIO.get(m, 1.0) * L[m] / (nrep.get(m, 0) + 1)))
                launch1(m, h, g, a, "" if respect_cap else " (overflow, idle slot)")
                avail.pop((h, g), None)
                done = True
                break
            if not done:
                return

    def fill_tp2_overflow():
        while True:
            cand = sorted(((PRIO.get(m, 1.0) * L[m] / (nrep.get(m, 0) + 1), m) for m in L
                           if m in ros and ros[m]["tp"] == 2 and L[m] > 0), reverse=True)
            done = False
            for score, m in cand:
                byh = {}
                for (h, g), a in avail.items():
                    if a >= ros[m]["min_mib"]:
                        byh.setdefault(h, []).append((g, a))
                pair = next(((h, sorted(gl)[:2]) for h, gl in byh.items() if len(gl) >= 2), None)
                if not pair:
                    continue
                h, gl = pair
                a = min(x[1] for x in gl)
                mm = dict(ros[m]); mm["pref_mib"] = int(min(ros[m]["pref_mib"], a))
                log(f"launch (overflow, idle pair) {m} on {h} g{gl[0][0]},{gl[1][0]} alloc {mm['pref_mib']} (load {L[m]}, replicas {nrep.get(m, 0)})", status=True)
                try:
                    fleet.do_launch(h, f"{gl[0][0]},{gl[1][0]}", mm, mm["pref_mib"])
                except Exception as e:
                    log(f"launch error {e}")
                nrep[m] = nrep.get(m, 0) + 1
                for g, _ in gl:
                    avail.pop((h, g), None)
                done = True
                break
            if not done:
                return

    # 3b. every TP1 model with pending work and NO replica gets one first (smallest fitting slot), so best fit cannot starve
    #     small models with little load (10-11 17:10: Qwen3-4B waited behind Qwen3-32B overflow)
    for m in sorted((m for m in L if m in ros and ros[m]["tp"] == 1 and L[m] > 0 and nrep.get(m, 0) == 0), key=lambda m: -ros[m]["min_mib"]):
        fits = sorted((a, h, g) for (h, g), a in avail.items() if a >= ros[m]["min_mib"])
        if not fits:
            victims = [s for s in S if s not in released and s.get("owner") == "serving" and ros.get(s["model"], {}).get("tp", 1) == 1
                       and nrep.get(s["model"], 0) > max(1, cap.get(s["model"], 0)) and est(s) >= ros[m]["min_mib"]]
            if not victims:
                log(f"no slot for zero-replica {m} (needs {ros[m]['min_mib']})")
                continue
            v = max(victims, key=lambda s: nrep[s["model"]] - cap.get(s["model"], 0))
            release(v, f"surplus replica, slot needed by zero-replica {m}")
            nrep[v["model"]] -= 1; released.append(v)
            time.sleep(150)
            st, _ = servers(); scan = fleet.scan_all(st)
            for g in gpus_of(v):
                info = scan[v["host"]]["gpus"][g]; a = info["free"] - (16000 if info["foreign"] else 6000)
                if a > 0:
                    avail[(v["host"], g)] = a
            fits = sorted((a, h, g) for (h, g), a in avail.items() if a >= ros[m]["min_mib"])
            if not fits:
                continue
        a, h, g = fits[0]
        launch1(m, h, g, a, " (first replica)")
        avail.pop((h, g), None)

    # 4. under-cap TP1 first, then overflow: idle pairs to TP2 models with load, then idle single slots to any model with load
    fill_tp1(True)
    fill_tp2_overflow()
    fill_tp1(False)


if __name__ == "__main__":
    log("start (v2: caps + TP2 pairs first)")
    once = "--once" in sys.argv
    while True:
        try:
            one_pass()
        except Exception as e:
            import traceback
            log(f"ERR {e} {traceback.format_exc()[-300:]}")
        if once:
            break
        time.sleep(600)
