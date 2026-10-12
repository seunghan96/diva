"""DEV-only test of an uncertainty-aware VOI for binned SEC checks: the elicited bin risks are treated as uncertain,
log-odds spread of the extreme bins perturbed by +/- sigma*sqrt(3) (3-point Gauss-Hermite, weights 1/6, 2/3, 1/6), and the
acquisition value is the expectation of VOI over this uncertainty. sigma grid on dev; ratio to never-ask per cell."""
import sys, os, glob, math, collections
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from evalx import read, score, summarize, choose_max, clip
from eval_real import fio_real
K = "dp_regret"
SIG = [1.0, 1.5, 2.0, 3.0]
GH = [(-math.sqrt(3), 1 / 6), (0.0, 2 / 3), (math.sqrt(3), 1 / 6)]


def shifted(fr, delta):
    out = {"prior": fr["prior"], "obs": fr["obs"], "chk": {}}
    for k, v in fr["chk"].items():
        vv = [x[0] if isinstance(x, list) else x for x in v]
        if any(x is None for x in vv) or len(vv) < 2:
            out["chk"][k] = v; continue
        n = len(vv)
        new = []
        for j, x in enumerate(vv):
            t = (j - (n - 1) / 2) / ((n - 1) / 2)  # -1 .. 1
            x = clip(x); lo = math.log(x / (1 - x)) + delta * t
            new.append([1 / (1 + math.exp(-lo))])
        out["chk"][k] = new
    return out


R = collections.defaultdict(list)
for f in glob.glob("outputs/real_dev/fr_*.jsonl.gz"):
    b = os.path.basename(f)[3:-9]
    tg = next((t for t in ["ret_5", "vol_5", "vol_20", "ret1"] if b.endswith(t)), None)
    if tg is None:
        continue
    tag = b[: -len(tg) - 1]; t2 = "ret_1" if tg == "ret1" else tg
    env = {s["id"]: s for s in read(f"envs_real/data/{t2}_dev.jsonl.gz")}; fr = {r["id"]: r for r in read(f)}
    ids = [i for i in env if i in fr]
    if len(ids) < 150:
        continue
    base = summarize([score(env[i], "S") for i in ids])[K]
    row = {}
    for sg in SIG:
        rows = []
        for i in ids:
            sc = env[i]
            nets = collections.defaultdict(float)
            for z, w in GH:
                fz = fio_real(sc, shifted(fr[i], sg * z), lam=0.75)
                for k2, v in fz["net"].items():
                    nets[k2] += w * v
            rows.append(score(sc, choose_max(dict(nets))))
        row[sg] = summarize(rows)[K] / base
    grp = "vol" if t2.startswith("vol") else "ret"
    print(tag, t2, len(ids), " ".join(f"{k}:{v:.2f}" for k, v in row.items()), flush=True)
    for k, v in row.items():
        R[(grp, k)].append(v)
for k, v in sorted(R.items()):
    print(k, round(float(np.mean(v)), 3), len(v))
