"""Lead's minimal scenario generator (smoke tests / pilot fallback). The full generator lives in envs_synth/."""
import random, json, gzip, sys
from domains import DOMAINS
from dcore import posterior, voi_exact, eig_exact, dp_value, vstar, best_action

LET = "ABCDEF"


def rel_pair(rng, kind):
    if kind == "strong":
        return rng.choice([(0.85, 0.15), (0.8, 0.1), (0.9, 0.25), (0.75, 0.1), (0.9, 0.2)])
    if kind == "onesided_hi":  # finding common in both, absence very informative of benign
        return rng.choice([(0.95, 0.6), (0.9, 0.5), (0.97, 0.7)])
    if kind == "onesided_lo":  # finding rare in both, presence strongly indicates adverse
        return rng.choice([(0.4, 0.03), (0.3, 0.02), (0.5, 0.08)])
    if kind == "weak":
        return rng.choice([(0.6, 0.45), (0.55, 0.4), (0.5, 0.38)])
    if kind == "placebo":
        x = rng.choice([0.3, 0.4, 0.5, 0.6]); return (x, x)
    return (0.7, 0.3)


def make(rng, dom, idx, stratum_want):
    D = DOMAINS[dom]
    for _ in range(5000):
        prior = rng.choice([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6])
        g = rng.choice([1, 2, 3]); l = rng.choice([2, 3, 5, 8])
        lib = rng.sample(D["checks"], 8)
        kinds = ["strong", "onesided_hi", "onesided_lo", "weak", "placebo", rng.choice(["strong", "onesided_hi", "onesided_lo", "mid"])]
        rng.shuffle(kinds)
        cs = rng.choice([0.0, 0.05, 0.1, 0.2, 0.4]) * min(g, l)
        checks = []
        for k, (kind, (name, ft)) in enumerate(zip(kinds, lib[:6])):
            s1, s0 = rel_pair(rng, kind)
            if rng.random() < 0.5 and kind not in ("placebo",):
                s1, s0 = (s1, s0)
            cost = round(cs * rng.choice([0.5, 1, 1.5]), 3)
            checks.append(dict(label=LET[k], name=name, finding_text=ft, s1=s1, s0=s0, cost=cost, kind=kind, is_placebo=kind == "placebo"))
        observed = []
        for (name, ft) in lib[6:6 + rng.choice([0, 1, 2])]:
            s1, s0 = rel_pair(rng, rng.choice(["strong", "mid", "weak"]))
            observed.append(dict(name=name, finding_text=ft, value=rng.random() < 0.5, s1=s1, s0=s0))
        b = posterior(prior, [(o["s1"], o["s0"], o["value"]) for o in observed])
        voi = {c["label"]: voi_exact(b, c["s1"], c["s0"], g, l) for c in checks}
        eig = {c["label"]: eig_exact(b, c["s1"], c["s0"]) for c in checks}
        net = {k: voi[k] - c["cost"] for k, c in zip(voi, checks)}
        mv = max(voi.values()); ae = max(eig, key=eig.get); av = max(net, key=net.get)
        if max(net.values()) <= 1e-9:
            stratum = "S2"
        elif ae != av and voi[ae] <= 0.5 * mv:
            stratum = "S1"
        elif ae == av:
            stratum = "S3"
        else:
            stratum = "other"
        if stratum != stratum_want:
            continue
        if stratum == "S2" and max(eig.values()) < 0.05:
            continue
        dpv, dpa = dp_value(b, [(c["s1"], c["s0"], c["cost"]) for c in checks], g, l)
        gold = "S" if max(net.values()) <= 1e-9 else av
        return dict(id=f"{dom}-{idx}", domain=dom, stratum=stratum, prior=prior, g=g, l=l, checks=checks,
                    observed=observed, style=rng.randrange(3),
                    exact=dict(posterior=b, best_action=best_action(b, g, l), voi=voi, eig=eig, net=net,
                               gold_choice=gold, dp_value=dpv, dp_first=("S" if dpa == "S" else checks[dpa]["label"]),
                               vstar=vstar(b, g, l)))
    return None


if __name__ == "__main__":
    n = int(sys.argv[1]); seed = int(sys.argv[2]); out = sys.argv[3]
    doms = sys.argv[4].split(",") if len(sys.argv) > 4 else list(DOMAINS)
    rng = random.Random(seed)
    rows = []
    for dom in doms:
        for i in range(n):
            st = ["S1", "S1", "S2", "S3", "S3"][i % 5] if i % 5 != 1 else "S1"
            st = ["S1", "S2", "S3", "S1", "S3"][i % 5]
            r = make(rng, dom, i, st)
            if r:
                rows.append(r)
    with gzip.open(out, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(len(rows), Counter((r["domain"], r["stratum"]) for r in rows))
