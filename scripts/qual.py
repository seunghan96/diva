"""Find qualitative examples on the controlled finance test track: direct LLM (and Fac-EIG) choose an informative check with
zero decision value while DIVA chooses the DP-optimal check (or stops correctly). Writes paper/tables/qual.tex."""
import sys, os, json, gzip
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read, fac_scores, choose_max

tag = sys.argv[1] if len(sys.argv) > 1 else "q8"
env = {s["id"]: s for s in read(f"{IS}/envs_synth/data/binary/finance_test.jsonl.gz")}
fac = {r["id"]: r for r in read(f"{IS}/outputs/test_bin/facnum_{tag}_finance.jsonl.gz")}
dr = {r["id"]: r for r in read(f"{IS}/outputs/test_dir/dir_{tag}_finance.jsonl.gz")}
ex = []
for i, sc in env.items():
    if i not in fac or i not in dr or not dr[i].get("dist") or len(sc["checks"]) > 6:
        continue
    E = sc["exact"]; d = max(dr[i]["dist"], key=dr[i]["dist"].get)
    dv = choose_max(fac_scores(sc, fac[i])["net"])
    gold = E["dp_first_ties"] if E.get("dp_first_ties") else [E["dp_first_action"]]
    if d != "S" and E["voi"].get(d, 1) < 1e-9 and E["eig"].get(d, 0) > 0.1 and dv in gold:
        ex.append((E["eig"][d], i, d, dv))
ex.sort(reverse=True)
out = []
for _, i, d, dv in ex[:2]:
    sc = env[i]; E = sc["exact"]
    name = {c["label"]: c["name"] for c in sc["checks"]}; name["S"] = "Stop and decide"
    out.append({"id": i, "text": sc["text"], "direct": d, "direct_name": name[d], "diva": dv, "diva_name": name[dv],
                "voi": E["voi"], "eig": E["eig"], "net": E["net"], "posterior": E["posterior"], "g": sc["g"], "l": sc["l"]})
json.dump(out, open(f"{IS}/results/qual_{tag}.json", "w"), indent=1)
print(len(ex), [(o["id"], o["direct"], o["diva"]) for o in out])
