"""Cost per first acquisition decision: LLM calls and generated tokens (chars/4) per scenario, from baselines' call logs
(controlled finance test, first 300 rows) and DIVA's design (1 + 2 x #checks+known short numeric answers of <= 6 tokens).
Writes paper/tables/cost.tex."""
import sys, os, json, gzip, glob, collections
import numpy as np
IS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{IS}/scripts")
from evalx import read

tags = sys.argv[1].split(",") if len(sys.argv) > 1 else ["q8", "q25_7", "m7"]
meths = [("direct", "Direct"), ("cot_need", "CoT"), ("sc_cot", "SC"), ("react", "ReAct"), ("voi_prompt", "VoI-P"),
         ("bed_llm", "BED-LLM"), ("pal", "PAL")]
rows = []
for m, name in meths:
    calls, toks = [], []
    for t in tags:
        for p in glob.glob(f"{IS}/baselines/outputs/{t}/{m}/*_test.calls.jsonl.gz"):
            for r in read(p)[:300]:
                cs = r.get("calls", [])
                calls.append(len(cs))
                toks.append(sum(sum(len(x) for x in (c.get("raw") or [])) // 4 for c in cs if c.get("kind") == "gen"))
    if calls:
        rows.append((name, np.mean(calls), np.mean(toks)))
env = read(f"{IS}/envs_synth/data/binary/finance_test.jsonl.gz")[:300]
dcalls = np.mean([1 + 2 * (len(s["checks"]) + len(s.get("observed", []))) for s in env])
rows.append(("\\diva{}", dcalls, dcalls * 3))
lines = [f"{n} & {c:.1f} & {t:.0f} \\\\" for n, c, t in rows]
tex = ("\\begin{table}[t]\n\\centering\n\\small\n\\begin{tabular}{lcc}\n\\toprule\nMethod & LLM calls & Generated tokens \\\\\n\\midrule\n"
       + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\caption{\\textbf{Cost of the first acquisition decision} per scenario on the controlled finance environment. \\diva{}'s specification is reused for all later steps of an episode, whereas the baselines query the LLM again after every observation.}\n\\label{tab:cost}\n\\end{table}\n")
open(f"{IS}/paper/tables/cost.tex", "w").write(tex)
print(rows)
