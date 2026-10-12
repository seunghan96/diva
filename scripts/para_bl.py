"""Prompt-paraphrase robustness of the baselines (reviewer S6, 2026-10-11), next to DIVA's paraphrase robustness (tab:para).
Controlled environments, first 300 test scenarios per domain, first-decision DP regret (x100), averaged over the four
domains and the 20 LLMs. Original prompts: baselines/outputs (Direct, CoT, VoI-P, PAL, as in tab:know); paraphrases 1-2:
baselines/outputs_para{k} (BL_PARA=k). DIVA: the two question
paraphrases of tab:para (outputs/para/fac_p{k}_*). Spread: per-LLM max - min over the three prompts, averaged over LLMs.
A method is reported only when all 20 LLMs x 4 domains x 300 scenarios x 3 prompts are complete (else TBD).
Writes results/para_bl.json."""
import sys, os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalx import read, fac_scores, choose_max, score, summarize
from complete import KNOW_LLMS

DOMS = ["finance", "credit", "clinical", "legal"]
METHODS = ["direct", "cot_need", "voi_prompt", "pal", "diva"]


def src(m, k, tag, d):
    if m == "diva":
        return f"{IS}/outputs/test_bin/facnum_{tag}_{d}.jsonl.gz" if k == 0 else f"{IS}/outputs/para/fac_p{k}_{tag}_{d}.jsonl.gz"
    base = "outputs" if k == 0 else f"outputs_para{k}"
    return f"{IS}/baselines/{base}/{tag}/{m}/{d}_test.jsonl.gz"


def choices(m, p, env):
    out = {}
    if not os.path.exists(p):
        return out
    for r in read(p):
        i = r.get("id")
        if i not in env:
            continue
        if m == "diva":
            out[i] = choose_max(fac_scores(env[i], r)["net"])
        elif "error" not in r and r.get("choice") is not None:
            out[i] = r["choice"]
    return out


def main():
    envs = {d: {s["id"]: s for s in read(f"{IS}/envs_synth/data/binary/{d}_test.jsonl.gz")[:300]} for d in DOMS}
    res = {"per_llm": {}, "summary": {}}
    for m in METHODS:
        per = {}
        for tag in KNOW_LLMS:
            v = []
            for k in (0, 1, 2):
                regs = []
                for d in DOMS:
                    env = envs[d]; ch = choices(m, src(m, k, tag, d), env)
                    if len(ch) < len(env):
                        regs = None; break
                    regs.append(100 * summarize([score(env[i], ch[i]) for i in env])["dp_regret"])
                v.append(None if regs is None else float(np.mean(regs)))
            per[tag] = v
        res["per_llm"][m] = per
        # a prompt column is reported only when all 20 LLMs completed it; the spread uses the complete columns (>= 2)
        cols = [k for k in range(3) if all(per[t][k] is not None for t in KNOW_LLMS)]
        sm = {}
        for k, nm in enumerate(("orig", "p1", "p2")):
            sm[nm] = float(np.mean([per[t][k] for t in KNOW_LLMS])) if k in cols else None
        if len(cols) >= 2:
            arr = np.array([[per[t][k] for k in cols] for t in KNOW_LLMS])
            sm["spread"] = float((arr.max(1) - arr.min(1)).mean()); sm["spread_over"] = len(cols)
        else:
            sm["spread"] = None
        res["summary"][m] = sm
        print(m, res["summary"][m], "complete LLMs per prompt:", [sum(per[t][k] is not None for t in KNOW_LLMS) for k in range(3)], flush=True)
    json.dump(res, open(f"{IS}/results/para_bl.json", "w"), indent=1)


if __name__ == "__main__":
    main()
