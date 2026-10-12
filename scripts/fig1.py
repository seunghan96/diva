"""Page-1 figure candidates (real data). Reads results/bin_main.json (+ optional knowledge results) and writes
paper/figs/fig1_cand{1..4}.pdf. Run with the pathcal venv (matplotlib)."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{IS}/paper/figs"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "STIXGeneral", "mathtext.fontset": "stix", "axes.spines.top": False, "axes.spines.right": False})
C = {"diva": "#1f77b4", "direct": "#7f7f7f", "fac_eig": "#e6781e", "always_stop": "#2ca02c", "bl": "#bbbbbb"}
NAMES = {"q32": "Qwen3-32B", "oss": "gpt-oss-20B", "msm": "Mistral-S-24B", "g12": "Gemma3-12B", "q14": "Qwen3-14B",
         "q8": "Qwen3-8B", "l8": "Llama3.1-8B", "g9": "Gemma2-9B", "q25_7": "Qwen2.5-7B", "r1_14": "R1-Qwen-14B",
         "m7": "Mistral-7B", "r1_7": "R1-Qwen-7B", "q25_14": "Qwen2.5-14B", "l70": "Llama3.3-70B", "q72": "Qwen2.5-72B",
         "phi4": "Phi-4", "g27": "Gemma2-27B"}

R = json.load(open(f"{IS}/results/bin_main.json"))
DOMS = ["finance", "credit", "clinical", "legal"]


def avg(tag, m, key="norm_regret"):
    v = [R[tag][d][m][key] for d in DOMS if m in R.get(tag, {}).get(d, {})]
    return float(np.mean(v)) if len(v) == len(DOMS) else None


tags = [t for t in R if avg(t, "diva") is not None and avg(t, "direct") is not None]
stop = np.mean([avg(t, "always_stop") for t in tags])

# Candidate 1: per-model dumbbell: direct vs DIVA, with always-stop line
fig, ax = plt.subplots(figsize=(2.3, 2.2))
ys = np.arange(len(tags))
d_dir = [avg(t, "direct") for t in tags]; d_div = [avg(t, "diva") for t in tags]
order = np.argsort(d_dir)
for k, i in enumerate(order):
    ax.plot([d_div[i], d_dir[i]], [k, k], color="#cccccc", lw=1, zorder=1)
ax.scatter([d_dir[i] for i in order], ys, color=C["direct"], s=14, label="LLM (Direct)", zorder=2)
ax.scatter([d_div[i] for i in order], ys, color=C["diva"], s=14, label="DIVA (Ours)", zorder=3)
ax.axvline(stop, color=C["always_stop"], ls="--", lw=1)
ax.text(stop, len(tags) - 0.3, " Never ask", color=C["always_stop"], fontsize=7, va="bottom")
ax.set_yticks(ys); ax.set_yticklabels([NAMES.get(tags[i], tags[i]) for i in order], fontsize=6)
ax.set_xlabel("Regret (lower is better)")
ax.legend(fontsize=6, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_cand1.pdf"); plt.close(fig)

# Candidate 2: grouped bars averaged over models: Direct / Fac-EIG / Never ask / DIVA (+ best baseline if present)
fig, ax = plt.subplots(figsize=(2.3, 1.9))
ms = [("direct", "Direct"), ("fac_eig", "Info. gain"), ("always_stop", "Never ask"), ("diva", "DIVA")]
vals = [np.mean([avg(t, m) for t in tags]) for m, _ in ms]
cols = [C["direct"], C["fac_eig"], C["always_stop"], C["diva"]]
ax.bar(range(len(ms)), vals, color=cols)
for i, v in enumerate(vals):
    ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=6)
ax.set_xticks(range(len(ms))); ax.set_xticklabels([n for _, n in ms], fontsize=7)
ax.set_ylabel("Regret (lower is better)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_cand2.pdf"); plt.close(fig)

# Candidate 3: behaviour decomposition: over-acquisition (S2) and information-trap rate (S1) for Direct / EIG / DIVA
fig, axs = plt.subplots(1, 2, figsize=(2.6, 1.7))
for ax, key, title in zip(axs, ["over_acq", "eig_trap"], ["Asks when\nshould stop", "Asks informative\nbut useless check"]):
    vals = [np.mean([avg(t, m, key) for t in tags if avg(t, m, key) is not None]) for m in ["direct", "fac_eig", "diva"]]
    ax.bar(range(3), vals, color=[C["direct"], C["fac_eig"], C["diva"]])
    ax.set_xticks(range(3)); ax.set_xticklabels(["Direct", "Info.\ngain", "DIVA"], fontsize=6)
    ax.set_title(title, fontsize=7); ax.set_ylim(0, 1)
axs[0].set_ylabel("Rate")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_cand3.pdf"); plt.close(fig)
print("tags", tags, "stop", stop)

# Candidates 4/5 from knowledge-required environments (results/know_main.json)
KM = json.load(open(f"{IS}/results/know_main.json")) if os.path.exists(f"{IS}/results/know_main.json") else {}
if KM:
    envs = [("tab:taiwan", "Credit\n(Taiwan)"), ("tab:german", "Credit\n(German)"), ("tab:cdc", "Diabetes\n(CDC)"), ("tab:heart", "Heart"),
            ("sec:ret_1", "SEC\nret-1d"), ("sec:ret_5", "SEC\nret-5d"), ("sec:vol_5", "SEC\nvol-5d"), ("sec:vol_20", "SEC\nvol-20d")]
    def mean(e, m):
        v = [r[f"full:{m}"]["dp_regret"] for r in KM.get(e, {}).values() if f"full:{m}" in r]
        return float(np.mean(v)) if v else np.nan
    fig, ax = plt.subplots(figsize=(2.4, 2.0))
    x = np.arange(len(envs)); w = 0.27
    for k, (m, name, col) in enumerate([("direct", "LLM (Direct)", C["direct"]), ("never", "Never ask", C["always_stop"]), ("diva_u", "DIVA-U (Ours)", C["diva"])]):
        ax.bar(x + (k - 1) * w, [100 * mean(e, m) for e, _ in envs], w, color=col, label=name)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in envs], fontsize=5.5); ax.set_ylabel("Regret ($\\times$100)")
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_cand4.pdf"); plt.close(fig)
    # candidate 5: regret relative to never asking (log scale), LLM direct vs DIVA, per environment
    fig, ax = plt.subplots(figsize=(1.6, 2.1))
    lab5 = ["Taiwan", "German", "CDC", "Heart", "Ret-1d", "Ret-5d", "Vol-5d", "Vol-20d"]
    rd = [mean(e, "direct") / mean(e, "never") for e, _ in envs]; rv = [mean(e, "diva_u") / mean(e, "never") for e, _ in envs]
    rp = [mean(e, "diva") / mean(e, "never") for e, _ in envs]
    ax.scatter(rd, range(len(envs)), color=C["direct"], s=14, label="LLM (Direct)")
    ax.scatter(rp, range(len(envs)), facecolors="white", edgecolors=C["diva"], s=14, label="DIVA (Ours)", zorder=3)
    ax.scatter(rv, range(len(envs)), color=C["diva"], s=14, label="DIVA-U (Ours)", zorder=3)
    for i in range(len(envs)):
        ax.plot([min(rv[i], rp[i], rd[i]), max(rv[i], rp[i], rd[i])], [i, i], color="#cccccc", lw=1, zorder=0)
    ax.axvline(1.0, color=C["always_stop"], ls="--", lw=1); ax.text(1.0, len(envs) - 0.45, " Never Ask", color=C["always_stop"], fontsize=6.5)
    ax.axhline(3.5, color="#dddddd", lw=0.6)
    ax.set_xscale("log"); ax.set_yticks(range(len(envs))); ax.set_yticklabels(lab5, fontsize=6.5)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_xlabel("Regret / Never Ask", fontsize=7); ax.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.35, -0.25), ncol=1, handletextpad=0.2)
    fig.tight_layout(pad=0.2); fig.savefig(f"{OUT}/fig1_cand5.pdf"); plt.close(fig)

# candidate 6 (R6-E9): episode regret relative to never asking (reference posterior), from results/episodes_know.json
EK = json.load(open(f"{IS}/results/episodes_know_k4.json")) if os.path.exists(f"{IS}/results/episodes_know_k4.json") else {}
if EK:
    order = ["taiwan", "german", "cdc", "heart", "ret_1", "ret_5", "vol_5", "vol_20"]
    lab6 = ["Taiwan", "German", "CDC", "Heart", "Ret-1d", "Ret-5d", "Vol-5d", "Vol-20d"]
    def em(e, m):
        cells = [c for c in EK.get(e, {}).values() if all(k in c for k in ("never", "diva", "diva_dp", "direct1"))]
        return float(np.mean([c[m]["regret_ref"] for c in cells])) / float(np.mean([c["never"]["regret_ref"] for c in cells]))
    fig, ax = plt.subplots(figsize=(3.05, 2.05))
    rd = [em(e, "direct1") for e in order]; rp = [em(e, "diva") for e in order]; rv = [em(e, "diva_dp") for e in order]
    ax.scatter(rd, range(8), color=C["direct"], s=22, label="LLM Direct First")
    ax.scatter(rp, range(8), facecolors="white", edgecolors=C["diva"], s=22, label="DIVA (Ours)", zorder=3)
    ax.scatter(rv, range(8), color=C["diva"], s=22, label="DIVA-DP (Ours)", zorder=3)
    for i in range(8):
        ax.plot([min(rd[i], rp[i], rv[i]), max(rd[i], rp[i], rv[i])], [i, i], color="#cccccc", lw=1, zorder=0)
    ax.axvline(1.0, color=C["always_stop"], ls="--", lw=1); ax.text(1.0, 7.45, " Never Ask", color=C["always_stop"], fontsize=7.5)
    ax.axhline(3.5, color="#dddddd", lw=0.6)
    ax.set_xscale("log"); ax.set_yticks(range(8)); ax.set_yticklabels(lab6, fontsize=7.5); ax.tick_params(axis="x", labelsize=7.5)
    ax.set_xlabel("Episode Regret / Never Ask (Log Scale)", fontsize=7.5)
    ax.legend(fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1, handletextpad=0.2)
    fig.tight_layout(pad=0.3); fig.savefig(f"{OUT}/fig1_cand6.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)
