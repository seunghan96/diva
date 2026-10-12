"""Six single-panel candidates for Figure fig:trap (controlled + knowledge-required in one plot; user 10-11, to save space).
Same data and method rows as figs_all.fig_trap. Writes paper/figs/cands/fig_trap_cand{1..6}.pdf and one review PDF
paper/figs/cands/fig_trap_candidates.pdf (one candidate per page, with its paper-scale height). Run with the pathcal venv."""
import os, sys, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figs_all as F  # noqa: E402
import complete as CP  # noqa: E402
from crop_figs import crop_pdf  # noqa: E402

OUTD = f"{F.OUT}/cands"; os.makedirs(OUTD, exist_ok=True)
TYPES = [("trap", "Decision-Irrelevant (VOI=0)", "#e6781e"), ("placebo", "Uninformative", "#bbbbbb"),
         ("overpriced", "Overpriced (VOI<Cost)", "#9467bd"), ("suboptimal", "Suboptimal Check", "#8c564b"),
         ("premature", "Premature Stop", "#2ca02c")]
OURS = ("DIVA", "DIVA-U")
HL = "#fdf5d3"


def data():
    T, X = F.J("trap"), F.J("trap_extra")
    meths = [("direct", "Direct", "Direct", "Direct"), ("cot_need", "CoT", "cot_need", "cot_need"), ("sc_cot", "SC", "sc_cot", "sc_cot"),
             ("react", "ReAct", "react", "react"), ("finrisk_ask", "Ask-P", "finrisk_ask", "finrisk_ask"),
             ("voi_prompt", "VoI-P", "voi_prompt", "voi_prompt"), ("voi_paper", "VoI-Ask", "voi_paper", "voi_paper"),
             ("bed_llm", "BED-LLM", "bed_llm", "bed_llm"), ("pal", "PAL", "pal", "pal"),
             ("never", "Never", "never", "never"), ("always_top", "Always ask", "always_top", "always_top"),
             ("diva", "DIVA", "DIVA", "DIVA"), ("diva_u", "DIVA-U", "DIVA", "DIVA-U")]
    meths = sorted(meths, key=lambda r: CP.mrank(r[0]))
    envs = ["taiwan", "german", "cdc", "heart", "ret_1", "ret_5", "vol_5", "vol_20"]
    out = {}
    for tr, tags, ev in [("controlled", list(CP.CTRL_BL_LLMS), None), ("knowledge", list(CP.KNOW_LLMS), envs)]:
        cells = {k: dict(v) for k, v in T.get(tr, {}).items()}
        for k, v in X.get(tr, {}).items():
            cells.setdefault(k, {}).update(v)
        M = []
        for key, name, mc, mk in meths:
            mm = mc if tr == "controlled" else mk
            keys_of = (lambda t: [t]) if ev is None else (lambda t: [f"{t}:{e}" for e in ev])
            agg = collections.defaultdict(float)
            for t in tags:
                for k in keys_of(t):
                    for kk, v in cells[k][mm].items():
                        agg[kk] += v
            M.append([100 * agg["reg_" + t] / agg["n"] for t, _, _ in TYPES])
        out[tr] = np.array(M)
    return [n for _, n, _, _ in meths], out["controlled"], out["knowledge"]


NAMES, C, K = data()
N = len(NAMES); FS = 7


def style_ticks(ax, pos, names):
    ax.set_yticks(pos); ax.set_yticklabels(names, fontsize=FS)
    for lab in ax.get_yticklabels():
        if lab.get_text() in OURS:
            lab.set_fontweight("bold")


def legend(fig, extra=(), ncol=2, y=0.005):
    h = [Patch(color=c, label=l) for _, l, c in TYPES] + list(extra)
    fig.legend(handles=h, loc="lower center", ncol=ncol, fontsize=7.8, frameon=True, fancybox=False, edgecolor="#555555",
               framealpha=1, borderpad=0.4, handlelength=1.4, columnspacing=1.0, bbox_to_anchor=(0.5, y))


def stack(ax, pos, M, h, **kw):
    left = np.zeros(len(pos))
    for j, (_, _, col) in enumerate(TYPES):
        ax.barh(pos, M[:, j], left=left, color=col, height=h, zorder=2, **kw); left += M[:, j]
    return left


def band(ax, lo, hi):
    ax.axhspan(lo, hi, color=HL, zorder=0, lw=0)


def c1():
    """paired thin bars per method (controlled above, knowledge below), group tag at the bar end"""
    fig, ax = plt.subplots(figsize=(3.1, 3.55))
    y = np.arange(N) * 1.0
    band(ax, y[-2] - 0.5, y[-1] + 0.5)
    lc = stack(ax, y - 0.19, C, 0.36); lk = stack(ax, y + 0.19, K, 0.36, hatch="////", edgecolor="white", linewidth=0)
    for i in range(N):
        ax.text(lc[i] + 0.06, y[i] - 0.19, "C", fontsize=5.5, va="center", color="#444")
        ax.text(lk[i] + 0.06, y[i] + 0.19, "K", fontsize=5.5, va="center", color="#444")
    style_ticks(ax, y, NAMES); ax.invert_yaxis(); ax.set_xlabel("Regret ($\\times$100) by Error Type", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=FS)
    legend(fig, [Patch(facecolor="white", edgecolor="#666", label="C: Controlled"), Patch(facecolor="#ddd", hatch="////", edgecolor="#666", label="K: Knowledge-Req.")])
    fig.tight_layout(rect=(0, 0.17, 1, 1)); return fig


def c2():
    """butterfly: controlled to the left, knowledge-required to the right, shared method labels in the middle"""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.1, 2.95), sharey=True, gridspec_kw={"wspace": 0.0})
    y = np.arange(N)
    for a in (a1, a2):
        band(a, y[-2] - 0.5, y[-1] + 0.5)
    stack(a1, y, C, 0.7); stack(a2, y, K, 0.7)
    a1.invert_xaxis(); a1.invert_yaxis()
    a1.set_yticks(y); a1.set_yticklabels([]); a2.tick_params(axis="y", length=0)
    for i, n in enumerate(NAMES):
        a2.text(-0.02, i, n, transform=a2.get_yaxis_transform(), ha="center", va="center", fontsize=6.3,
                fontweight="bold" if n in OURS else "normal", bbox=dict(facecolor="white", edgecolor="none", pad=0.4, alpha=0.85), zorder=6)
    a1.set_title("Controlled", fontsize=7.5, fontweight="bold"); a2.set_title("Knowledge-Required", fontsize=7.5, fontweight="bold")
    for a in (a1, a2):
        a.tick_params(axis="x", labelsize=FS)
    fig.text(0.5, 0.235, "Regret ($\\times$100) by Error Type", ha="center", fontsize=7.5)
    legend(fig); fig.subplots_adjust(left=0.03, right=0.97, top=0.91, bottom=0.34); return fig


def c3():
    """one bar per method = the two groups side by side on one row (controlled | knowledge-required), separated by a gap axis"""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.1, 2.95), sharey=True, gridspec_kw={"wspace": 0.08, "width_ratios": [6.2, 4.6]})
    y = np.arange(N)
    for a, M, t in ((a1, C, "(a) Controlled"), (a2, K, "(b) Knowledge-Req.")):
        band(a, y[-2] - 0.5, y[-1] + 0.5); stack(a, y, M, 0.7); a.set_title(t, fontsize=7.5, fontweight="bold")
        a.tick_params(axis="x", labelsize=FS)
    style_ticks(a1, y, NAMES); a1.invert_yaxis()
    fig.text(0.6, 0.235, "Regret ($\\times$100) by Error Type", ha="center", fontsize=7.5)
    legend(fig); fig.subplots_adjust(left=0.2, right=0.98, top=0.91, bottom=0.34); return fig


def c4():
    """vertical: methods on the x axis, two adjacent stacked columns (controlled solid, knowledge hatched)"""
    fig, ax = plt.subplots(figsize=(3.1, 2.55))
    x = np.arange(N); w = 0.38
    ax.axvspan(x[-2] - 0.5, x[-1] + 0.5, color=HL, zorder=0, lw=0)
    for M, dx, kw in ((C, -w / 2, {}), (K, w / 2, dict(hatch="////", edgecolor="white", linewidth=0))):
        bot = np.zeros(N)
        for j, (_, _, col) in enumerate(TYPES):
            ax.bar(x + dx, M[:, j], bottom=bot, width=w, color=col, zorder=2, **kw); bot += M[:, j]
    ax.set_xticks(x); ax.set_xticklabels(NAMES, rotation=55, ha="right", fontsize=6.5)
    for lab in ax.get_xticklabels():
        if lab.get_text() in OURS:
            lab.set_fontweight("bold")
    ax.set_ylabel("Regret ($\\times$100)", fontsize=7.5); ax.tick_params(axis="y", labelsize=FS)
    legend(fig, [Patch(facecolor="#ddd", edgecolor="#666", label="Left: Controlled"), Patch(facecolor="#ddd", hatch="////", edgecolor="#666", label="Right: Knowledge-Req.")])
    fig.tight_layout(rect=(0, 0.22, 1, 1)); return fig


def c5():
    """average over the two groups as one stacked bar, with each group's total as a marker"""
    fig, ax = plt.subplots(figsize=(3.1, 2.75))
    y = np.arange(N); A = (C + K) / 2
    band(ax, y[-2] - 0.5, y[-1] + 0.5); stack(ax, y, A, 0.66)
    ax.scatter(C.sum(1), y, marker="D", s=11, color="black", zorder=4, label="Controlled Total")
    ax.scatter(K.sum(1), y, marker="o", s=12, facecolor="white", edgecolor="black", lw=0.8, zorder=4, label="Knowledge-Req. Total")
    style_ticks(ax, y, NAMES); ax.invert_yaxis(); ax.tick_params(axis="x", labelsize=FS)
    ax.set_xlabel("Regret ($\\times$100), Mean of Both Groups by Error Type", fontsize=7.2)
    h, l = ax.get_legend_handles_labels()
    legend(fig, h); fig.tight_layout(rect=(0, 0.27, 1, 1)); return fig


def c6():
    """composition (share of each error type, 100% bars) for both groups as paired thin bars, total regret printed at the end"""
    fig, ax = plt.subplots(figsize=(3.1, 3.4))
    y = np.arange(N) * 1.0
    band(ax, y[-2] - 0.5, y[-1] + 0.5)
    for M, dy, kw, tag in ((C, -0.19, {}, "C"), (K, 0.19, dict(hatch="////", edgecolor="white", linewidth=0), "K")):
        tot = M.sum(1); S = M / np.maximum(tot[:, None], 1e-9) * 100
        stack(ax, y + dy, S, 0.36, **kw)
        for i in range(N):
            ax.text(101.5, y[i] + dy, f"{tag} {tot[i]:.2f}", fontsize=5.6, va="center", color="#333")
    style_ticks(ax, y, NAMES); ax.invert_yaxis(); ax.set_xlim(0, 100); ax.tick_params(axis="x", labelsize=FS)
    ax.set_xlabel("Share of Regret by Error Type (%)", fontsize=7.5)
    legend(fig, [Patch(facecolor="white", edgecolor="#666", label="C: Controlled (Total)"), Patch(facecolor="#ddd", hatch="////", edgecolor="#666", label="K: Knowledge-Req. (Total)")])
    fig.tight_layout(rect=(0, 0.17, 0.86, 1)); return fig


if __name__ == "__main__":
    DESC = ["1. Paired bars per method (C above, K below, hatched), one axis",
            "2. Butterfly: Controlled to the left, Knowledge-required to the right, shared labels",
            "3. Two side-by-side panels sharing the method labels (one row per method)",
            "4. Vertical columns: methods on the x axis, two columns each (solid / hatched)",
            "5. One stacked bar (mean of both groups) + markers for each group's total",
            "6. Composition: 100% shares for both groups (paired), totals printed at the end"]
    with PdfPages(f"{OUTD}/fig_trap_candidates.pdf") as pp:
        for i, (f, d) in enumerate(zip([c1, c2, c3, c4, c5, c6], DESC), 1):
            fig = f(); p = f"{OUTD}/fig_trap_cand{i}.pdf"; fig.savefig(p); crop_pdf(p)
            hin = fig.get_size_inches()[1]; plt.close(fig)
            fig = f(); fig.suptitle(d, fontsize=7, y=1.06)
            pp.savefig(fig, bbox_inches="tight"); plt.close(fig)
    print("ok", OUTD)
