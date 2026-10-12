"""All result figures (vector PDF) -> paper/figs/. Run with the pathcal venv (matplotlib)."""
import json, os, sys, gzip, glob
import numpy as np
import collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
IS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{IS}/scripts")
OUT = f"{IS}/paper/figs"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "STIXGeneral", "mathtext.fontset": "stix", "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})
BLUE, ORANGE, GRAY, GREEN, PURPLE = "#1f77b4", "#e6781e", "#7f7f7f", "#2ca02c", "#9467bd"
DOMS = ["finance", "credit", "clinical", "legal"]


def _legend(ax, **kw):
    """legend entries in the canonical method order (complete.METHOD_ORDER)"""
    from complete import METHOD_ORDER
    h, lab = ax.get_legend_handles_labels()
    key = lambda x: METHOD_ORDER.index(x[1].replace("DIVA", "\\diva{}")) if x[1].replace("DIVA", "\\diva{}") in METHOD_ORDER else len(METHOD_ORDER)
    hl = sorted(zip(h, lab), key=key)
    ax.legend([a for a, _ in hl], [b for _, b in hl], frameon=False, **kw)


def J(n):
    p = f"{IS}/results/{n}.json"
    return json.load(open(p)) if os.path.exists(p) else {}


def read(p):
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in gzip.open(p, "rt")]


def fig_incoh(h=1.4, out=None, inside=False):  # h=1.4 (user 10-12, candidate 4)
    """(a) imagined vs exact belief change (binned, pooled over the probe LLMs); (b) top-1 check by decision value
    on controlled scenarios where acquiring is optimal: Chance | LLMs (mean, range) | DIVA (stated reliabilities).
    Data: scripts/incoh_probe.py (cached, recomputed when probe/scenario files change)."""
    from incoh_probe import load, summary
    from complete import PROBE_LLMS
    P = load(); S = summary(P)
    FS, FT, FSM, ORANGE_L, GRID = 7, 7.5, 6.5, "#f6c9a3", "#e6e6e6"
    XA = np.concatenate([v["X"] for v in P.values()]); YA = np.concatenate([v["Y"] for v in P.values()])
    E = np.quantile(XA, np.linspace(0, 1, 9)); E[-1] += 1e-9
    B = []
    for lo, hi in zip(E[:-1], E[1:]):
        m = (XA >= lo) & (XA < hi)
        B.append((XA[m].mean(), YA[m].mean(), *np.percentile(YA[m], [25, 75])))
    B = np.array(B)
    # h: figure height in inches (margins kept fixed in inches); inside: panel titles drawn inside the axes (saves the title band)
    fig = plt.figure(figsize=(3.1, h))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.95], wspace=0.95, left=0.135, right=0.975, bottom=0.376 / h,
                          top=1 - (0.03 if inside else 0.218) / h)
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    def style(a, t):
        a.tick_params(labelsize=FSM, pad=1.5, length=2.5, width=0.6)
        for sp in ("left", "bottom"):
            a.spines[sp].set_linewidth(0.6)
        if inside:
            a.text(0.02 if a is ax else 0.5, 0.98, t, transform=a.transAxes, fontsize=FT, ha="left" if a is ax else "center", va="top", zorder=6)
        else:
            a.set_title(t, fontsize=FT, loc="left", pad=3, x=0.0)
    # (a) belief change
    lim = 0.7
    ax.fill_between(B[:, 0], B[:, 2], B[:, 3], color=ORANGE, alpha=0.18, lw=0, zorder=1)
    ax.plot([0, lim], [0, lim], ls="--", color=BLUE, lw=1.0, zorder=2)
    ax.plot(B[:, 0], B[:, 1], "-o", color=ORANGE, ms=2.2, lw=1.3, zorder=3)
    ax.axhline(0, color=GRID, lw=0.5, zorder=0)
    ax.set_xlim(0, lim); ax.set_ylim(-0.1, lim); ax.set_xticks([0, 0.3, 0.6]); ax.set_yticks([0, 0.3, 0.6])
    ax.text(0.30, 0.47, "Exact (Bayes)", color=BLUE, fontsize=FSM, ha="right", va="center")  # one line (user 10-11)
    ax.text(0.69, 0.17, "LLM-Imagined", color=ORANGE, fontsize=FSM, ha="right", va="bottom")  # above the band (user 10-12)
    ax.set_xlabel("Exact Swing", fontsize=FS, labelpad=1); ax.set_ylabel("Imagined Swing", fontsize=FS, labelpad=1)
    style(ax, "(a) Belief Change")
    # (b) ranking checks by decision value
    vals = [P[t]["top1"] for t in lsort_(P) if P[t]["top1"] is not None]
    rows = [("Chance", S["chance"], GRAY), ("LLMs", S["top1_mean"], ORANGE), ("DIVA", 1.0, BLUE)]
    for i, (lab, v, c) in enumerate(rows):
        bx.barh(i, v, color=c, alpha=0.3 if lab != "DIVA" else 0.25, height=0.6, lw=0, zorder=1)
        bx.barh(i, v, color="none", edgecolor=c, height=0.6, lw=0.8, zorder=2)
    bx.errorbar([S["top1_mean"]], [1], xerr=[[S["top1_mean"] - min(vals)], [max(vals) - S["top1_mean"]]], fmt="none",
                ecolor="black", elinewidth=0.7, capsize=1.8, capthick=0.7, zorder=3)
    bx.text(S["chance"] + 0.03, 0, f"{S['chance']:.2f}", fontsize=FSM, va="center", ha="left", color=GRAY)
    bx.text(max(vals) + 0.03, 1, f"{S['top1_mean']:.2f}", fontsize=FSM, va="center", ha="left", color=ORANGE)
    bx.text(0.96, 2, "1.00", fontsize=FSM, va="center", ha="right", color="white", fontweight="bold")
    n = len(PROBE_LLMS)
    bx.set_yticks([0, 1, 2])
    bx.set_yticklabels(["Chance", f"LLMs ({len(vals)})" if len(vals) == n else f"LLMs ({len(vals)}/{n}, TBD)",
                        "DIVA"], fontsize=FSM, linespacing=0.95)
    for tl, c in zip(bx.get_yticklabels(), [GRAY, ORANGE, BLUE]):
        tl.set_color(c)
    bx.set_ylim(2.5, -0.5); bx.spines["left"].set_visible(False)
    bx.set_xlim(0, 1.04); bx.set_xticks([0, 0.5, 1]); bx.set_xticklabels(["0", "0.5", "1"])
    bx.set_xlabel("Top-1 by Decision Value", fontsize=FS, labelpad=1)
    style(bx, "(b) Ranking Checks"); bx.tick_params(axis="y", length=0, pad=1.5)
    fig.savefig(out or f"{OUT}/fig_incoh.pdf", dpi=300); plt.close(fig)
    print("fig_incoh:", {k: round(v, 3) for k, v in S.items()}, "missing:", [t for t in PROBE_LLMS if t not in P])


def lsort_(P):
    from complete import lsort
    return lsort(list(P))


def avg_methods(R, keys, metric, sub=True):
    out = {k: [] for k in keys}
    for tag, row in R.items():
        for k in keys:
            v = [row.get(d, {}).get(k + ("@sub" if sub else ""), {}).get(metric) for d in DOMS]
            v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
            if len(v) == 4:
                out[k].append(np.mean(v))
    return {k: (np.mean(v) if v else np.nan) for k, v in out.items()}


def fig_behavior():
    R = J("bin_main")
    ms = [("bl_direct", "Direct", GRAY), ("bl_cot_need", "CoT", GRAY), ("bl_voi_paper", "VoI-Ask", GRAY), ("bl_bed_llm", "BED-LLM", ORANGE), ("bl_pal", "PAL", GRAY),
          ("diva", "DIVA", BLUE)]
    from complete import mrank
    ms = sorted(ms, key=lambda x: mrank(x[0]))  # canonical (category) order
    fig, axs = plt.subplots(1, 3, figsize=(3.1, 1.55), sharey=True)  # F6: horizontal bars, readable method labels
    from complete import CTRL_BL_LLMS
    for ax, (met, title) in zip(axs, [("over_acq", "(a) Over-Acquire"), ("eig_trap", "(b) Info. Trap"), ("under_acq", "(c) Under-Acquire")]):
        vals = []
        for m, _, _ in ms:
            v = [R.get(t, {}).get(d, {}).get(m + "@sub", {}).get(met) for t in CTRL_BL_LLMS for d in DOMS]
            vals.append(float(np.mean(v)) if all(x is not None for x in v) else np.nan)
        ax.barh(range(len(ms)), [0 if np.isnan(x) else x for x in vals], color=[c for _, _, c in ms], height=0.7)
        for k, x in enumerate(vals):
            if np.isnan(x):
                ax.text(0.05, k, "TBD", fontsize=6, color="gray", va="center")
        ax.set_yticks(range(len(ms))); ax.set_yticklabels([n for _, n, _ in ms], fontsize=7)
        for lab in ax.get_yticklabels():
            if lab.get_text() == "DIVA":
                lab.set_fontweight("bold")
        ax.set_title(title, fontsize=7.5); ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", ".5", "1"], fontsize=7)
    axs[0].invert_yaxis()
    fig.text(0.6, 0.01, "Rate", ha="center", fontsize=7.5)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.86, bottom=0.2, wspace=0.12)
    fig.savefig(f"{OUT}/fig_behavior.pdf"); plt.close(fig)



def _plain_log(ax):
    """plain tick labels on log axes (1, 10 instead of 10^0, 10^1)"""
    from matplotlib.ticker import FuncFormatter, NullFormatter
    fmt = FuncFormatter(lambda v, _: (f"{v:g}"))
    for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
        if scale == "log":
            axis.set_major_formatter(fmt)
            axis.set_minor_formatter(NullFormatter() if axis is ax.xaxis else FuncFormatter(lambda v, _: f"{v:g}" if round(v, 3) in (0.3, 0.5, 2.0, 3.0, 5.0) else ""))


def _box_legend(fig, ax, ncol=2, h_in=0.32, plain=True):
    """F6 (user 10-11): appendix figures use the main-text legend style, boxed below the plot at 7.8pt"""
    h, l = ax.get_legend_handles_labels()
    if ax.get_legend():
        ax.get_legend().remove()
    W, H = fig.get_size_inches(); fig.set_size_inches(W, H + h_in)
    fig.legend(h, l, loc="lower center", ncol=ncol, fontsize=7.8, frameon=True, fancybox=False, edgecolor="#555555", framealpha=1,
               borderpad=0.4, handlelength=1.4, columnspacing=1.0, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, (h_in - 0.02) / (H + h_in), 1, 1))
    if plain:
        _plain_log(ax)


def fig_budget():
    E = J("episodes")
    from complete import EP_CTRL_LLMS
    curves = {"fiovoi_budget": [], "eig_budget": []}
    if not all(all(f"curve:{m}" in E.get(t, {}).get(d, {}) for m in curves for d in DOMS) for t in EP_CTRL_LLMS):
        E = {}
    for tag, row in {t: E[t] for t in EP_CTRL_LLMS if t in E}.items():
        for d in DOMS:
            o = row.get(d, {})
            for m in curves:
                c = o.get("curve:" + m)
                if c:
                    curves[m].append([c[str(k)] if str(k) in c else c.get(k) for k in range(7)])
    fig, ax = plt.subplots(figsize=(3.0, 1.8))
    for m, name, col in [("fiovoi_budget", "DIVA order", BLUE), ("eig_budget", "Information-gain order", ORANGE)]:
        A = np.array([x for x in curves[m] if None not in x], dtype=float)
        if len(A):
            mu = A.mean(0); se = A.std(0) / np.sqrt(len(A))
            ax.plot(range(7), mu, marker="o", ms=3, color=col, label=name)
            ax.fill_between(range(7), mu - se, mu + se, color=col, alpha=0.15)
    ax.set_xlabel("Number of Acquired Checks ($k$)"); ax.set_ylabel("Decision Utility")
    ax.legend(frameon=False, fontsize=6)
    _box_legend(fig, ax, 2); fig.savefig(f"{OUT}/fig_budget.pdf"); plt.close(fig)


def fig_scaling():
    R = J("bin_main")
    sizes = [("q06b", 0.6), ("q17b", 1.7), ("q4b", 4), ("q8", 8), ("q14", 14), ("q32", 32)]
    fig, ax = plt.subplots(figsize=(3.0, 1.8))
    for m, name, col in [("direct", "Direct", GRAY), ("diva", "DIVA", BLUE)]:
        xs, ys = [], []
        for t, s in sizes:
            v = [R.get(t, {}).get(d, {}).get(m, {}).get("dp_regret") for d in DOMS]
            if all(x is not None for x in v):
                xs.append(s); ys.append(100 * np.mean(v))
        if xs:
            ax.plot(xs, ys, marker="o", ms=3, color=col, label=name)
    never = [100 * R[t][d]["always_stop"]["dp_regret"] for t, _ in sizes if t in R for d in DOMS if d in R[t] and "always_stop" in R[t][d]]
    if never:
        ax.axhline(np.mean(never), ls="--", color=GREEN, lw=0.8, label="Never")
    ax.set_xscale("log"); ax.set_xlabel("Qwen3 Model Size (B)"); ax.set_ylabel("Regret ($\\times$100)")
    _legend(ax, fontsize=6)
    _box_legend(fig, ax, 3); fig.savefig(f"{OUT}/fig_scaling.pdf"); plt.close(fig)




def fig_scaling_know():
    KM = J("know_main")
    if not KM:
        return
    sizes = [("q06b", 0.6), ("q17b", 1.7), ("q4b", 4), ("q8", 8), ("q14", 14), ("q32", 32)]
    envs = ["tab:taiwan", "tab:german", "tab:cdc", "tab:heart", "sec:ret_1", "sec:ret_5", "sec:vol_5", "sec:vol_20"]
    fig, ax = plt.subplots(figsize=(3.0, 1.8))
    for m, name, col in [("direct", "Direct", GRAY), ("diva", "DIVA", BLUE), ("diva_u", "DIVA-U", ORANGE)]:
        xs, ys = [], []
        for tag, sz in sizes:
            v = [KM[e][tag][f"full:{m}"]["dp_regret"] / KM[e][tag]["full:never"]["dp_regret"] for e in envs
                 if tag in KM.get(e, {}) and f"full:{m}" in KM[e][tag] and KM[e][tag]["full:never"]["dp_regret"] > 0]
            if len(v) == 8:
                xs.append(sz); ys.append(float(np.exp(np.mean(np.log(v)))))
        if xs:
            ax.plot(xs, ys, marker="o", ms=3, color=col, label=name)
    ax.axhline(1.0, ls="--", color=GREEN, lw=0.8, label="Never")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Qwen3 Parameters (B)"); ax.set_ylabel("Regret / Never")
    _legend(ax, fontsize=6.5)
    _box_legend(fig, ax, 4); fig.savefig(f"{OUT}/fig_scaling_know.pdf"); plt.close(fig)


def fig_lc():
    L = J("learning_curve")
    if not L:
        return
    ns = [0, 10, 25, 50, 100, 200, 1000]
    fig, ax = plt.subplots(figsize=(3.0, 1.55))
    for kind, name, col in [("llm", "LLM Prior + Records", BLUE), ("generic", "Generic Prior + Records", ORANGE), ("flat", "Flat Prior + Records", GRAY)]:
        ys = []
        for n in ns:
            from complete import KNOW_LLMS
            if kind == "generic":
                v = [r[f"{kind}:{n}"] / r["never"] for r in L.values() if f"{kind}:{n}" in r and r["never"] > 0]
                ok = len(v) == 4
            else:
                keys = [f"{t}:{d}" for t in KNOW_LLMS for d in ("taiwan", "german", "cdc", "heart")]
                v = [L[k][f"{kind}:{n}"] / L[k]["never"] for k in keys if k in L and f"{kind}:{n}" in L[k] and L[k]["never"] > 0]
                ok = len(v) == len(keys)
            ys.append(np.mean(v) if ok else np.nan)
        if all(np.isnan(ys)):
            continue
        ax.plot([max(n, 3) for n in ns], ys, marker="o", ms=3, color=col, label=name)
    ax.axhline(1.0, ls="--", color=GREEN, lw=0.8, label="Never")
    ax.set_xscale("log"); ax.set_xticks([3, 10, 25, 100, 1000]); ax.set_xticklabels(["0", "10", "25", "100", "1000"])
    ax.set_xlabel("Number of Labeled Training Records"); ax.set_ylabel("Regret / Never")
    ax.legend(frameon=False, fontsize=6)
    _box_legend(fig, ax, 2, 0.5, plain=False); fig.savefig(f"{OUT}/fig_lc.pdf"); plt.close(fig)




# Styling of OUR rows (the trailing DIVA* rows) in fig_trap. None = default (no highlight).
# Options: "box" (v1), "outline" (v2), "band" (v3), "box_bold" (v4), "dashed_tag" (v5), "sep_outline" (v6).
HIGHLIGHT_STYLE = "band"  # user 10-10: yellow band + bold labels for our rows (fig_trap_hl_v3)
OURS_SHADE = (255 / 255, 248 / 255, 214 / 255)
OURS_ROWS = ("DIVA", "DIVA-U", "DIVA-DP")  # our rows (must be the trailing rows of the method order)
# Design drafts only: aggregate each method over the LLMs whose data for it is complete, print "(k/N)" at each bar end
# and stamp "DRAFT". False (paper default) = strict: a method is drawn only when complete on every LLM, else TBD.
DRAFT_PARTIAL = False  # Same as the tables' "ours" shading


def _trap_highlight(fig, ax, n_rows, n_ours, style):
    """Emphasize the last n_ours rows of a horizontal bar panel (rows inverted: index 0 on top)."""
    from matplotlib.patches import Rectangle
    if not style or n_ours <= 0:
        return
    lo, hi = n_rows - n_ours - 0.5, n_rows - 0.5  # data-y span of our rows (lo above hi after invert)
    ylim_low = max(ax.get_ylim())  # bottom of the inverted axis in data units
    hi_c = min(hi, ylim_low - 0.02)  # keep the box above the x-axis spine
    bold = style in ("band", "box_bold")
    if bold:
        for t in ax.get_yticklabels()[-n_ours:]:
            t.set_fontweight("bold")
    if style in ("outline", "sep_outline"):
        for i in range(n_rows - n_ours, n_rows):  # one outline around each full stacked bar
            segs = [list(c.patches)[i] for c in ax.containers]
            x0 = min(b.get_x() for b in segs); x1 = max(b.get_x() + b.get_width() for b in segs)
            if x1 > x0:
                b0 = segs[0]
                ax.add_patch(Rectangle((x0, b0.get_y()), x1 - x0, b0.get_height(), fill=False, ec="black", lw=1.2, zorder=4, clip_on=False))
    if style == "sep_outline":
        ax.axhline(lo, color="black", lw=0.6, ls="-", clip_on=False, zorder=4)
    if style == "band":
        ax.axhspan(lo + 0.05, hi_c, color=OURS_SHADE, zorder=0, lw=0)
        ax.set_axisbelow(True)
        r = fig.canvas.get_renderer()
        x0 = min(t.get_window_extent(r).x0 for t in ax.get_yticklabels()[-n_ours:]) - 2
        xa = ax.transAxes.transform((0, 0))[0]
        y_disp = [ax.transData.transform((0, y))[1] for y in (lo + 0.05, hi_c)]
        inv = fig.transFigure.inverted()
        (fx0, fy0), (fx1, fy1) = inv.transform((x0, min(y_disp))), inv.transform((xa, max(y_disp)))
        fig.patches.append(Rectangle((fx0, fy0), fx1 - fx0, fy1 - fy0, transform=fig.transFigure, color=OURS_SHADE, lw=0, zorder=-1))
    if style in ("box", "box_bold", "dashed_tag"):
        r = fig.canvas.get_renderer()
        x0 = min(t.get_window_extent(r).x0 for t in ax.get_yticklabels()[-n_ours:]) - 2.5
        x1 = ax.transAxes.transform((1, 0))[0] - 1
        y_disp = [ax.transData.transform((0, y))[1] for y in (lo + 0.07, hi_c)]
        inv = fig.transFigure.inverted()
        (fx0, fy0), (fx1, fy1) = inv.transform((x0, min(y_disp))), inv.transform((x1, max(y_disp)))
        ls = (0, (3, 1.5)) if style == "dashed_tag" else "-"
        fig.patches.append(Rectangle((fx0, fy0), fx1 - fx0, fy1 - fy0, transform=fig.transFigure, fill=False, ec="black", lw=1.0, ls=ls, zorder=5))
        if style == "dashed_tag":
            # "Ours" tag placed where it touches no bar: inside the box at right if our bars leave room, else just above the box
            def bar_end(i):  # right end (display px) of stacked bar i
                return max(ax.transData.transform((c.patches[i].get_x() + c.patches[i].get_width(), 0))[0] for c in ax.containers)
            tag = fig.text(0, 0, "Ours", fontsize=7, fontstyle="italic", fontweight="bold", zorder=6)
            tw = tag.get_window_extent(r).width + 6
            tx = fig.transFigure.transform((fx1, 0))[0] - 3
            if max(bar_end(i) for i in range(n_rows - n_ours, n_rows)) < tx - tw:
                tag.set_position((fx1 - 0.008, (fy0 + fy1) / 2)); tag.set_ha("right"); tag.set_va("center")
            else:
                tag.set_position((fx1 - 0.008, fy1 + 0.002)); tag.set_ha("right"); tag.set_va("bottom")
                if n_rows > n_ours and bar_end(n_rows - n_ours - 1) >= tx - tw:
                    tag.set_bbox(dict(boxstyle="square,pad=0.1", fc="white", ec="none"))


def fig_trap(style=None, out=None, partial=None):
    """Both panels: the Table-3 method list on the complete fixed LLM sets (complete.py); incomplete methods show TBD.
    style: highlight of OUR trailing DIVA* rows (default HIGHLIGHT_STYLE); out: output path stem (default paper/figs/fig_trap)."""
    style = HIGHLIGHT_STYLE if style is None else style
    out = out or f"{OUT}/fig_trap"
    partial = DRAFT_PARTIAL if partial is None else partial
    T = J("trap")
    if not T:
        return
    X = J("trap_extra")  # Never / Always ask / Reason. / DIVA-kappa / DIVA-beta / DIVA-DP (scripts/trap_extra.py)
    import complete as CP
    from complete import KNOW_LLMS, CTRL_BL_LLMS, REASON_KNOW, REASON_CTRL
    types = [("trap", "Decision-Irrelevant", "#e6781e"), ("placebo", "Uninformative", "#bbbbbb"), ("overpriced", "Overpriced", "#9467bd"),
             ("suboptimal", "Suboptimal Check", "#8c564b"), ("premature", "Premature Stop", "#2ca02c")]
    # (canonical key for complete.mrank, display name, cell key per panel; None = not applicable in that panel)
    meths = [("direct", "Direct", "direct", "Direct"),  # controlled: the Table-3 Direct baseline
             ("cot_need", "CoT", "cot_need", "cot_need"), ("sc_cot", "SC", "sc_cot", "sc_cot"),
             ("react", "ReAct", "react", "react"), ("finrisk_ask", "Ask-P", "finrisk_ask", "finrisk_ask"),
             ("voi_prompt", "VoI-P", "voi_prompt", "voi_prompt"), ("voi_paper", "VoI-Ask", "voi_paper", "voi_paper"),
             ("bed_llm", "BED-LLM", "bed_llm", "bed_llm"), ("pal", "PAL", "pal", "pal"),
             ("never", "Never", "never", "never"), ("always_top", "Always ask", "always_top", "always_top"),
             ("diva", "DIVA", "DIVA", "DIVA"),
             ("diva_u", "DIVA-U", "DIVA", "DIVA-U"),  # controlled: DIVA-U coincides with DIVA (exact gold)
]  # exactly the Table-3 rows (user rule 2026-10-11); no DIVA-DP in main-text floats (user 10-11)
    meths = sorted(meths, key=lambda r: CP.mrank(r[0])) if hasattr(CP, "mrank") else meths  # canonical METHOD_ORDER
    envs = ["taiwan", "german", "cdc", "heart", "ret_1", "ret_5", "vol_5", "vol_20"]
    # user 10-11: one row of two panels sharing the method labels, knowledge-required first (as in Tables 3 and 4)
    panels = [("knowledge", "(a) Knowledge-Req.", list(KNOW_LLMS), list(REASON_KNOW), envs, 3),
              ("controlled", "(b) Controlled", list(CTRL_BL_LLMS), list(REASON_CTRL), None, 2)]
    nr = len(meths)
    H = 1.05 + 0.12 * nr  # user 10-12: shorter
    fig, axs = plt.subplots(1, 2, figsize=(3.1, H), sharey=True, gridspec_kw={"wspace": 0.08, "width_ratios": [4.6, 6.2]})
    for ax, (tr, title, tags, rtags, ev, col_i) in zip(axs, panels):
        cells = {k: dict(v) for k, v in T.get(tr, {}).items()}
        for k, v in X.get(tr, {}).items():
            cells.setdefault(k, {}).update(v)
        rows = []  # (name, agg or None, status, k_llms, n_llms)
        for key, name, mc, mk in meths:
            mm = mc if tr == "controlled" else mk
            pool = rtags if key == "think" else tags
            if mm is None:
                rows.append((name, None, "na", 0, len(pool))); continue
            keys_of = (lambda t: [t]) if ev is None else (lambda t: [f"{t}:{e}" for e in ev])
            done = [t for t in pool if all(k in cells and mm in cells[k] for k in keys_of(t))]
            use = done if (partial or len(done) == len(pool)) else []
            if not use:
                rows.append((name, None, "tbd", len(done), len(pool))); continue
            agg = collections.defaultdict(float)
            for t in use:
                for k in keys_of(t):
                    for kk, v in cells[k][mm].items():
                        agg[kk] += v
            rows.append((name, agg, "ok", len(done), len(pool)))
        left = np.zeros(len(rows))
        for t, lab, col in types:
            vals = np.array([0.0 if a is None else 100 * a["reg_" + t] / a["n"] for _, a, *_ in rows])
            ax.barh(range(len(rows)), vals, left=left, color=col, label=lab, height=0.7, zorder=2)
            left += vals
        for k, (_, a, st, kk, nn) in enumerate(rows):
            if st != "ok":
                ax.text(0.02, k, "TBD" if st == "tbd" else "--", va="center", fontsize=7, color="gray", transform=ax.get_yaxis_transform())
        if partial:
            ax.set_xlim(0, 1.24 * max(left.max(), 1e-9))
            for k, (_, a, st, kk, nn) in enumerate(rows):
                if st == "ok":
                    ax.text(left[k] + 0.012 * ax.get_xlim()[1], k, f"({kk}/{nn})", va="center", fontsize=7, color="gray", zorder=3)
            n_o = 0  # stamp sits right of the rows just above ours (short bars), clear of the count labels
            while n_o < len(rows) and rows[len(rows) - 1 - n_o][0] in OURS_ROWS:
                n_o += 1
            ax.text(0.985, len(rows) - n_o - 1.5, "DRAFT", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=18,
                    fontweight="bold", color="#e4e4e4", zorder=0)
        n_ours = sum(1 for n, *_ in rows if n in OURS_ROWS)
        ax.axhspan(len(rows) - n_ours - 0.5, len(rows) - 0.5, color="#fdf5d3", lw=0, zorder=0)  # our rows highlighted
        if ax is axs[0]:
            ax.set_yticks(range(len(rows))); ax.set_yticklabels([n.replace("\\u2021", "\u2021") for n, *_ in rows], fontsize=7); ax.invert_yaxis()
            for t in ax.get_yticklabels():
                if t.get_text() in OURS_ROWS:
                    t.set_fontweight("bold")
        ax.set_title(title, fontsize=7.5, fontweight="bold"); ax.tick_params(axis="x", labelsize=7)
    h, lab = axs[0].get_legend_handles_labels()
    # user 10-12: two legend rows -- top: first two types, bottom: last three (column-major fill, one invisible spacer)
    from matplotlib.patches import Patch
    blank = Patch(facecolor="none", edgecolor="none")
    h = [h[0], h[2], h[1], h[3], blank, h[4]]; lab = [lab[0], lab[2], lab[1], lab[3], "", lab[4]]
    fig.legend(h, lab, fontsize=7.2, frameon=True, fancybox=False, edgecolor="#555555", framealpha=1, borderpad=0.35, handlelength=1.0,
               columnspacing=0.6, handletextpad=0.3, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.005))  # user 10-11: boxed
    fig.text(0.6, 0.162, "Regret ($\\times$100) by Error Type", ha="center", fontsize=7.5)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.935, bottom=0.285)
    if False:  # highlight drawn per panel above
        fig.canvas.draw()
        for ax in axs:
            labs = [t.get_text() for t in ax.get_yticklabels()]
            n_ours = 0
            while n_ours < len(labs) and labs[len(labs) - 1 - n_ours] in OURS_ROWS:
                n_ours += 1
            _trap_highlight(fig, ax, len(labs), n_ours, style)
    fig.savefig(f"{out}.pdf")
    if out != f"{OUT}/fig_trap":
        fig.savefig(f"{out}.png", dpi=300)
    plt.close(fig)


def fig_cost():
    S = J("analysis_strata")
    if not S:
        return
    regs = ["0.0", "0.05", "0.1", "0.2", "0.4"]
    fig, ax = plt.subplots(figsize=(3.0, 1.7))
    for m, name, col in [("direct", "Direct", GRAY), ("always_stop", "Never", GREEN), ("diva", "DIVA", BLUE)]:
        ys = []
        for r in regs:
            from complete import CTRL_LLMS
            v = [S.get(t, {}).get(f"cost:{r}|{m}", {}).get("dp_regret") for t in CTRL_LLMS]
            ys.append(100 * np.mean(v) if all(x is not None for x in v) else np.nan)
        ax.plot(range(len(regs)), ys, marker="o", ms=3, color=col, label=name, ls="--" if m == "always_stop" else "-")
    ax.set_xticks(range(len(regs))); ax.set_xticklabels(["0", "0.05", "0.1", "0.2", "0.4"])
    ax.set_xlabel("Cost Regime ($\\times\\min(g,l)$)"); ax.set_ylabel("Regret ($\\times$100)")
    ax.legend(frameon=True, fancybox=False, edgecolor="#555555", framealpha=1, fontsize=6.5, borderpad=0.4, handlelength=1.6,
              loc="upper left", bbox_to_anchor=(0.22, 0.99))  # boxed, moved right into the empty area (user 10-12)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_cost.pdf"); plt.close(fig)


def fig_llms():
    """Figure fig:llms (main text): per LLM, regret of the best LLM baseline chosen in hindsight (orange) and of DIVA
    (controlled, average over the four domains) / DIVA-U (knowledge-required, average over the eight environments).
    Controlled best baseline = the set of make_numbers.RedBin (prompting, decision-aware, information gain, program-aided,
    native reasoning where available); knowledge-required = the external baselines of Table tab:know. An LLM row without
    complete data shows TBD (no partial averages)."""
    from complete import CTRL_BL_LLMS, KNOW_LLMS, REASON_CTRL, LLM_FAMILIES, NAMES
    from make_tables import KNOW_METHS, KNOW_ENVS, EXTERNAL_GROUPS
    from complete import METHOD_GROUP, mname
    B, K = J("bin_main"), J("know_main")
    cbl = ["bl_direct", "bl_cot_need", "bl_sc_cot", "bl_react", "bl_voi_prompt", "bl_voi_paper", "bl_finrisk_ask", "bl_bed_llm", "bl_pal", "think"]

    def cval(t, m):
        v = [B.get(t, {}).get(d, {}).get(m + "@sub", {}) for d in DOMS]
        return float(np.mean([x["dp_regret"] for x in v])) if all(x.get("n", 0) >= 300 and x.get("dp_regret") is not None for x in v) else None

    def ctrl(t):
        ms = [m for m in cbl if m != "think" or t in REASON_CTRL]
        v = {m: cval(t, m) for m in ms}; dv = cval(t, "diva")
        return (min(v.values()), dv) if dv is not None and all(x is not None for x in v.values()) else None
    ext = [m for m, n in KNOW_METHS if METHOD_GROUP.get(mname(m)) in EXTERNAL_GROUPS]

    def kval(t, m):
        v = [K.get(e, {}).get(t, {}).get(f"s300:{m}", {}).get("dp_regret") for e, _ in KNOW_ENVS]
        return float(np.mean(v)) if all(x is not None for x in v) else None

    def know(t):
        v = [kval(t, m) for m in ext]; du = kval(t, "diva_u")
        return (min(v), du) if du is not None and all(x is not None for x in v) else None
    tags = [t for _, ts in LLM_FAMILIES for t in ts]
    y = {t: i for i, t in enumerate(tags)}
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 3.3), sharey=True)
    for ax, (title, fn, llms) in zip(axes, [("Knowledge-Required", know, KNOW_LLMS), ("Controlled", ctrl, CTRL_BL_LLMS)]):  # knowledge-required first (user 10-12)
        vals = [100 * x for t in llms for x in (fn(t) or ())]
        lo, hi = min(vals) / 1.6, max(vals) * 1.6
        for t in tags:
            r = fn(t) if t in llms else None
            if r is None:
                ax.text(np.sqrt(lo * hi), y[t], "TBD", color=GRAY, ha="center", va="center", fontsize=7); continue
            b, d = 100 * r[0], 100 * r[1]
            ax.annotate("", xy=(d, y[t]), xytext=(b, y[t]), arrowprops=dict(arrowstyle="-|>", color="black", lw=0.7, shrinkA=2.5, shrinkB=2.5, mutation_scale=6))
            first = not ax.get_legend_handles_labels()[1]
            ax.scatter([b], [y[t]], s=14, color=ORANGE, zorder=3, label="Best LLM Baseline" if first else None)
            ax.scatter([d], [y[t]], s=14, color=BLUE, zorder=3, label="DIVA (-U)" if first else None)
        ax.set_xscale("log"); ax.set_xlim(lo, hi); ax.set_title(title, fontsize=8)
        ax.set_xlabel("Regret ($\\times$100)")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks([x for x in ((0.3, 1, 3, 10) if hi / lo > 15 else (0.5, 1, 2, 3, 5)) if lo <= x <= hi])
        k = 0
        for f, ts in LLM_FAMILIES[:-1]:
            k += len(ts); ax.axhline(k - 0.5, color="#dddddd", lw=0.5)
        ax.grid(axis="y", color="#eeeeee", lw=0.4, ls="--")
    axes[0].set_yticks(range(len(tags)))
    axes[0].set_yticklabels(["14B" if t == "phi4" else NAMES[t].rsplit("-", 1)[-1] for t in tags], fontsize=7)
    axes[0].set_ylim(len(tags) - 0.5, -0.5)
    k = 0
    for f, ts in LLM_FAMILIES:
        axes[0].text(-0.3, 1 - (k + len(ts) / 2) / len(tags), f, transform=axes[0].transAxes, ha="right", va="center", fontsize=7.5, fontweight="bold") if f != "Phi" else \
            axes[0].text(-0.3, 1 - (k + len(ts) / 2) / len(tags), "Phi-4", transform=axes[0].transAxes, ha="right", va="center", fontsize=7.5, fontweight="bold")
        k += len(ts)
    h, l = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.3, right=0.98, bottom=0.205, top=0.95, wspace=0.08)  # legend closer to the x labels (user 10-12)
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7.8, frameon=True, fancybox=False, edgecolor="#555555", framealpha=1, borderpad=0.4,
               handlelength=1.4, columnspacing=1.2, bbox_to_anchor=(0.5, 0.005))  # user 10-11: below the plot, larger and boxed, as fig_trap
    fig.savefig(f"{OUT}/fig_llms.pdf"); plt.close(fig)


if __name__ == "__main__":
    for f in sys.argv[1:] or ["fig_incoh", "fig_behavior", "fig_budget", "fig_scaling", "fig_lc", "fig_trap", "fig_cost", "fig_scaling_know", "fig_llms"]:
        try:
            globals()[f](); print("ok", f)
        except Exception as e:
            print("FAIL", f, repr(e))
    from crop_figs import crop_pdf  # user 10-11: no white margin on any side of a figure
    for f in sys.argv[1:] or ["fig_incoh", "fig_behavior", "fig_budget", "fig_scaling", "fig_lc", "fig_trap", "fig_cost", "fig_scaling_know", "fig_llms"]:
        if os.path.exists(f"{OUT}/{f}.pdf"):
            crop_pdf(f"{OUT}/{f}.pdf")
