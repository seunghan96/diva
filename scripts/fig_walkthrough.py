"""End-to-end worked example of DIVA (Appendix app:walkthrough) from results/walkthrough_phi4_taiwan71.json, which
scripts/walkthrough_compute.py writes with the codebase's own functions (diva_scores, diva_u, spec_dp_first, score):
  paper/figs/fig_walkthrough.pdf      imagined beliefs | decision value vs cost | EIG | DP regret per policy
  paper/tables/walkthrough_tab.tex    every quantity of Steps 1-4 per check (table*)
  paper/tables/walkthrough_prompt.tex the scenario, the direct prompt suffix, the elicitation questions and the raw answers (figure*)
Run with the pathcal venv (matplotlib)."""
import json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
IS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = json.load(open(f"{IS}/results/walkthrough_phi4_taiwan71.json"))
plt.rcParams.update({"font.size": 8, "font.family": "STIXGeneral", "mathtext.fontset": "stix", "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})
BLUE, ORANGE, GRAY, RED = "#1f77b4", "#e6781e", "#7f7f7f", "#b23a48"
C = D["checks"]; lab = [f"{c['label']}: {c['name']}" for c in C]; y = np.arange(len(C))[::-1]
pick = D["choices"]["diva"]; trap = D["choices"]["direct"]
col = lambda c: BLUE if c["label"] == pick else (ORANGE if c["label"] == trap else GRAY)


def fig():
    fig, ax = plt.subplots(1, 4, figsize=(7.0, 2.25), gridspec_kw={"width_ratios": [1.35, 1, 0.8, 1.05]})
    a = ax[0]; thr = D["threshold"]
    a.axvspan(thr, 0.75, color=RED, alpha=0.07, lw=0); a.axvline(thr, color=RED, ls="--", lw=0.9)
    a.text(thr + 0.015, 1.5, "Decline\n($>b^\\dagger$)", color=RED, fontsize=6.5, va="center")
    for c, yy in zip(C, y):
        a.plot([c["b0"], c["b1"]], [yy, yy], color=col(c), lw=1.6, zorder=1)
        a.scatter([c["b1"]], [yy], s=18, color=col(c), zorder=3)
        a.scatter([c["b0"]], [yy], s=18, facecolor="white", edgecolor=col(c), zorder=3)
        a.scatter([D["b"]], [yy], s=8, color="black", zorder=4)
    a.set_yticks(y); a.set_yticklabels(lab, fontsize=7); a.set_xlim(0, 0.75); a.set_xlabel("Imagined Belief $\\hat b(e,v)$")
    a.set_title("(a) Imagined Beliefs", fontsize=8)
    b = ax[1]
    b.barh(y, [c["voi"] for c in C], color=[col(c) for c in C], height=0.55)
    for c, yy in zip(C, y):
        b.plot([c["cost"]] * 2, [yy - 0.38, yy + 0.38], color="black", lw=1.2)
    b.set_yticks(y); b.set_yticklabels([]); b.set_xlim(0, 0.32); b.set_xlabel("Decision Value (Bar) and Cost (Tick)")
    b.set_title("(b) Decision Value", fontsize=8)
    c_ = ax[2]
    c_.barh(y, [c["eig"] for c in C], color=[col(c) for c in C], height=0.55)
    c_.set_yticks(y); c_.set_yticklabels([]); c_.set_xlabel("EIG (Bits)"); c_.set_title("(c) Information Gain", fontsize=8)
    d = ax[3]
    ms = [("direct", "Direct"), ("eig", "EIG"), ("never", "Never"), ("diva", "DIVA"), ("diva_u", "DIVA-U"), ("diva_dp", "DIVA-DP")]
    yy = np.arange(len(ms))[::-1]
    vals = [100 * D["regret"][m]["dp_regret"] for m, _ in ms]
    cs = [ORANGE if D["choices"][m] == trap else (GRAY if D["choices"][m] == "S" else BLUE) for m, _ in ms]
    d.barh(yy, vals, color=cs, height=0.55)
    for v, t, (m, _) in zip(vals, yy, ms):
        d.text(v + 0.06, t, f"{v:.2f} ({D['choices'][m]})", va="center", fontsize=6.5)
    d.set_yticks(yy); d.set_yticklabels([n for _, n in ms], fontsize=7); d.set_xlim(0, 4.6)
    d.set_xlabel("DP Regret ($\\times$100)"); d.set_title("(d) Outcome", fontsize=8)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(f"{IS}/paper/figs/fig_walkthrough.pdf"); plt.close(fig)


def f2(x):
    return f"{x:.2f}"


def table():
    rows = []
    for c in C:
        sh = "\\rowcolor{oursrow} " if c["label"] == pick else ""
        ref = D["ref"]  # reference model (scoring only): likelihoods are not stored; report its imagined beliefs
        rows.append(sh + f"{c['label']} & {c['cost']:.2f} & {100*D['raw']['chk'][c['label']]['c1'][0]:.0f}\\% / {100*D['raw']['chk'][c['label']]['c0'][0]:.0f}\\% / {100*D['raw']['chk'][c['label']]['pf'][0]:.0f}\\% & "
                    f"{c['s1']:.3f} / {c['s0']:.3f} & {c['p1']:.3f} & {c['b1']:.3f} / {c['b0']:.3f} & {c['voi']:.3f} & "
                    f"{'$+$' if c['net'] > 0 else '$-$'}{abs(c['net']):.3f} & {'$+$' if c['net_u'] > 0 else '$-$'}{abs(c['net_u']):.3f} & "
                    f"{D['dp_top'][c['label']]:.3f} & {c['eig']:.3f} \\\\")
    head = ("Check & Cost & $\\hat r_e(1)$ / $\\hat r_e(0)$ / $\\hat P(f)$ & $\\hat\\ell_e(1|1)$ / $\\hat\\ell_e(1|0)$ & $\\hat P(e{=}1)$ & "
            "$\\hat b(e,1)$ / $\\hat b(e,0)$ & VOI & Net & Net (U) & DP & EIG \\\\")
    tex = ("\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{2pt}\n\\begin{tabular}{l|cccccccccc}\n\\toprule\n" + head + "\n\\midrule\n"
           + "\n".join(rows) + "\n\\midrule\nStop and decide & -- & -- & -- & -- & " + f"{D['b']:.3f}" + " & -- & -- & -- & " + f"{D['dp_top']['S']:.3f}" + " & -- \\\\\n"
           "\\bottomrule\n\\end{tabular}\n"
           "\\caption{\\textbf{Steps 1--4 of the worked example} (Figure~\\ref{fig:walkthrough}). $\\hat r_e(1)$ / $\\hat r_e(0)$ / $\\hat P(f)$: Elicited risks and frequency of the finding. "
           "$\\hat\\ell_e$: Likelihoods after shrinkage. Net (U): \\diva{}-U. DP: Value of acquiring the check first under \\diva{}-DP.}\n"
           "\\label{tab:walkthrough}\n\\end{table*}\n")
    open(f"{IS}/paper/tables/walkthrough_tab.tex", "w").write(tex)


def esc(s):
    s = s.replace("\\", "\\textbackslash{}")
    for a, b in [("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_"), ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}")]:
        s = s.replace(a, b)
    return s.replace("\\textbackslash\\{\\}", "\\textbackslash{}")


def lines(s):
    """verbatim-like lines: line breaks between lines, a small gap for a blank line ("\\relax" guards a line starting with "[")"""
    out, gap = [], False
    for l in s.split("\n"):
        if not l.strip():
            gap = True; continue
        out.append(("" if not out else ("\\\\[3pt]" if gap else "\\\\") + "\\relax ") + esc(l)); gap = False
    return "\n".join(out)


def prompt():
    c = [x for x in C if x["label"] == pick][0]
    sc = D["text"].strip()
    sfx = D["direct_prompt"].rsplit("\n\n", 1)[1].strip()  # the direct prompt also letters the stop option and rotates its position
    raw = D["raw"]["chk"][pick]
    pct = lambda v: f"{100*v[0]:.0f}%"
    q = ("Question: Based on the information above and your general knowledge, among cases where the check '" + c["name"] + "' shows this finding (" + c["finding"]
         + "), what share end up in the case where the adverse case (as defined above) occurs?\nAnswer with a single percentage between 0% and 100% (e.g. 37%), nothing else.")
    box = lambda title, body: ("\\fbox{\\begin{minipage}[t]{0.475\\textwidth}\\raggedright\\scriptsize\\textbf{" + title + "}\\par\\smallskip\\ttfamily " + body + "\\end{minipage}}")
    left = box("Scenario (identical for every method)", lines(sc) + "\\\\\\smallskip\\normalfont\\textbf{Direct policy suffix:}\\ttfamily\\\\" + lines(sfx)
               + "\\\\\\smallskip\\normalfont\\textbf{Phi-4:} " + ", ".join(f"({k}) {v:.3f}" for k, v in sorted(D["direct_dist"].items(), key=lambda kv: -kv[1])[:3]))
    right = box("DIVA, Step 1: Elicitation for check " + pick,
                "\\normalfont System: \\ttfamily " + esc("You are a careful decision analyst. Answer exactly in the requested format.")
                + "\\\\\\smallskip\\normalfont User: \\ttfamily [scenario above]\\\\\\smallskip " + lines(q)
                + "\\\\\\smallskip\\normalfont Assistant (prefilled): \\ttfamily Answer: \\textbf{" + esc(pct(raw["c1"])) + "}"
                + "\\\\\\bigskip\\normalfont The same template asks for the risk when the finding is absent (\\textbf{" + esc(pct(raw["c0"])) + "}), the frequency of the finding (\\textbf{"
                + esc(pct(raw["pf"])) + "}), and the base rate (\\textbf{" + esc(pct(D["raw"]["prior"])) + "}), and the same three questions for every other check and known finding: "
                + str(1 + 3 * (len(C) + len(D["obs_steps"]))) + " greedy answers of at most six tokens, asked once per episode.")
    tex = ("\\begin{figure*}[t]\n\\centering\n" + left + "\\hfill" + right + "\n"
           "\\caption{\\textbf{Prompts and raw answers of the worked example.} Left: The scenario and the direct policy's question, with Phi-4's three most likely letters. "
           "Right: One elicitation question of \\diva{} for check " + pick + " and Phi-4's raw answers.}\n\\label{fig:walkthrough_prompt}\n\\end{figure*}\n")
    open(f"{IS}/paper/tables/walkthrough_prompt.tex", "w").write(tex)


def utable():
    """Table tab:walkthrough_u: the DIVA-U valuation of checks C and F step by step (log-odds shift +-sigma*z/2, sigma=1,
    three-point Gauss-Hermite nodes and weights); the weighted net reproduces the stored DIVA-U net value."""
    import math
    b, g, l = D["b"], D["g"], D["l"]
    V = lambda q: max((1 - q) * g - q * l, 0.0)
    lo = lambda p: math.log(p / (1 - p)); sg = lambda x: 1 / (1 + math.exp(-x))
    nodes = [(-math.sqrt(3), 1 / 6, "$z{=}-\\sqrt{3}$"), (0.0, 2 / 3, "$z{=}0$ (as elicited)"), (math.sqrt(3), 1 / 6, "$z{=}+\\sqrt{3}$")]
    rows = []
    for lab in ("C", "F"):
        c = [x for x in C if x["label"] == lab][0]
        tot = 0.0
        for j, (z, w, nm) in enumerate(nodes):
            a, bb = sg(lo(c["s1"]) + z / 2), sg(lo(c["s0"]) - z / 2)
            p1 = b * a + (1 - b) * bb; q1 = b * a / p1; q0 = b * (1 - a) / (1 - p1)
            voi = p1 * V(q1) + (1 - p1) * V(q0) - V(b); net = voi - c["cost"]; tot += w * net
            sep = lo(a) - lo(bb); wstr = "2/3" if j == 1 else "1/6"
            sgn = lambda x: ("$+$" if x > 0 else ("$-$" if x < 0 else "")) + f"{abs(x):.3f}"
            first = ("\\multirow{-3}{*}{" + lab + "}") if j == 2 else ""  # in the last row, so the shaded middle row does not cover it
            rows.append((("\\rowcolor{oursrow} " if lab == pick and j == 1 else "") + first + f" & {nm} & {wstr} & {100*a:.1f}\\% / {100*bb:.1f}\\% & "
                         f"{sgn(sep)} & {p1:.3f} & {q1:.3f} / {q0:.3f} & {voi:.3f} & {sgn(net)} \\\\"))
        assert abs(tot - c["net_u"]) < 1e-6, (lab, tot, c["net_u"])  # reproduces the stored DIVA-U net value
        rows.append(f" & \\multicolumn{{7}}{{r}}{{Weighted net value of \\diva{{}}-U}} & {sgn(tot)} \\\\")
        if lab == "C":
            rows.append("\\midrule")
    head = ("Check & Case & Weight & $\\ell_e(1|1)$ / $\\ell_e(1|0)$ & Strength & $\\hat P(e{=}1)$ & $\\hat b(e,1)$ / $\\hat b(e,0)$ & VOI & Net \\\\")
    tex = ("\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l|cccccccc@{}}\n\\toprule\n" + head
           + "\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n"
           "\\caption{\\textbf{\\diva{}-U in the worked example.} Log-odds shifts of $\\pm\\sigma z/2$ ($\\sigma=1$) at the weighted Gauss--Hermite nodes $z$."
           " Strength: Log-odds difference of the two likelihoods. $\\hat b(e,v)$: Imagined belief ($b^\\dagger=0.375$).}\n\\label{tab:walkthrough_u}\n\\end{table*}\n")
    open(f"{IS}/paper/tables/walkthrough_u.tex", "w").write(tex)


if __name__ == "__main__":
    fig(); table(); prompt(); utable()
    from crop_figs import crop_pdf  # user 10-11: no white margin on any side of a figure
    crop_pdf(f"{IS}/paper/figs/fig_walkthrough.pdf"); print("ok")
