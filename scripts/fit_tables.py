"""Post-process paper/tables/*.tex: font size, our-method shading, and the vertical rule after the label column.

Font size (user rule 2026-10-10): every table is set in \\small (9pt) WITHOUT automatic shrinking, so no table is wrapped
in an adjustbox any more. Tables that are too wide are redesigned at the source (generator). Only the tables listed in
SHRINK_ALLOW (none at present) may still be wrapped; any adjustbox left from earlier runs is removed. \\scriptsize and
\\tiny are replaced by \\small."""
import glob, re, os
IS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHRINK_ALLOW = set()  # basenames of tables that truly cannot fit after redesign (report them)
_ADJ = re.compile(r"\\begin\{adjustbox\}\{[^}]*\}\n?(.*?)\n?\\end\{adjustbox\}", re.S)
for f in [x for x in glob.glob(f"{IS}/paper/tables/*.tex") if os.path.basename(x) != "diff.tex"]:
    s = open(f).read(); s0 = s
    if os.path.basename(f) in SHRINK_ALLOW:
        if "adjustbox" not in s:
            s = re.sub(r"(\\begin\{tabular\}.*?\\end\{tabular\})", r"\\begin{adjustbox}{max width=\\linewidth}\n\1\n\\end{adjustbox}", s, flags=re.S)
    else:
        s = _ADJ.sub(lambda m: m.group(1), s)
    if os.path.basename(f) != "qual.tex":  # qual.tex is a figure (boxed scenario text), not a table
        s = s.replace("\\scriptsize", "\\small").replace("\\tiny", "\\small")
    if s != s0:
        open(f, "w").write(s)


# ---- shade our method (DIVA variants) light yellow: rows whose first cell is a DIVA variant, and columns whose header is one
_OURS = re.compile(r"^\s*(\\textbf\{)?(\\diva\{\}|DIVA)")


class _O:
    @staticmethod
    def match(c):
        return _OURS.match(c) and not re.search(r"kappa|beta", c)


OURS = _O


def _split(row):
    """split a tabular row on top-level & (ignoring escaped \\&)"""
    cells, depth, cur = [], 0, ""
    i = 0
    while i < len(row):
        ch = row[i]
        if ch == "\\" and i + 1 < len(row):
            cur += row[i:i + 2]; i += 2; continue
        if ch == "{": depth += 1
        if ch == "}": depth -= 1
        if ch == "&" and depth == 0:
            cells.append(cur); cur = ""
        else:
            cur += ch
        i += 1
    cells.append(cur)
    return cells


def shade(body):
    lines = body.split("\n"); out = []; ours_cols = set(); header_done = False
    for ln in lines:
        st = ln.strip()
        if not st.endswith("\\\\") or "\\multicolumn" in st or st.startswith(("\\toprule", "\\midrule", "\\bottomrule", "\\cmidrule")):
            # header detection for column-method tables
            if st.endswith("\\\\") and not header_done and "\\multicolumn" not in st:
                pass
            out.append(ln); continue
        cells = _split(st[:-2])
        if not header_done:
            header_done = True
            ours_cols = {j for j, c in enumerate(cells) if j > 0 and OURS.match(c)}
            out.append(ln); continue
        if OURS.match(cells[0]):
            out.append("\\rowcolor{oursrow} " + ln.lstrip()); continue
        if ours_cols:
            cells = [("\\cellcolor{oursrow}" + c) if j in ours_cols else c for j, c in enumerate(cells)]
            out.append(" & ".join(cells) + " \\\\"); continue
        out.append(ln)
    return "\n".join(out)


NO_SHADE = {"generic_main.tex"}  # user 10-12: the generic rows also use DIVA's computation, so no row is shaded there
for f in [x for x in glob.glob(f"{IS}/paper/tables/*.tex") if os.path.basename(x) != "diff.tex"]:
    s = open(f).read()
    if "oursrow" in s or "tabular" not in s or os.path.basename(f) in NO_SHADE:
        continue
    s2 = re.sub(r"(\\begin\{tabular\}\{[^}]*\})(.*?)(\\end\{tabular\})", lambda m: m.group(1) + shade(m.group(2)) + m.group(3), s, flags=re.S)
    if s2 != s:
        open(f, "w").write(s2)


# ---- vertical rule after the label column(s) (user rule for all tables)
def vrule(m):
    spec = m.group(1)
    if "|" in spec or "@" in spec:  # already ruled, or a spec with @{} (regex sees only its first brace group)
        return m.group(0)
    k = 0
    while k < len(spec) and spec[k] == "l":
        k += 1
    if k == 0:
        k = 1
    return "\\begin{tabular}{" + spec[:k] + "|" + spec[k:] + "}"


for f in [x for x in glob.glob(f"{IS}/paper/tables/*.tex") if os.path.basename(x) != "diff.tex"]:
    if os.path.basename(f) in ("qual.tex",):
        continue
    s = open(f).read()
    s2 = re.sub(r"\\begin\{tabular\}\{([^}]*)\}", vrule, s)
    if s2 != s:
        open(f, "w").write(s2)


# ---- F6 (user 10-11): floats should fill their width. FILL maps a table file to how its top-level tabulars are set:
#   "text": stretch to \textwidth (table*), "col": stretch to \columnwidth (table), "to_col": table* -> table, stretched to
#   \columnwidth. Stretching uses tabular* with \extracolsep{\fill} (column spacing only; the font size is unchanged).
FILL = {
    "know_detail.tex": "text", "realized.tex": "text", "ci_know.tex": "text", "know1000.tex": "text", "corr.tex": "text", "prior_baselines.tex": "text", "f8_stop.tex": "text", "f8_mis.tex": "text", "f8_cost.tex": "text", "f8_us.tex": "text", "extra_bl.tex": "text", "sig_wins.tex": "text", "realized_big.tex": "col", "para_bl.tex": "col",
    "voiask_know.tex": "text", "episodes_know_prereg.tex": "text", "episodes_know300.tex": "text", "m1_decomp.tex": "text",
    "full.tex": "text", "episodes_know_exact.tex": "text", "episodes_know_own.tex": "text", "abl_eig.tex": "text",
    "transfer.tex": "col", "generic_main.tex": "col", "decomp.tex": "col", "ms_diag.tex": "col", "realized_ref.tex": "col", "controls.tex": "col",
    "ci_main.tex": "text", "large.tex": "col", "gold.tex": "col", "cost_all.tex": "col", "ref_metrics.tex": "col", "alt_ref.tex": "col",
    "think_know.tex": "to_col", "k4.tex": "to_col", "const.tex": "text", "fresh.tex": "to_col", "altgold.tex": "col",
    "full_know_llm.tex": "text", "full_ctrl_llm.tex": "text", "full_episodes_llm.tex": "text",
}


def _fill(s, width):
    out, i, depth = [], 0, 0
    tok = re.compile(r"\\begin\{tabular\*?\}|\\end\{tabular\*?\}")
    while True:
        m = tok.search(s, i)
        if not m:
            out.append(s[i:]); break
        out.append(s[i:m.start()]); t = m.group(0)
        if t.startswith("\\begin"):
            if depth == 0 and t == "\\begin{tabular}" and s[m.end():m.end() + 1] == "{":
                k = m.end() + 1; d = 1
                while d:
                    d += {"{": 1, "}": -1}.get(s[k], 0); k += 1
                spec = s[m.end() + 1:k - 1]
                spec = spec[3:] if spec.startswith("@{}") else spec
                out.append("\\begin{tabular*}{" + width + "}{@{\\extracolsep{\\fill}}" + spec + "}"); i = k; depth += 1
                stack.append(True); continue
            stack.append(False); depth += 1; out.append(t); i = m.end(); continue
        top = stack.pop(); depth -= 1
        out.append("\\end{tabular*}" if top else t); i = m.end()
    return "".join(out)


for name, mode in FILL.items():
    f = f"{IS}/paper/tables/{name}"
    if not os.path.exists(f):
        continue
    s = open(f).read(); s0 = s
    if "\\extracolsep{\\fill}" in s:
        continue
    if mode == "to_col":
        s = s.replace("\\begin{table*}", "\\begin{table}").replace("\\end{table*}", "\\end{table}")
        s = re.sub(r"\\setlength\{\\tabcolsep\}\{[^}]*\}", lambda _: "\\setlength{\\tabcolsep}{2pt}", s)  # single column: tighter spacing (font unchanged)
    stack = []
    s = _fill(s, "\\textwidth" if mode == "text" else "\\linewidth")
    if s != s0:
        open(f, "w").write(s)


# ---- F6 (user 10-11): pairs of small single-column tables of the same section are set side by side in one table*, each
#      with its own caption, so the float fills the text width and the appendix needs fewer float pages.
PAIRS = [("large.tex", "think_know.tex", 0.485, 0.485, None), ("k4.tex", "para.tex", 0.465, 0.515, "2pt"),
         ("dev_diag.tex", "gold.tex", 0.55, 0.43, "2pt"), ("realized_big.tex", "para_bl.tex", 0.49, 0.47, None)]  # (first, second, widths as fractions of \textwidth, tabcolsep or None)
_INNER = re.compile(r"\\begin\{table\*?\}(?:\[[^\]]*\])?\s*(.*?)\\end\{table\*?\}", re.S)
_SEP = re.compile(r"\\setlength\{\\tabcolsep\}\{[^}]*\}")


def _mp(x, w, sep):
    body = x.group(1).replace("\\centering\n", "", 1).strip()
    if sep:
        body = _SEP.sub(lambda _: "\\setlength{\\tabcolsep}{" + sep + "}", body)
    return "\\begin{minipage}[t]{" + str(w) + "\\textwidth}\n\\vspace{0pt}\\centering\n" + body + "\n\\end{minipage}"  # vspace: top-align the pair (F11)


for a, b, wa, wb, sep in PAIRS:
    fa, fb = f"{IS}/paper/tables/{a}", f"{IS}/paper/tables/{b}"
    if not (os.path.exists(fa) and os.path.exists(fb)):
        continue
    ia, ib = _INNER.search(open(fa).read()), _INNER.search(open(fb).read())
    if not (ia and ib):
        continue
    t = ("% F6: " + a + " and " + b + " side by side (generated by fit_tables.py; edit the two source tables, not this file)\n"
         "\\begin{table*}[t]\n\\centering\n" + _mp(ia, wa, sep) + "\\hfill\n" + _mp(ib, wb, sep) + "\n\\end{table*}\n")
    open(f"{IS}/paper/tables/pair_{a[:-4]}_{b[:-4]}.tex", "w").write(t)


# ---- user 10-12: family names stay on one line (complete.FAM_LABEL = {}), except in the tables too dense for it,
# where "DeepSeek-R1" is split over its two rows
SPLIT_FAMILY = {"full_know_llm.tex", "full_episodes_llm.tex", "para.tex", "pair_k4_para.tex"}
for f in [x for x in glob.glob(f"{IS}/paper/tables/*.tex") if os.path.basename(x) in SPLIT_FAMILY]:
    s = open(f).read()
    s2 = s.replace("{\\textbf{DeepSeek-R1}}", "{\\textbf{\\begin{tabular}[c]{@{}l@{}}DeepSeek-\\\\R1\\end{tabular}}}")
    if s2 != s:
        open(f, "w").write(s2)
