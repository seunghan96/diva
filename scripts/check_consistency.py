"""Consistency checker for paper/tables/*.tex (user rules 2026-10-10). Run after the table generators, fit_tables,
apply_captions and add_avg_rank (rebuild_all.sh). Exit code 1 on any violation.

Asserts, for every tabular:
  1. canonical METHOD order (complete.METHOD_ORDER) of the method names in the header row(s) and in the row labels;
     a repeated name starts a new group (e.g. DP / myopic blocks, the qualitative-belief block of tab:ms);
  2. canonical environment / domain order (complete.ENV_ORDER, DOM_ORDER) of environment names in headers and row labels;
  3. LLM rows grouped by family (Family | Size columns, complete.family_rows): families in LLM_FAMILIES order, sizes in
     ascending order, every LLM of the family present, a rule between families; no per-LLM row labelled with a full
     model name (such tables must use the family grouping);
  4. font size \\small (no \\scriptsize / \\tiny, \\footnotesize only for allow-listed full tables), no adjustbox outside
     fit_tables.SHRINK_ALLOW, a vertical rule after the label column, 'Avg.' (not Average / Mean) as the average header.
"""
import glob, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from complete import METHOD_ORDER, METHOD_GROUP, LLM_FAMILIES, SIZE_OF, NAMES, ENV_ORDER, DOM_ORDER, ENVN, _mbase

FOOTNOTE_ALLOW = set()  # basenames allowed to use \footnotesize (wide Full Results tables); none at present (ACL: \footnotesize = \small = 9pt)
SHRINK_ALLOW = set()
ENV_SEQ = [ENVN[e] for e in ENV_ORDER + DOM_ORDER]
FAMS = [f for f, _ in LLM_FAMILIES]


def split_cells(row):
    cells, depth, cur, i = [], 0, "", 0
    while i < len(row):
        ch = row[i]
        if ch == "\\" and i + 1 < len(row):
            cur += row[i:i + 2]; i += 2; continue
        depth += {"{": 1, "}": -1}.get(ch, 0)
        if ch == "&" and depth == 0:
            cells.append(cur.strip()); cur = ""
        else:
            cur += ch
        i += 1
    cells.append(cur.strip())
    return cells


def plain(c):
    """cell text without colour, emphasis, multirow and nested tabular wrappers"""
    c = re.sub(r"\\rowcolor\{[^}]*\}|\\cellcolor\{[^}]*\}", "", c)
    c = re.sub(r"\\multirow\{[^}]*\}\{[^}]*\}(\[[^\]]*\])?", "", c)
    c = re.sub(r"\\multicolumn\{[^}]*\}\{[^}]*\}", "", c)
    c = re.sub(r"\\begin\{tabular\}(\[[a-z]\])?\{(@\{\}|[^{}])*\}", "", c)
    c = c.replace("\\end{tabular}", "").replace("\\shortstack[l]{", "")
    c = re.sub(r"\\(textbf|underline|textit)\{", "", c)
    c = c.replace("\\\\", " ").strip()
    while c.startswith("{") and c.endswith("}") and c.count("{") == 1:
        c = c[1:-1].strip()
    c = re.sub(r"\s+", " ", c).replace("- ", "-")
    while c.count("}") > c.count("{") and c.endswith("}"):
        c = c[:-1]
    return c.strip()


def tabular_bodies(s):
    """(spec, body) of every outermost tabular"""
    out, i = [], 0
    while True:
        mt = re.compile(r"\\begin\{tabular\*?\}").search(s, i)
        if not mt:
            return out
        j = mt.start(); k = mt.end()
        if mt.group(0).endswith("*}"):  # tabular*{width}{spec}: skip the width (F6)
            dd, k = 0, k
            while True:
                dd += {"{": 1, "}": -1}.get(s[k], 0); k += 1
                if dd == 0:
                    break
        if s[k] == "[":
            k = s.index("]", k) + 1
        depth, m = 0, k
        while True:
            depth += {"{": 1, "}": -1}.get(s[m], 0); m += 1
            if depth == 0:
                break
        spec = s[k + 1:m - 1]
        d, p = 1, m
        while d:
            ma = re.compile(r"\\begin\{tabular\*?\}").search(s, p); mb = re.compile(r"\\end\{tabular\*?\}").search(s, p)
            a = ma.start() if ma else -1; b = mb.start() if mb else -1
            if a != -1 and a < b:
                d += 1; p = a + 1
            else:
                d -= 1; p = b + 1
        out.append((spec, s[m:p - 1]))
        i = p


def rows_of(body):
    body = re.sub(r"(?<!\\)%.*", "", body)
    rows, cur, depth, i = [], "", 0, 0
    while i < len(body):
        if body.startswith("\\begin{tabular}", i):
            depth += 1
        if body.startswith("\\end{tabular}", i):
            depth -= 1
        if body.startswith("\\\\", i) and depth == 0:
            rows.append(cur); cur = ""; i += 2; continue
        cur += body[i]; i += 1
    rows.append(cur)
    out = []
    for r in rows:
        r = re.sub(r"\\(toprule|midrule|bottomrule)", "", r)
        r = re.sub(r"\\cmidrule(\([a-z]*\))?\{[^}]*\}", "", r).strip()
        if r:
            out.append(r)
    return out


def order_ok(seq, ref):
    """seq of names in ref; split into groups at repeats; each group must follow ref order"""
    bad, seen, last = [], set(), -1
    for x in seq:
        k = ref.index(x)
        if x in seen:
            seen, last = set(), -1
        if k < last:
            bad.append(x)
        seen.add(x); last = max(last, k)
    return bad


ALL_METHODS = set()


def check_file(f):
    s = open(f).read(); name = os.path.basename(f); errs = []
    if "\\tiny" in s or "\\scriptsize" in s:
        errs.append("uses \\scriptsize or \\tiny")
    if "\\footnotesize" in s and name not in FOOTNOTE_ALLOW and name != "qual.tex":
        errs.append("uses \\footnotesize")
    if "adjustbox" in s and name not in SHRINK_ALLOW:
        errs.append("wrapped in adjustbox")
    if "\\begin{table" in s and "\\small" not in s:
        errs.append("no \\small")
    for spec, body in tabular_bodies(s):
        if "|" not in spec and name != "qual.tex":
            errs.append(f"no vertical rule after the label column ({spec})")
        k = body.find("\\midrule")
        head_rows = rows_of(body[:k]) if k >= 0 else []
        rows = rows_of(body[k:]) if k >= 0 else rows_of(body)
        hmeth, henv, meth, envs, fams = [], [], [], [], []
        for r in head_rows:  # header: column order (cells spanning several columns are group labels, not environments)
            for c in [plain(x) for x in split_cells(r) if not re.match(r"\s*\\multicolumn\{[2-9]", x)]:
                if c in ("Average", "Mean"):
                    errs.append(f"average header '{c}' (use Avg.)")
                b = _mbase(c)
                if b:
                    hmeth.append(b); ALL_METHODS.add(b)
                if c in ENV_SEQ:
                    henv.append(c)
        cur_g, has_g = None, "\\textit{" in body and re.search(r"\\multicolumn\{\d+\}\{l\|?\}\{\\textit\{", body)
        for r in rows:  # body: row order (label cells only)
            mg = re.match(r"\s*\\multicolumn\{\d+\}\{l\|?\}\{\\textit\{([^}]*)\}\}(?:[\s&]|\\multicolumn\{1\}\{c\|?\}\{\})*$", r)
            if mg:
                cur_g = mg.group(1); continue
            cells = [plain(c) for c in split_cells(r)]
            lab = cells[:3]
            for c in lab:
                b = _mbase(c)
                if b:
                    meth.append(b)
                    ALL_METHODS.add(b)
                    if has_g and METHOD_GROUP.get(b) != cur_g:
                        errs.append(f"{b} under category '{cur_g}' (expected '{METHOD_GROUP.get(b)}')")
                    break
            for c in lab:
                if c in ENV_SEQ:
                    envs.append(c); break
            if any(c in NAMES.values() for c in lab):
                errs.append(f"per-LLM row with a full model name ({[c for c in lab if c in NAMES.values()][0]}); use the family grouping")
            for c in lab:
                if c in FAMS:
                    fams.append([c, []])
                    break
            if fams and len(cells) > 1:
                idx = [i for i, c in enumerate(cells[:4]) if c in SIZE_OF.values()]
                if idx:
                    fams[-1][1].append(cells[idx[0]])
        for seq, what in [(hmeth, "column"), (meth, "row")]:
            bad = order_ok(seq, METHOD_ORDER)
            if bad:
                errs.append(f"method {what} order violated at {bad} in {seq}")
        # merged Table tab:know: knowledge-required | controlled column groups, each in canonical order
        _ctl = set(ENVN[d] for d in DOM_ORDER)
        for seq, what in [([e for e in henv if e not in _ctl], "column"), ([e for e in henv if e in _ctl], "column"), (envs, "row")]:
            bad = order_ok(seq, ENV_SEQ)
            if bad:
                errs.append(f"environment {what} order violated at {bad} in {seq}")
        if fams:
            seq = [f for f, _ in fams]
            if order_ok(seq, FAMS):
                errs.append(f"family order violated: {seq}")
            ref = dict(LLM_FAMILIES)
            for fam, sizes in fams:
                want = [SIZE_OF[t] for t in ref[fam]]
                if sizes != want:
                    errs.append(f"{fam}: sizes {sizes} != {want} (all LLMs, ascending size)")
            if body.count("\\cmidrule") + body.count("\\midrule") < len(fams):
                errs.append("missing rule between families")
    return errs


def main():
    files = sorted(f for f in glob.glob(f"{IS}/paper/tables/*.tex"))
    n_bad = 0
    for f in files:
        errs = check_file(f)
        for e in errs:
            print(f"{os.path.basename(f)}: {e}")
        n_bad += bool(errs)
    # text coverage: Section 4.1 'Baseline methods.' names every row of tab:know / tab:main; Appendix B.1 every method of any table
    sec = open(f"{IS}/paper/sections/experiments.tex").read()
    para = sec[sec.index("\\textbf{Baseline methods.}"):sec.index("\\textbf{Evaluation metrics.}")]
    app = open(f"{IS}/paper/sections/appendix.tex").read()
    b1 = app[app.index("\\label{app:baselines}"):app.index("\\subsection{Evaluation Metrics}")]
    for fn in ["main_know.tex"]:  # tab:main is merged into tab:know
        for spec, body in tabular_bodies(open(f"{IS}/paper/tables/{fn}").read()):
            k = body.find("\\midrule")
            for r in rows_of(body[k:]):
                b = _mbase(plain(split_cells(r)[0]))
                if b and b not in para and not b.startswith(("\\diva{}", "Generic")):  # our rows need no listing (user 10-11 removed 'Ours are ...')
                    print(f"Section 4.1 baseline paragraph misses {b} ({fn})"); n_bad += 1
    for b in sorted(ALL_METHODS):
        if b not in b1:
            print(f"Appendix B.1 misses {b}"); n_bad += 1
    print(f"check_consistency: {len(files)} table files, {n_bad} with violations")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
