"""Capitalization sweep of paper/tables/*.tex (user rule): the first letter of every text cell (row labels, headers,
group headers, text-valued cells) and the first word after any colon (cells and captions) must be capitalized.
Exceptions: proper names starting lowercase (gpt-oss, vLLM), math/symbols, numbers. Prints violations; exit 1 if any."""
import glob, os, re, sys
IS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXC = ("gpt-oss", "vLLM", "n/a")
WRAP = re.compile(r"^\s*(\\(cellcolor|rowcolor|textcolor)\{[^}]*\}|\\(textbf|underline|textit|emph|mbox)\{|\\multirow\{[^}]*\}(\[[^\]]*\])?\{[^}]*\}\{|"
                  r"\\multicolumn\{[^}]*\}\{[^}]*\}\{|\\quad|\\qquad|\\cmidrule\S*|\\hline|\$[^$]*\$|[+\-~\s{}()]|\\rightarrow)")


def first_word(c):
    prev = None
    while prev != c:
        prev = c
        c = WRAP.sub("", c, count=1)
    return c


def split_cells(row):
    cells, depth, cur, i = [], 0, "", 0
    while i < len(row):
        ch = row[i]
        if ch == "\\" and i + 1 < len(row):
            cur += row[i:i + 2]; i += 2; continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "&" and depth == 0:
            cells.append(cur); cur = ""
        else:
            cur += ch
        i += 1
    cells.append(cur)
    return cells


def bad(text):
    w = first_word(text)
    if not w or not w[0].isalpha() or w.startswith(EXC):
        return False
    if w.startswith("\\"):  # a macro such as \textcolor{gray}{TBD}, \diva{}, \citep: not a word
        return False
    return w[0].islower()


def after_colons(text):
    out = []
    for m in re.finditer(r"(?<![a-zA-Z]):\s+(\S.*)", text):
        if bad(m.group(1)):
            out.append(m.group(1)[:40])
    return out


viol = []
for f in sorted(glob.glob(f"{IS}/paper/tables/*.tex")):
    s = open(f).read(); name = os.path.basename(f)
    for tab in re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", s, flags=re.S):
        body = re.sub(r"^\\begin\{tabular\}\{[^}]*\}", "", tab)
        for row in re.split(r"\\\\", body):
            row = re.sub(r"\\(toprule|midrule|bottomrule|end\{tabular\})", "", row)
            row = re.sub(r"\\cmidrule(\([^)]*\))?\{[^}]*\}", "", row)
            for c in split_cells(row):
                c = c.strip()
                if c and bad(c):
                    viol.append((name, "cell", c[:60]))
                for a in after_colons(c):
                    viol.append((name, "cell after colon", a))
    for cap in re.findall(r"\\caption\{(.*)\}\s*\n\\label", s):
        if bad(cap):
            viol.append((name, "caption start", cap[:40]))
        for a in after_colons(cap):
            viol.append((name, "caption after colon", a))
for v in viol:
    print(*v, sep=" | ")
print(f"{len(viol)} violations")
sys.exit(1 if viol else 0)
