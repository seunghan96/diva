"""Check that every figure/table/algorithm label used in a paper is referenced in the body text (outside captions).
usage: check_refs.py main_full.tex [main_8p.tex ...]  (run from paper/). Exit code 1 if any label is unreferenced."""
import re, sys, os


def expand(path, seen=None):
    seen = seen or set()
    if not path.endswith(".tex"):
        path += ".tex"
    if not os.path.exists(path) or path in seen:
        return ""
    seen.add(path)
    s = open(path).read()
    s = re.sub(r"(?<!\\)%.*", "", s)
    return re.sub(r"\\input\{([^}]+)\}", lambda m: expand(m.group(1), seen), s)


def strip_captions(s):
    out, i = [], 0
    while True:
        j = s.find("\\caption{", i)
        if j < 0:
            out.append(s[i:]); break
        out.append(s[i:j]); k = j + len("\\caption{"); depth = 1
        while depth and k < len(s):
            depth += {"{": 1, "}": -1}.get(s[k], 0); k += 1
        i = k
    return "".join(out)


bad = 0
for main in sys.argv[1:]:
    doc = expand(main)
    labels = re.findall(r"\\label\{((?:fig|tab|alg):[^}]+)\}", doc)
    body = strip_captions(doc)
    refs = set()
    for m in re.finditer(r"\\(?:ref|autoref|cref|Cref)\{([^}]+)\}", body):
        refs.update(x.strip() for x in m.group(1).split(","))
    miss = [l for l in labels if l not in refs]
    print(f"{main}: {len(labels)} float labels, {len(miss)} unreferenced: {miss}")
    bad += len(miss)
sys.exit(1 if bad else 0)
