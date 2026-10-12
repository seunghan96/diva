"""Count rendered caption lines in a compiled PDF and check the user's limits:
one-column figure/table <= 5 lines, two-column table <= 3 lines, two-column figure <= 5 lines.
usage: check_captions.py main_full.pdf [main_8p.pdf ...]. Exit code 1 if any caption exceeds its limit."""
import re, sys
import fitz

bad = 0
for fn in sys.argv[1:]:
    d = fitz.open(fn)
    viol = []; n = 0
    for pno, page in enumerate(d):
        W = page.rect.width
        for b in page.get_text("dict")["blocks"]:
            if "lines" not in b or not b["lines"]:
                continue
            first = "".join(sp["text"] for sp in b["lines"][0]["spans"]).strip()
            m = re.match(r"^(Figure|Table)\s+(\d+):", first)
            if not m:
                continue
            # merge visual lines by y position (spans of one line can be split into several 'lines')
            ys = sorted({round(l["bbox"][1]) for l in b["lines"]})
            rows = []
            for y in ys:
                if not rows or y - rows[-1] > 3:
                    rows.append(y)
            width = b["bbox"][2] - b["bbox"][0]
            wide = width > 0.6 * W
            limit = 3 if (m.group(1) == "Table" and wide) else 5
            n += 1
            if len(rows) > limit:
                viol.append((m.group(1) + " " + m.group(2), pno + 1, len(rows), limit, "two-column" if wide else "one-column"))
    print(f"{fn}: {n} captions, {len(viol)} over the limit")
    for v in viol:
        print("  %s (p.%d): %d lines > %d (%s)" % v)
    bad += len(viol)
sys.exit(1 if bad else 0)
