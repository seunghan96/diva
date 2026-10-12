"""Flag paragraphs and captions of paper/main_full.pdf whose last line is short (user rule 2026-10-11).

Body text: the lines of each column (review line numbers removed) are split into paragraphs at a first-line indent, a
vertical gap or a heading; a paragraph is flagged when its last line fills less than FILL of the column width.
Captions ("Table N:" / "Figure N:" blocks): flagged when the last line fills less than FILL of the caption width.
Skipped: paragraphs or captions with red text ([TBD] placeholders), itemize items, equations (indented, centered lines),
table and figure text (smaller fonts), headings and the bibliography.
Usage: python scripts/check_lastline.py [pdf] [first_page] [last_page]"""
import re
import sys
import pymupdf

FILL = 0.75
PDF = sys.argv[1] if len(sys.argv) > 1 else "paper/main_full.pdf"


def red(span):
    c = span["color"]
    return ((c >> 16) & 255) > 180 and ((c >> 8) & 255) < 80 and (c & 255) < 80


def main():
    doc = pymupdf.open(PDF)
    p0 = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    p1 = min(int(sys.argv[3]), doc.page_count) if len(sys.argv) > 3 else doc.page_count
    flags = []
    in_refs = False
    for pno in range(p0 - 1, p1):
        page = doc[pno]
        W = page.rect.width
        D = page.get_text("dict")["blocks"]
        # ---- captions
        for b in D:
            ls = [l for l in b.get("lines", []) if "".join(s["text"] for s in l["spans"]).strip()]
            if not ls:
                continue
            t0 = "".join(s["text"] for s in ls[0]["spans"]).strip()
            if not re.match(r"(Table|Figure|Algorithm) \d+:", t0) or len(ls) < 2:
                continue
            if any(red(s) for l in ls for s in l["spans"]):
                continue
            x0 = min(l["bbox"][0] for l in ls); x1 = max(l["bbox"][2] for l in ls)
            fill = (ls[-1]["bbox"][2] - x0) / (x1 - x0)
            if fill < FILL:
                flags.append((pno + 1, "caption", fill, t0[:50], "".join(s["text"] for s in ls[-1]["spans"]).strip()))
        # ---- body paragraphs
        L = []
        for b in D:
            for l in b.get("lines", []):
                spans = [s for s in l["spans"] if s["text"].strip()]
                if not spans:
                    continue
                txt = "".join(s["text"] for s in l["spans"]).strip()
                if re.fullmatch(r"\d{1,4}", txt):
                    continue  # review line numbers / page number
                size = max(s["size"] for s in spans)
                L.append(dict(x0=l["bbox"][0], x1=l["bbox"][2], y=l["spans"][0]["origin"][1], size=size, txt=txt,
                              red=any(red(s) for s in spans), bold=all(s["flags"] & 16 for s in spans)))
        if any(l["txt"].startswith("References") and l["bold"] for l in L):
            in_refs = True
        for left in (True, False):
            col = sorted([l for l in L if (l["x0"] < W / 2 - 5) == left], key=lambda l: l["y"])
            body = [l for l in col if 10.4 < l["size"] < 11.5]
            if not body:
                continue
            cx0 = min(l["x0"] for l in body); cw = max(l["x1"] for l in body) - cx0
            if cw < 150:
                continue
            paras, cur = [], []
            for k, l in enumerate(body):
                if cur:
                    prev = cur[-1]
                    indent = 7 < l["x0"] - cx0 < 16
                    gap = l["y"] - prev["y"] > 1.6 * l["size"]
                    if indent or gap or l["txt"].startswith("•") or prev["bold"] and len(prev["txt"]) < 60 and prev["x1"] - cx0 < 0.6 * cw:
                        paras.append(cur); cur = []
                cur.append(l)
            if cur:
                paras.append(cur)
            for P in paras:
                last = P[-1]
                if len(P) < 2 or any(l["red"] for l in P) or P[0]["txt"].startswith("•") or (in_refs and pno > 7 and False):
                    continue
                if last["x0"] - cx0 > 16 or last["bold"]:
                    continue  # equation or heading
                if in_refs and pno < 12:
                    continue
                fill = (last["x1"] - cx0) / cw
                if fill < FILL:
                    flags.append((pno + 1, "paragraph", fill, P[0]["txt"][:50], last["txt"][-50:]))
    for f in flags:
        print(f"p.{f[0]} {f[1]} fill={f[2]:.2f} | {f[3]} | last: {f[4]}")
    print(f"check_lastline: {len(flags)} short last lines")


if __name__ == "__main__":
    main()
