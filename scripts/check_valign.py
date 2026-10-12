"""Vertical-centering check of every table cell in paper/main_full.pdf (user rule 2026-10-11).

For each table (a stack of horizontal rules sharing the same x-extent above a "Table N:" caption):
  * header band (top rule .. first full-width rule): a header cell whose column has no other header text above or below it
    spans all header rows, so its text block must be centered on the band;
  * body bands (between full-width rules): text lines are grouped into table rows. A row whose label cell has several lines
    (e.g. a method name with its citation underneath) must have its single-line cells centered on the label block, and a
    single-line label next to a multi-line data cell must be centered on that cell.
A cell is flagged when its vertical center is more than TOL pt off the center of its row block.
Usage: python scripts/check_valign.py [paper/main_full.pdf] ; exit code 1 if any flag."""
import re
import sys
import pymupdf

TOL = 1.0
PDF = sys.argv[1] if len(sys.argv) > 1 else "paper/main_full.pdf"


def hrules(page):
    out = []
    for g in page.get_drawings():
        for it in g["items"]:
            if it[0] == "l":
                a, b = it[1], it[2]
                if abs(a.y - b.y) < 0.5 and abs(b.x - a.x) > 8:
                    out.append((a.y, min(a.x, b.x), max(a.x, b.x)))
            elif it[0] == "re":
                r = it[1]
                if r.height < 1.2 and r.width > 8:
                    out.append(((r.y0 + r.y1) / 2, r.x0, r.x1))
    return out


def vrules(page):
    out = []
    for g in page.get_drawings():
        for it in g["items"]:
            if it[0] == "l":
                a, b = it[1], it[2]
                if abs(a.x - b.x) < 0.5 and abs(b.y - a.y) > 2:
                    out.append((a.x, min(a.y, b.y), max(a.y, b.y)))
            elif it[0] == "re":
                r = it[1]
                if r.width < 1.2 and r.height > 2:
                    out.append(((r.x0 + r.x1) / 2, r.y0, r.y1))
    return out


def lines_in(page, x0, x1, y0, y1):
    """text pieces of the region, one per table cell line: each PDF text line is split where the horizontal gap between
    consecutive characters exceeds 3.5 pt (column separation)"""
    out = []
    for b in page.get_text("rawdict")["blocks"]:
        for l in b.get("lines", []):
            for sp in l["spans"]:
                size = sp["size"]; base = sp["origin"][1]
                italic = bool(sp["flags"] & 2)
                cur = []
                for ch in sp["chars"]:
                    if cur and ch["bbox"][0] - cur[-1]["bbox"][2] > 3.5:
                        out.append((cur, size, base, italic)); cur = []
                    cur.append(ch)
                if cur:
                    out.append((cur, size, base, italic))
    res = []
    for chs, size, base, italic in out:
        txt = "".join(c["c"] for c in chs).strip()
        bx0 = chs[0]["bbox"][0]; bx1 = chs[-1]["bbox"][2]
        if not txt or bx1 < x0 - 1 or bx0 > x1 + 1 or base - 0.7 * size < y0 - 0.5 or base > y1 + 0.5:
            continue
        res.append(dict(x0=bx0, x1=bx1, top=base - 0.70 * size, bot=base, txt=txt, italic=italic, size=size))
    if res:  # drop super-/subscripts (e.g. a double dagger): they belong to the text beside them
        big = max(r["size"] for r in res)
        res = [r for r in res if r["size"] > 0.8 * big]
    # merge pieces of the same cell line that were split into several spans (same baseline, touching)
    res.sort(key=lambda r: (round(r["bot"], 1), r["x0"]))
    merged = []
    for r in res:
        m = merged[-1] if merged else None
        if m and abs(m["bot"] - r["bot"]) < 0.6 and r["x0"] - m["x1"] < 1.0:
            m["x1"] = r["x1"]; m["txt"] += r["txt"]; m["top"] = min(m["top"], r["top"]); m["italic"] = m["italic"] and r["italic"]
        else:
            merged.append(dict(r))
    return merged


def tables(page):
    """yield (number, top_rule_y, full_rules, x0, x1) for each table on the page"""
    hr = hrules(page)
    caps = []
    for b in page.get_text("blocks"):
        m = re.match(r"Table (\d+):", b[4].strip())
        if m:
            caps.append((int(m.group(1)), b[1], b[0], b[2]))
    for num, cy, cx0, cx1 in caps:
        # rules above the caption that overlap its x-range; the bottom rule is the closest full-width one
        # only rules below the closest caption above this one (stacked tables of equal width; F6)
        above = [c[1] for c in caps if c[1] < cy - 1 and c[2] < cx1 and c[3] > cx0]
        floor = max(above) if above else -1
        cand = [r for r in hr if floor < r[0] < cy and r[1] < cx1 and r[2] > cx0 and cy - r[0] < 900]
        if not cand:
            continue
        width = max(r[2] - r[1] for r in cand)
        full = sorted([r for r in cand if r[2] - r[1] > 0.9 * width and abs((r[2] - r[1]) - width) < 3], key=lambda r: r[0])
        if not full:
            continue
        bottom = full[-1]
        x0, x1 = bottom[1], bottom[2]
        # walk up while the rules keep the same x-extent and the gaps are small (contiguous table)
        stack = [bottom]
        for r in reversed(full[:-1]):
            if abs(r[1] - x0) < 2 and abs(r[2] - x1) < 2 and stack[-1][0] - r[0] < 400:
                stack.append(r)
            else:
                break
        stack = sorted(stack, key=lambda r: r[0])
        if len(stack) < 2:
            continue
        yield num, [r[0] for r in stack], x0, x1


def chars_in(page, x0, x1, y0, y1):
    out = []
    for b in page.get_text("rawdict")["blocks"]:
        for l in b.get("lines", []):
            for sp in l["spans"]:
                for ch in sp["chars"]:
                    cx = (ch["bbox"][0] + ch["bbox"][2]) / 2; base = ch["origin"][1]
                    if ch["c"].strip() and x0 - 1 < cx < x1 + 1 and y0 - 0.5 < base - 0.7 * sp["size"] and base < y1 + 0.5:
                        out.append((cx, base, sp["size"]))
    if out:
        big = max(o[2] for o in out)
        out = [o for o in out if o[2] > 0.8 * big]
    return out


def check_header(page, num, y0, y1, x0, x1, cols, out):
    """cols: x-intervals of the table columns (label column first). A column whose header holds one text block spans all
    header rows and must be centered on the header band."""
    mid = (y0 + y1) / 2
    cmid = [r for r in hrules(page) if y0 + 0.5 < r[0] < y1 - 0.5 and r[2] - r[1] < 0.9 * (x1 - x0) and r[1] >= x0 - 1 and r[2] <= x1 + 1]
    for a, b in cols:
        if any(min(b, r[2]) - max(a, r[1]) > 0.5 * (b - a) for r in cmid):
            continue  # a group label sits above this column (\cmidrule): the cell spans the lower header rows only
        ch = chars_in(page, a, b, y0, y1)
        if not ch:
            continue
        bases = sorted(set(round(c[1], 1) for c in ch))
        size = max(c[2] for c in ch)
        # stacked lines of one block are one baseline-skip apart (about 1.25 x size); a larger gap separates header levels
        blocks = [[bases[0]]]
        for bl in bases[1:]:
            if bl - blocks[-1][-1] < 1.45 * size:
                blocks[-1].append(bl)
            else:
                blocks.append([bl])
        if len(blocks) != 1:
            continue
        top = blocks[0][0] - 0.7 * size; bot = blocks[0][-1]
        off = (top + bot) / 2 - mid
        if abs(off) > TOL + 0.2:  # strict (user 2026-10-11: Table 4 header sat 2 pt low and passed the old 2.5 pt tolerance)
            txt = "".join(c["c"] for c in [])  # text for the report
            words = [w[4] for w in page.get_text("words") if a - 1 < (w[0] + w[2]) / 2 < b + 1 and y0 < w[3] <= y1 + 1]
            out.append((num, page.number + 1, "header", " ".join(words)[:40], round(off, 1)))


def check_body(page, num, y0, y1, x0, x1, vx, out):
    L = lines_in(page, x0, x1, y0, y1)
    if not L:
        return
    lab0 = sorted([l for l in L if l["x0"] < vx - 1], key=lambda l: (round(l["bot"], 1), l["x0"]))
    lab = []  # one piece per label line: merge the words of a label cell split by math spacing
    for l in lab0:
        if lab and abs(lab[-1]["bot"] - l["bot"]) < 0.6 and l["x0"] - lab[-1]["x1"] < 12:
            lab[-1] = dict(lab[-1], x1=l["x1"], txt=lab[-1]["txt"] + " " + l["txt"], italic=lab[-1]["italic"] and l["italic"])
        else:
            lab.append(dict(l))
    lab.sort(key=lambda l: l["top"])
    dat = sorted([l for l in L if l["x0"] >= vx - 1], key=lambda l: l["top"])
    cen = lambda l: (l["top"] + l["bot"]) / 2
    # data rows: data lines clustered by their vertical center
    rows = []
    for d in dat:
        if rows and abs(cen(d) - rows[-1][0]) < 1.5:
            rows[-1][1].append(d)
        else:
            rows.append([cen(d), [d]])
    if not rows:
        return
    near = lambda l: min(range(len(rows)), key=lambda k: abs(cen(l) - rows[k][0]))
    # label sub-columns (e.g. Family | Size, Env. | Family | Size): cluster label lines by their left edge
    cols = []
    for l in lab:
        if l["italic"]:
            continue
        for c in cols:
            if abs(c[0] - l["x0"]) < 8:  # indented continuation lines belong to the same label cell
                c[1].append(l); break
        else:
            cols.append([l["x0"], [l]])
    for _, ls in cols:
        groups = {}
        for l in ls:
            groups.setdefault(near(l), []).append(l)
        if len(groups) < len(rows):
            continue  # a \multirow column spanning several rows (family names, environment names): centered by construction
        for k, blk in groups.items():
            top = min(l["top"] for l in blk); bot = max(l["bot"] for l in blk); c = (top + bot) / 2
            rc, ds = rows[k]
            if abs(rc - c) > TOL:
                out.append((num, page.number + 1, " / ".join(l["txt"] for l in blk)[:40], ds[0]["txt"][:20], round(rc - c, 1)))


def main():
    doc = pymupdf.open(PDF)
    out, n_tab = [], 0
    for page in doc:
        vr = vrules(page)
        for num, ys, x0, x1 in tables(page):
            n_tab += 1
            # the label rule: leftmost vertical rule covering a good part of the table body
            cover = {}
            for v in vr:
                if x0 + 2 < v[0] < x1 - 2 and v[1] < ys[-1] and v[2] > ys[0]:
                    k = round(v[0])
                    cover[k] = cover.get(k, 0) + min(v[2], ys[-1]) - max(v[1], ys[0])
            vxs = sorted(k for k, c in cover.items() if c > 0.3 * (ys[-1] - ys[0]))
            vx = vxs[0] if vxs else x0 + 0.3 * (x1 - x0)
            # column x-intervals from the data cells of the body (label column = left of the label rule)
            body = lines_in(page, x0, x1, ys[1], ys[-1])
            xs = sorted(((d["x0"] + d["x1"]) / 2, d["x0"], d["x1"]) for d in body if d["x0"] >= vx - 1)
            cl = []
            for c, a_, b_ in xs:
                if cl and c - cl[-1][2] < 4:
                    cl[-1][2] = max(cl[-1][2], c); cl[-1][0] = min(cl[-1][0], a_); cl[-1][1] = max(cl[-1][1], b_)
                else:
                    cl.append([a_, b_, c])
            cols = [(x0, vx - 1)] + [(c[0] - 2, c[1] + 2) for c in cl]
            check_header(page, num, ys[0], ys[1], x0, x1, cols, out)
            for a, b in zip(ys[1:-1], ys[2:]):
                check_body(page, num, a, b, x0, x1, vx, out)
    for f in out:
        print(f"Table {f[0]} (p.{f[1]}): row '{f[2]}' cell '{f[3]}' off {f[4]:+.1f} pt")
    print(f"check_valign: {n_tab} tables, {len(out)} flags")
    sys.exit(1 if out else 0)


if __name__ == "__main__":
    main()
