"""Crop every figure the paper includes to its content (no white margin on any side), as the user requires (2026-10-11).
Reads the graphics list from paper/main_full.fls; for each PDF sets the page's MediaBox/CropBox to the bounding box of non-white pixels
(rendered at 300 dpi). Usage: crop_figs.py [--check]  (--check only reports margins)."""
import sys, os, fitz, numpy as np
P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper")
TOL = 0.25  # pt


def crop_pdf(path, check=False):
    """Set the first page's MediaBox to the bounding box of its non-white pixels; returns the margins (L, T, R, B) in pt before cropping."""
    doc = fitz.open(path); pg = doc[0]; r = pg.rect; z = 300 / 72
    pix = pg.get_pixmap(matrix=fitz.Matrix(z, z), alpha=False)
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]
    ys, xs = np.where((a < 250).any(axis=2))
    x0, x1, y0, y1 = xs.min() / z, (xs.max() + 1) / z, ys.min() / z, (ys.max() + 1) / z
    m = (x0, y0, r.width - x1, r.height - y1)
    if max(m) > TOL and not check:
        out = fitz.open(); out.insert_pdf(doc); q = out[0]; mb = q.mediabox  # page coords (top-left) -> MediaBox coords (bottom-left)
        q.set_mediabox(fitz.Rect(mb.x0 + x0, mb.y1 - y1, mb.x0 + x1, mb.y1 - y0))
        doc.close(); out.save(path + ".tmp", garbage=3, deflate=True); out.close(); os.replace(path + ".tmp", path)
    else:
        doc.close()
    return m


if __name__ == "__main__":
    check = "--check" in sys.argv
    files = sorted({l.split(" ", 1)[1].strip() for l in open(f"{P}/main_full.fls") if l.startswith("INPUT ./figs/") and l.strip().endswith((".pdf", ".png"))})
    bad = 0
    for f in files:
        if f.endswith(".png"):
            print("png (not handled):", f); continue
        m = crop_pdf(os.path.join(P, f), check)
        flag = max(m) > TOL; bad += flag
        print(f"{f:32s} margins L/T/R/B pt: " + " ".join(f"{v:5.1f}" for v in m) + ("  <- margin" if flag else ""))
    print(("with margin " if check else "cropped ") + str(bad))
