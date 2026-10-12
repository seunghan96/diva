"""Pending jobs per queue file over ALL models (progress.py skips 70B/72B/27B). Prints 'file done/total' for unfinished files only."""
import glob, gzip, os
for f in sorted(glob.glob('jobs/*.tsv')):
    jd = jt = 0
    for line in open(f):
        if not line.strip() or line.startswith('#'):
            continue
        jid, model, n, out, cmd = line.rstrip('\n').split('\t', 4)
        try:
            h = sum(1 for _ in gzip.open(out, 'rt')) if os.path.exists(out) else 0
        except Exception:
            h = 0
        jt += 1; jd += h >= int(n)
    if jd < jt:
        print(f"{os.path.basename(f)} {jd}/{jt}")
