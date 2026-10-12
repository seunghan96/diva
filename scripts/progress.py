import glob,gzip,os,sys,collections
want=sys.argv[1:] or None
for f in sorted(glob.glob('jobs/*.tsv')):
    k=os.path.basename(f)
    if want and not any(w in k for w in want): continue
    need=have=0; jd=jt=0
    for line in open(f):
        if not line.strip(): continue
        jid,model,n,out,cmd=line.rstrip('\n').split('\t',4); n=int(n)
        if any(x in model for x in ['70b','72B','27b']): continue
        try: h=sum(1 for _ in gzip.open(out,'rt')) if os.path.exists(out) else 0
        except Exception: h=0
        need+=n; have+=min(h,n); jt+=1; jd+=h>=n
    print(f"{k:24s} {jd}/{jt} jobs {100*have/max(need,1):.0f}%")
