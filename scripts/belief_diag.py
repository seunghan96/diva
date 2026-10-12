import sys,os,json,gzip,numpy as np; sys.path.insert(0,os.path.dirname(__file__))
from dcore import bayes
env={json.loads(l)['id']:json.loads(l) for l in gzip.open(sys.argv[1],'rt')}
for f in sys.argv[2:]:
    B0=[];P=[];D=[];DE=[];ASC=[];DESC=[]
    for l in gzip.open(f,'rt'):
        pr=json.loads(l); sc=env[pr['id']]; b=sc['exact']['posterior']
        if not pr['b0']: continue
        B0.append(pr['b0']['b']);P.append(b);ASC.append(pr['b0']['b_asc']);DESC.append(pr['b0']['b_desc'])
        for c in sc['checks']:
            bv=pr['bv'][c['label']]
            if bv['1'] and bv['0']:
                D.append(bv['1']['b']-bv['0']['b']); DE.append(bayes(b,c['s1'],c['s0'],1)-bayes(b,c['s1'],c['s0'],0))
    B0=np.array(B0);P=np.array(P);D=np.array(D);DE=np.array(DE)
    print(os.path.basename(f), 'corr(b0,post)=%.2f mean|b0-post|=%.3f asc-desc=%.3f  corr(dLLM,dExact)=%.2f slope=%.2f mean|d|LLM=%.3f exact=%.3f'%(np.corrcoef(B0,P)[0,1],np.abs(B0-P).mean(),np.mean(np.array(ASC)-np.array(DESC)),np.corrcoef(D,DE)[0,1],np.polyfit(DE,D,1)[0],np.abs(D).mean(),np.abs(DE).mean()))
