import sys, json, collections, numpy as np
sys.path.insert(0,'wsp_work/twin_r3')
from common import *
import build
C='wsp_work/twin_r3_cache/out1/'
tr=load('train'); T=Table(tr)
rep=json.load(open(C+'report.json')); inv=np.array(rep['meta']['inv_freq']); lf=build.low_freq_mask(inv); print('lf dims',lf.sum())
km=np.load(C+'key_mean.npy').astype(np.float32); kc=np.load(C+'key_cnt.npy')
ok=kc>=3
for (L,g,ty) in [(21,1,1),(21,1,0),(18,0,0),(20,0,0),(20,1,1),(14,0,1),(10,0,1)]:
    K=km[L,:,ty,g*128:(g+1)*128]
    Kl=K*lf
    grand=Kl[ok].mean(0)
    setm=np.array([Kl[(T.set_of_pid==s)&ok].mean(0) for s in range(len(T.sets)) if ((T.set_of_pid==s)&ok).sum()>=2])
    d=np.zeros_like(Kl)
    for s in range(len(T.sets)):
        m=(T.set_of_pid==s)&ok
        if m.sum()>=2: d[m]=Kl[m]-Kl[m].mean(0)
    print(f'L{L} g{g} type{ty}: |K| full {np.linalg.norm(K[ok],axis=1).mean():.1f} lf {np.linalg.norm(Kl[ok],axis=1).mean():.1f} | between-set dev {np.linalg.norm(setm-grand,axis=1).mean():.1f} | within-set d {np.linalg.norm(d[ok],axis=1).mean():.2f}')
# Likert set detail at L21 g1 end
lik=[i for i,(s,t) in enumerate(T.pairs) if t=='Agree a little' and 'Agree strongly' in s][0]
s=T.set_of_pid[lik]; m=np.where(T.set_of_pid==s)[0]
for (L,g,ty) in [(21,1,1),(21,1,0),(18,0,0)]:
    K=km[L,:,ty,g*128:(g+1)*128]*lf; d=K[m]-K[m].mean(0)
    G=d@d.T
    print(L,g,ty,[T.pairs[i][1] for i in m]); print(np.round(G,1)); print('w',T.w[m].round(2),'cnt',kc[m])
