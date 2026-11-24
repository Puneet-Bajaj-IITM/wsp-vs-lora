import sys, json, numpy as np
sys.path.insert(0,'wsp_work/twin_r3')
from make2 import *
st=Stats(); t=st.table
nset=len(t.sets)
# examples per set in train
cnt=np.zeros(nset)
for e in st.train: cnt[t.sets.index(t.col_set[e['column']])]+=1
def per_set(layer,group,kt):
    out=[]
    for s in range(nset):
        tau,keep=st.target([s]); 
        if keep.sum()<2: out.append((np.nan,np.nan,np.nan)); continue
        d=st.centred_keys(layer,group,kt,keep)[keep]; n2=(d**2).sum(1).mean()
        pred=d@((tau[keep,None]*d).sum(0))/n2
        tk=tau[keep]
        # does predicted argmax match the desired top option (by w)?
        out.append((np.corrcoef(pred,tk)[0,1], (pred*tk).sum()/(tk**2).sum(), np.sqrt(n2)))
    return np.array(out)
for (L,g,kt) in [(21,1,1),(21,1,0),(18,0,0),(20,0,0),(20,1,1)]:
    r=per_set(L,g,kt); w=cnt/cnt.sum()
    ok=~np.isnan(r[:,0])
    print(f'L{L} g{g} type{kt}: weighted corr {np.nansum(r[:,0]*w)/w[ok].sum():.3f} slope {np.nansum(r[:,1]*w)/w[ok].sum():.2f} |d| {np.nansum(r[:,2]*w)/w[ok].sum():.2f}')
r=per_set(21,1,1)
for s in np.argsort(-cnt)[:12]: print(int(cnt[s]), t.sets[s][:3], r[s].round(2))
# FC detail
for s in st.set_ids(['Strongly support']):
    tau,keep=st.target([s]); d=st.centred_keys(21,1,1,keep)[keep]; n2=(d**2).sum(1).mean(); pred=d@((tau[keep,None]*d).sum(0))/n2
    for i,p in zip(np.where(keep)[0],pred): print('  ',t.pairs[i][1], 'tau',round(tau[i],2),'pred',round(p,2))
    print(np.round(d@d.T,1))
# gating: xq means
xq=np.load(CACHE/'xq_mean.npy').astype(np.float32); xc=np.load(CACHE/'xq_cnt.npy')
col_set=np.array([t.sets.index(t.col_set[c]) for c in t.cols])
for L in (18,20,21):
    X=xq[L]; print('L',L,'|x|',np.linalg.norm(X,axis=1).mean().round(1))
    for s in np.argsort(-cnt)[:6]:
        m=(col_set==s)&(xc>0); o=(col_set!=s)&(xc>0)
        a=np.average(X[m],axis=0,weights=xc[m]); b=np.average(X[o],axis=0,weights=xc[o]); r=(a-b)/((a-b)**2).sum()
        g=(X-b)@r
        print('  set',t.sets[s][0][:20],'ncols',m.sum(),'|a-b|',np.linalg.norm(a-b).round(2),'gate in-set',g[m].mean().round(2),g[m].min().round(2),'out-set mean',g[o].mean().round(2),'min/max',g[o].min().round(2),g[o].max().round(2))
