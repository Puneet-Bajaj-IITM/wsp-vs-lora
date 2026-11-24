import sys, json, collections, numpy as np
sys.path.insert(0,'wsp_work/twin_r3')
from common import *
C='wsp_work/twin_r3_cache/out1/'
tr=load('train'); T=Table(tr)
idx=json.load(open(C+'idx.json')); dl=np.load(C+'diag_logp.npy'); ex=[tr[i] for i in idx['diag']]
b=-dl.mean(0); b-=b.mean()
pick=collections.Counter(); truth=collections.Counter(); shown=collections.Counter(); lp=collections.defaultdict(list)
for e,l in zip(ex,dl):
    o=opts_of(e); s=len(T.col_set[e['column']]); key='LIK' if 'Agree strongly' in o else ('FC' if 'Strongly support' in o else None)
    if key is None: continue
    z=l+b
    pick[(key,o[int(z.argmax())])]+=1; truth[(key,o[e['label']])]+=1
    for t,v in zip(o,z-z.mean()): lp[(key,t)].append(v)
for k in sorted(lp): print(k,'model picks',pick[k],'truth',truth[k],'mean centred logit',round(np.mean(lp[k]),2), 'sd',round(np.std(lp[k]),2))
# vocab: model's correctness on modal answer
c=0;n=0
for e,l in zip(ex,dl):
    if e['block']=='Cognitive tests':
        ids=T.pids(e); w=[T.w[i] for i in ids]; n+=1; c+= int(np.argmax(l+b)==int(np.argmax(w)))
print('cog: model picks set-modal option',c/n,n)
# attention analysis: which heads' option attention (content types) tracks logits, and attention to types
att=np.load(C+'diag_att.npy').astype(np.float32)[...,1:].reshape(len(ex),28,12,5,5)
pred=dl.argmax(1)
tot=att.sum(-1) # n,L,H,5 opt
for name,sl in [('letter',slice(0,1)),('dot',slice(1,2)),('content',slice(2,3)),('last',slice(3,4)),('end',slice(4,5))]:
    m=att[...,sl].sum(-1); ag=(m.argmax(-1)==pred[:,None,None]).mean(0); ms=m.sum(-1).mean(0)
    top=np.argsort(-(ag*(ms>0.05)),axis=None)[:8]
    print(name,[(int(i//12),int(i%12),round(float(ag.flat[i]),2),round(float(ms.flat[i]),2)) for i in top])
dla=np.load(C+'diag_dla.npy').astype(np.float32) # n,L,H,5
d=dla-dla.mean(-1,keepdims=True)
sel=d[np.arange(len(ex)),:,:,pred].mean(0)  # mean DLA toward predicted letter
top=np.argsort(-sel,axis=None)[:12]; print('DLA to pred',[(int(i//12),int(i%12),round(float(sel.flat[i]),2)) for i in top])
print('layer total DLA to pred',sel.sum(1).round(1))
