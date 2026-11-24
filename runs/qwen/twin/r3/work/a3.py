import sys, json, collections, numpy as np
sys.path.insert(0,'wsp_work/twin_r3')
from common import *
C='wsp_work/twin_r3_cache/out1/'
tr=load('train'); dv=load('dev'); T=Table(tr)
idx=json.load(open(C+'idx.json'))
dl=np.load(C+'diag_logp.npy'); hl=np.load(C+'held_logp.npy'); vl=np.load(C+'dev_logp.npy')
dex=[tr[i] for i in idx['diag']]; hex_=[tr[i] for i in idx['held']]
def lab(ex): return np.array([e['label'] for e in ex])
def stats(lp,ex,name):
    y=lab(ex); p=lp.argmax(1); byb=collections.defaultdict(list)
    for e,ok in zip(ex,p==y): byb[e['block'][:10]].append(ok)
    print(name,'acc',round((p==y).mean(),4),'nll',round(-lp[np.arange(len(y)),y].mean(),4),{b:round(np.mean(v),3) for b,v in byb.items() if len(v)>=40})
stats(dl,dex,'diag');stats(vl,dv,'dev')
# calibration from diag (train): letter bias = -mean logp ; temperature grid
def cal(lp,b,t):
    z=(lp+b)/t; return z-np.log(np.exp(z).sum(1,keepdims=True))
b=-dl.mean(0); b-=b.mean(); print('bias',b.round(3))
for t in (1,2,3,4,6):
    stats(cal(dl,b,t),dex,f'diag bias T{t}'); stats(cal(vl,b,t),dv,f'dev  bias T{t}')
# how does the model's own preference align with prior? P(model picks modal option)
def prior_scores(ex): 
    return np.array([[T.w[p] if p>=0 else 0 for p in T.pids(e)] for e in ex])
ws=prior_scores(dv); y=lab(dv)
print('prior(set-level) dev acc',(ws.argmax(1)==y).mean(), 'model agrees with prior argmax',(vl.argmax(1)==ws.argmax(1)).mean())
for lam in (0.5,1,2,4):
    for t in (2,4):
        z=cal(vl,b,t)+lam*ws; z-=np.log(np.exp(z).sum(1,keepdims=True)); stats(z,dv,f'dev combo lam{lam} T{t}')
