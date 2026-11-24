"""Local CPU analysis of job-1 arrays (no model): calibration, logit lens, class-mean readouts."""
import json, sys, collections
from pathlib import Path
import numpy as np
A = Path(sys.argv[1] if len(sys.argv) > 1 else "wsp_work/twin_o1_cache/arrays1")
dv = [json.loads(l) for l in open("data/processed/dev.jsonl")]
tr = [json.loads(l) for l in open("data/processed/train.jsonl")]
L = np.load(A/"letter_emb.npy"); g = np.load(A/"norm_w.npy")
def ld(tag, k): return np.load(A/f"{tag}_{k}.npy")
dl, dy = ld("dev","logp"), ld("dev","labels"); tl, ty = ld("train","logp"), ld("train","labels")
tidx = ld("train","idx"); trs = [tr[i] for i in tidx]
def lsm(z): z = z - z.max(1, keepdims=True); return z - np.log(np.exp(z).sum(1, keepdims=True))
def met(lp, y): return round(float((lp.argmax(1)==y).mean()),4), round(float(-lp[np.arange(len(y)),y].mean()),4)
print("base dev", met(dl,dy), "train", met(tl,ty))
print("pred hist dev", np.bincount(dl.argmax(1), minlength=5)/len(dy))
blocks = collections.Counter(e["block"] for e in dv)
for b in [b for b,_ in blocks.most_common(4)]:
    m = np.array([e["block"]==b for e in dv]); print("  ", b, m.sum(), met(dl[m], dy[m]))
# temperature + letter bias from train
best=None
for T in [1,1.5,2,2.5,3,4,5,6,8]:
    z = tl/T; bias = -np.log(np.exp(lsm(z)).mean(0)); 
    r = met(lsm(z+bias), ty)
    if best is None or r[1] < best[1][1]: best = (T, r, bias)
    print("T",T,"train",met(lsm(z),ty),"with bias",r)
T, _, bias = best
print("chosen T", T, "bias", bias, "dev", met(lsm(dl/T), dy), "dev+bias", met(lsm(dl/T+bias), dy))
# logit lens
for lay in (10,16,20,23,25,26,27):
    for tag, y in (("dev", dy),):
        h = ld(tag, f"h{lay}").astype(np.float32)
        n = h/np.sqrt((h**2).mean(1,keepdims=True)+1e-6)*g
        print("lens", lay, met(lsm(n@L.T), y))
# class-mean readout
for lay in (16,20,23,25,26,27,-1):
    Htr = ld("train", f"h{lay}").astype(np.float32); Hdv = ld("dev", f"h{lay}").astype(np.float32)
    if lay != -1:
        Htr = Htr/np.sqrt((Htr**2).mean(1,keepdims=True)); Hdv = Hdv/np.sqrt((Hdv**2).mean(1,keepdims=True))
    mu = Htr.mean(0); C = np.stack([Htr[ty==k].mean(0)-mu for k in range(5)])
    s_tr = (Htr-mu)@C.T; s_dv = (Hdv-mu)@C.T
    res = []
    for a in [0, 0.5, 1, 2, 4, 8, 16]:
        sc = s_tr.std()
        res.append((a, met(lsm(dl/T + bias + a*s_dv/sc), dy)))
    print("centroid", lay, "alone", met(lsm(s_dv/s_dv.std()*3), dy), res)
