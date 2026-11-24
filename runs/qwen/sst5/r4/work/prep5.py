"""Local: steering constants along PREDICTED-class mean offsets (what the readout treats as each class)."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "wsp_work/sst5_r4"; CACHE = ROOT / "wsp_work/sst5_r4_cache"
N_STAT = 2560
MLP_LAYERS = (17, 19, 21, 23)
ex = [json.loads(l) for l in open(ROOT / "data/processed/sst5/train.jsonl")][:3072]
shown = np.array([e["shown_option_ids"] for e in ex])
logp = np.load(WORK / "base_train_logp.npy")
pred_cls = shown[np.arange(len(ex)), logp.argmax(1)][:N_STAT]
H = np.load(CACHE / "base_train_hidden.npy", mmap_mode="r"); nw = np.load(WORK / "norm_weights.npz")
mt, D, SIG = [], [], []
for L in MLP_LAYERS:
    hin = np.asarray(H[:N_STAT, L - 1, :], np.float32); hout = np.asarray(H[:N_STAT, L, :], np.float32)
    x = nw[f"ln2_{L}"].astype(np.float32) * hin / np.sqrt((hin ** 2).mean(1, keepdims=True) + 1e-6)
    mux = x.mean(0); m = mux / (mux @ mux); s = x @ m
    mu = np.stack([hout[pred_cls == c].mean(0) for c in range(5)]); d = mu - hout.mean(0)
    dhat = d / np.linalg.norm(d, axis=1, keepdims=True)
    sig = ((hout - hout.mean(0)) @ dhat.T).std(0)
    print(f"L{L}: s std {s.std():.3f} min {s.min():.3f} |d| {np.round(np.linalg.norm(d,axis=1),1)} sigma {np.round(sig,1)}")
    mt.append(m); D.append(dhat); SIG.append(sig)
np.save(WORK / "psteer_mtilde.npy", np.stack(mt).astype(np.float32))
np.save(WORK / "psteer_dhat.npy", np.stack(D).astype(np.float32))
np.save(WORK / "psteer_sigma.npy", np.stack(SIG).astype(np.float32))
(WORK / "psteer_layers.json").write_text(json.dumps(list(MLP_LAYERS)))
