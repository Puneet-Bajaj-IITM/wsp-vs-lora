"""Local: build steering constants (class-mean offsets per layer, gate signature) from train[0:2560]."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "wsp_work/sst5_r4"; CACHE = ROOT / "wsp_work/sst5_r4_cache"
N_STAT = 2560
MLP_LAYERS = (11, 13, 15, 17, 19)
ex = [json.loads(l) for l in open(ROOT / "data/processed/sst5/train.jsonl")][:3072]
label = np.array([e["label"] for e in ex]); shown = np.array([e["shown_option_ids"] for e in ex])
true_cls = shown[np.arange(len(ex)), label][:N_STAT]
H = np.load(CACHE / "base_train_hidden.npy", mmap_mode="r")
nw = np.load(WORK / "norm_weights.npz")
mt, D, SIG = [], [], []
for L in MLP_LAYERS:
    hin = np.asarray(H[:N_STAT, L - 1, :], np.float32)      # approx input of MLP L (attn_L ignored)
    hout = np.asarray(H[:N_STAT, L, :], np.float32)
    x = nw[f"ln2_{L}"].astype(np.float32) * hin / np.sqrt((hin ** 2).mean(1, keepdims=True) + 1e-6)
    mux = x.mean(0); m = mux / (mux @ mux); s = x @ m
    mu = np.stack([hout[true_cls == c].mean(0) for c in range(5)]); d = mu - mu.mean(0)
    sig = np.array([((hout - hout.mean(0)) @ (d[c] / np.linalg.norm(d[c]))).std() for c in range(5)])
    print(f"L{L}: s mean {s.mean():.3f} std {s.std():.3f} min {s.min():.3f} |mux| {np.linalg.norm(mux):.2f} |x| {np.linalg.norm(x,axis=1).mean():.2f} "
          f"|d_c| {np.round(np.linalg.norm(d,axis=1),2)} sigma {np.round(sig,2)} |h| {np.linalg.norm(hout,axis=1).mean():.1f}")
    mt.append(m); D.append(d); SIG.append(sig)
np.save(WORK / "steer_mtilde.npy", np.stack(mt).astype(np.float32))
np.save(WORK / "steer_D.npy", np.stack(D).astype(np.float32))
np.save(WORK / "steer_sigma.npy", np.stack(SIG).astype(np.float32))
(WORK / "steer_layers.json").write_text(json.dumps(list(MLP_LAYERS)))
np.save(WORK / "steer_heldmean.npy", np.stack([np.asarray(H[N_STAT:, L, :], np.float32).mean(0) for L in MLP_LAYERS]))
