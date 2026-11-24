import numpy as np
W = np.load("wsp_work/facts_r1/job1/weights.npz")
f = {k: W[k].astype(np.float64) for k in W.files}
for k, v in f.items(): print(k, v.shape, float(np.abs(v).mean()), float(np.abs(v).max()))
E = f["emb_vocab"]; print("vocab emb norms", np.percentile(np.linalg.norm(E, axis=1), [0, 10, 50, 90, 100]))
print("all emb norms", np.percentile(f["emb_norm_all"], [0, 10, 50, 90, 100]), "mean vec norm", np.linalg.norm(f["emb_mean"]))
for n in ("ln1", "ln2"):
    print(n, np.percentile(f[n], [0, 1, 10, 50, 90, 99, 100]), "min abs", np.abs(f[n]).min())
H, hd = 12, 128
O = f["o_w"]
print("o head col norms", [round(float(np.linalg.norm(O[:, h*hd:(h+1)*hd])), 2) for h in range(H)])
print("q head row norms", [round(float(np.linalg.norm(f["q_w"][h*hd:(h+1)*hd])), 2) for h in range(H)])
print("q bias head norms", [round(float(np.linalg.norm(f["q_b"][h*hd:(h+1)*hd])), 2) for h in range(H)])
for g in range(2):
    s = np.linalg.svd(f["v_w"][g*hd:(g+1)*hd], compute_uv=False)
    print("v group", g, "sv", np.round(s[[0, 10, 40, 64, 80, 96, 112, 120, 127]], 3), "bias", np.linalg.norm(f["v_b"][g*hd:(g+1)*hd]))
    s = np.linalg.svd(f["k_w"][g*hd:(g+1)*hd], compute_uv=False)
    print("k group", g, "sv", np.round(s[[0, 10, 40, 64, 96, 127]], 3), "bias", np.linalg.norm(f["k_b"][g*hd:(g+1)*hd]))
    print("k rows 63,127 norms", np.linalg.norm(f["k_w"][g*hd+63]), np.linalg.norm(f["k_w"][g*hd+127]), "typ", np.median(np.linalg.norm(f["k_w"][g*hd:(g+1)*hd], axis=1)))
imp = f["gate_norm"] * f["up_norm"] * f["down_norm"]
print("neuron importance pct", np.percentile(imp, [0, 1, 5, 10, 50, 90, 100]))
print("down norm pct", np.percentile(f["down_norm"], [0, 5, 50, 100]))
