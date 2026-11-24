"""Local: choose the final-norm temperature from train log-probs of the chosen config (letter logits scale exactly)."""
import numpy as np

from make_stats import W, load

_, base, shown, cls, _ = load()
o = np.load(W / "job6_out.npz")
logp = np.concatenate([o["combined"], o["final_block1"]])
lab = np.array([list(s).index(c) for s, c in zip(shown[:1024], cls[:1024])])
print("acc", (logp.argmax(1) == lab).mean(), "base acc same blocks", (base[:1024].argmax(1) == lab).mean())
for T in (1, 2, 3, 4, 5, 6, 7, 8, 10):
    q = logp / T
    q -= np.log(np.exp(q).sum(1, keepdims=True))
    halves = [-q[s][np.arange(512), lab[s]].mean() for s in (slice(0, 512), slice(512, 1024))]
    print("T", T, "nll", round(float(-q[np.arange(1024), lab].mean()), 4), "per block", np.round(halves, 4))
