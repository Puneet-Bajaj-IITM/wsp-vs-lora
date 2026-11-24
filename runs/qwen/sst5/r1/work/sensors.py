"""Local helpers: class-plane sensor directions at the input of a layer's MLP and the exact reader value."""
import numpy as np

SQRT_D = float(np.sqrt(1536))


def plane(H, cls, layer_in):
    """Unit polarity (z1: negative -> positive) and intensity (z2: neutral -> very) directions."""
    h = H[layer_in]
    M = np.stack([h[cls == k].mean(0) for k in range(5)])
    _, _, Vt = np.linalg.svd(M - M.mean(0), full_matrices=False)
    u1, u2 = Vt[0].copy(), Vt[1].copy()
    if (M[4] - M[0]) @ u1 < 0:
        u1 *= -1
    if (M[0] + M[4] - 2 * M[2]) @ u2 < 0:
        u2 *= -1
    return h.mean(0), u1, u2


def centred(mu, u):
    """Direction whose reader value is ~ (u.(h - mu)) * sqrt(d)/|h|: mean removed via the mean direction."""
    return u - (mu @ u) / (mu @ mu) * mu


def read(h, direction):
    return SQRT_D * (h @ direction) / np.linalg.norm(h, axis=1)


def silu(x):
    return x / (1 + np.exp(-x))
