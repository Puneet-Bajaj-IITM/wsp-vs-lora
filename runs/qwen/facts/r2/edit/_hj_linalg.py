"""Small closed-form linear-algebra helpers (float64, CPU)."""
import torch

DT = torch.float64


def rms_norm(x, gain, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gain


def right_inverse(rows):
    """rows (n, m), n <= m, full row rank -> (m, n) matrix P with rows @ P = I."""
    return torch.linalg.pinv(rows)


def map_rows(sources, targets):
    """Matrix O (d, m) with O @ sources[i] = targets[i]; sources (n, m) independent, targets (n, d)."""
    return targets.T @ torch.linalg.pinv(sources.T)


def orthonormal_rows(count, dim, avoid, seed):
    """`count` orthonormal vectors, all orthogonal to the rows of `avoid`."""
    gen = torch.Generator().manual_seed(seed)
    raw = torch.randn(count, dim, generator=gen, dtype=DT)
    basis, _ = torch.linalg.qr(avoid.T)
    raw = raw - (raw @ basis) @ basis.T
    q, _ = torch.linalg.qr(raw.T)
    return q.T


def condition(rows):
    s = torch.linalg.svdvals(rows)
    return float(s[0] / s[-1])
