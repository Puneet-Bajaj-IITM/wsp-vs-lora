import numpy as np
theta = 1e6 ** (-np.arange(64) / 64)
D = np.arange(0, 121)
for P in (12, 20, 32, 64):
    # features: cos(theta_i (D-2)) and sin(theta_i (D-2)) -> kernel K(D)=sum a_i cos + b_i sin
    F = np.concatenate([np.cos(np.outer(D - 2, theta[:P])), np.sin(np.outer(D - 2, theta[:P]))], 1)
    tgt = F[2]; off = np.delete(F, 2, 0)
    for lam in (1e-3, 1e-2, 1e-1):
        C = off.T @ off + lam * np.trace(off.T @ off) / F.shape[1] * np.eye(F.shape[1])
        q = np.linalg.solve(C, tgt); q /= q @ tgt
        K = F @ q
        print(P, lam, "max sidelobe %.3f at D=%d  |q| %.2f  near:" % (np.delete(K, 2).max(), D[np.argmax(np.where(D == 2, -9, K))], np.linalg.norm(q)), K[:8].round(2))
