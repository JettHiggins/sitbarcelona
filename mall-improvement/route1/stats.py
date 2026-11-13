"""The diagnostics' summary statistics on a layer-0 game (metres), to compare with the target report."""
import numpy as np
FPS = 25

def q(x, qs=(25, 50, 75)):
    x = x[np.isfinite(x)]
    return " ".join("{:.2f}".format(v) for v in np.percentile(x, qs)) if len(x) else "-"

def summary(B, P, team, graded, pitch):
    seen = np.isfinite(B[:, 0])
    inside = seen & (B[:, 0] > 1) & (B[:, 0] < pitch[0] - 1) & (B[:, 1] > 1) & (B[:, 1] < pitch[1] - 1)
    vb = np.full_like(B, np.nan); vb[1:-1] = (B[2:] - B[:-2]) * (FPS / 2)
    sb = np.linalg.norm(vb, axis=1)
    d = np.linalg.norm(P - B[:, None, :], axis=2); d = np.where(np.isnan(d), np.inf, d)
    k = np.argmin(d, axis=1); d = d[np.arange(len(d)), k]
    base = graded & inside & np.isfinite(d)
    out = {"seen": float(seen[graded].mean()),
           "players": float(np.isfinite(P[graded][:, :, 0]).sum(1).mean() / 2)}
    for lo, hi in ((0.5, 2), (2, 6), (6, 12), (12, 40)):
        out["d@{}-{}".format(lo, hi)] = q(d[base & (sb >= lo) & (sb < hi)])
    out["d all"] = q(d[base])
    out["<1/1.5/2.5"] = " ".join("{:.3f}".format(np.mean(d[base] < r)) for r in (1, 1.5, 2.5))
    slow = base & (sb < 2) & (d < 3); idx = np.flatnonzero(slow)
    off = B[idx] - P[idx, k[idx]]
    out["|off| med"] = float(np.median(np.linalg.norm(off, axis=1)))
    out["off mean"] = "{:+.2f} {:+.2f}".format(*off.mean(0))
    acc = np.linalg.norm(B[2:] - 2 * B[1:-1] + B[:-2], axis=1) * FPS * FPS
    out["acc"] = q(acc[graded[1:-1]], (25, 50, 75, 90))
    out["speed"] = q(sb[graded], (25, 50, 75, 90))
    miss = (~seen) & graded; e = np.diff(np.r_[0, miss.astype(int), 0]); L = np.flatnonzero(e == -1) - np.flatnonzero(e == 1)
    out["gaps"] = "{} len {} | share >125: {:.3f}".format(len(L), q(L.astype(float)), L[L > 125].sum() / max(1, L.sum()))
    return out
