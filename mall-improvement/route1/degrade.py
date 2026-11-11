"""Make an outside game's tracking look like the challenge match's, using only numbers measured
on the match's automatic tracking (tracking_diagnostics_report.txt), never its labels.

Measured on the match                          how it is imitated here
  ball seen on 59% of graded frames, 99.7% of   the ball is hidden through each stoppage of the
  the missing frames in gaps longer than 5 s,   outside game (official BallStatus), from shortly
  ~150 gaps per match                           after play stops to shortly before it restarts
  ball sits further from the players            a slow random drift is added to the ball track
  (nearest player 1.04 m at 0.5-2 m/s,           (Gaussian, per axis, at knots every KNOT frames,
  against ~0.6 m in the outside games)          cubic interpolation between them)
  ball - player offset: dy -0.19 m and           the same offset and stretch are applied
  dx = -0.985 * (x - 0.5)
  10.2-10.3 players seen per team                players drop out now and then

The stoppage timing uses the outside game's labels, which is how the match's own tracking behaves
(the provider drops the ball while play is stopped); it uses nothing from the match's labels.
"""
import numpy as np
from scipy.interpolate import CubicSpline

DEFAULT = dict(sigma=0.45, knot=40, tail=0.0, env_knot=250, smooth=0.0, hide_min=100, lead=(0, 50), trail=(0, 40), keep=0.08,
               dy=-0.19, stretch=-0.985, on_mean_s=100.0, off_median_s=4.0, live_gap_rate=0.12)


def drift(n, sigma, knot, rng):
    t = np.arange(0, n + 2 * knot, knot)
    return np.column_stack([CubicSpline(t, rng.normal(0, sigma, len(t)))(np.arange(n)) for _ in range(2)])


def ball(raw, labels, half, pitch, rng, p=DEFAULT):
    """raw (n,2) metres -> degraded raw ball."""
    n = len(raw)
    out = raw.astype(float)
    if p["smooth"]:                       # the match's ball track is smoother than the outside games'
        from scipy.ndimage import gaussian_filter1d
        ok = np.isfinite(out[:, 0])
        filled = np.where(ok[:, None], out, 0.0)
        w = gaussian_filter1d(ok.astype(float), p["smooth"])
        out = np.where(ok[:, None], gaussian_filter1d(filled, p["smooth"], axis=0) / np.maximum(w, 1e-6)[:, None], np.nan)
    env = np.exp(p["tail"] * drift(n, 1.0, p["env_knot"], rng)[:, :1]) if p["tail"] else 1.0
    out = out + env * drift(n, p["sigma"], p["knot"], rng)
    xn = raw[:, 0] / pitch[0]
    out[:, 0] += p["stretch"] * (xn - 0.5)
    out[:, 1] += p["dy"]
    dead = (labels == 2) | (half == 0)
    e = np.diff(np.r_[0, dead.astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        if b - a < p["hide_min"] or rng.random() < p["keep"]:
            continue
        lo = a + int(rng.integers(p["lead"][0], p["lead"][1] + 1))
        hi = b - int(rng.integers(p["trail"][0], p["trail"][1] + 1))
        if hi > lo:
            out[lo:hi] = np.nan
    # a few short losses in live play
    live = np.flatnonzero(~dead)
    for _ in range(int(p["live_gap_rate"] * len(live) / 1500)):
        s = int(rng.choice(live))
        out[s:s + int(np.clip(rng.lognormal(np.log(15), 1.0), 2, 400))] = np.nan
    out[half == 0] = np.nan
    return out


def players(z, rng, p=DEFAULT):
    """Random dropouts in place on the cleaned player arrays."""
    n, S = z["player_xy"].shape[:2]
    for k in range(S):
        f = int(rng.exponential(p["on_mean_s"]) * 25)
        while f < n:
            L = max(5, int(rng.lognormal(np.log(p["off_median_s"]), 1.0) * 25))
            for key in ("player_xy", "player_xy_raw", "player_vel"):
                z[key][f:f + L, k] = np.nan
            z["player_valid"][f:f + L, k] = False
            f += L + max(25, int(rng.exponential(p["on_mean_s"]) * 25))
    return z
