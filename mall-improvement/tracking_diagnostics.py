"""Why is the ball so far from the players in this match's tracking? Label-free diagnostics.

    python tracking_diagnostics.py --data-dir ..\\init\\data

Reads only the automatic tracking (tactical and, if present, broadcast ATD). It never opens a
possession reference file. It prints aggregate statistics only (no frame-level data) and writes
the same text to tracking_diagnostics_report.txt next to this script.

  A  tactical ball vs players
     A1 nearest-player distance while a player plausibly has the ball
     A2 time lag: does shifting the ball track in time bring it closer to the players?
     A3 velocity lag: at which shift does the ball's velocity best match the carrier's?
     A4 spatial offset: is the ball systematically displaced from the carrier, and does the
        displacement depend on where on the pitch it is (camera parallax / homography bias)?
     A5 noise and availability
  B  broadcast tracking as a second ball source
     B1 availability
     B2 alignment to the tactical timeline, found from the ball track itself
     B3 on aligned frames: do the two balls agree, which sits closer to the players, and how
        many tactical ball gaps the broadcast ball could fill
"""

import argparse
import io
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

PITCH = np.array([105.3, 68.0])
HALVES = ((11185, 90651), (110808, 190197))          # graded video frames, inclusive
FPS = 25
OUT = []


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    OUT.append(line)


def read_atd(txt, xml):
    with open(txt) as f:
        text = f.read()
    first = text[:text.index("\n")]
    _, players, ball = first.split(":")
    n_slots = len(players.split(";"))
    sw = len(players.split(";")[0].split(","))
    bw = len(ball.split(","))
    flat = text.replace(":", ",").replace(";", ",")
    try:
        import pandas as pd
        v = pd.read_csv(io.StringIO(flat), header=None, dtype=np.float64).to_numpy()
    except ImportError:
        v = np.loadtxt(io.StringIO(flat), delimiter=",", dtype=np.float64, ndmin=2)
    frame = v[:, 0].astype(np.int64) - 1
    n = int(frame.max()) + 1
    P = np.full((n, n_slots, 2), np.nan)
    B = np.full((n, 2), np.nan)
    P[frame] = v[:, 1:1 + n_slots * sw].reshape(-1, n_slots, sw)[:, :, :2] * PITCH
    B[frame] = v[:, 1 + n_slots * sw:1 + n_slots * sw + 2] * PITCH
    root = ET.parse(xml).getroot()
    teams = list(root.iter("Team"))
    home = teams[0].get("id")
    team = np.array([0 if p.get("teamId") == home else 1 for p in root.iter("Player")], np.int8)
    if len(team) != n_slots:
        team = np.resize(team, n_slots)
    return P, B, team


def graded_mask(n):
    m = np.zeros(n, bool)
    for a, b in HALVES:
        m[a:min(n, b + 1)] = True
    return m


def nearest(P, B):
    """distance to, and slot of, the nearest player for every frame (inf where undefined)."""
    d = np.linalg.norm(P - B[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    k = np.argmin(d, axis=1)
    return d[np.arange(len(d)), k], k


def vel(X):
    v = np.full_like(X, np.nan)
    v[1:-1] = (X[2:] - X[:-2]) * (FPS / 2.0)
    return v


def q(x, qs=(10, 25, 50, 75, 90)):
    x = x[np.isfinite(x)]
    if not len(x):
        return "n=0"
    return "n={} ".format(len(x)) + " ".join("p{}={:.2f}".format(p, v) for p, v in zip(qs, np.percentile(x, qs)))


def shift(X, s):
    """X at frame t+s, aligned to t."""
    Y = np.full_like(X, np.nan)
    if s > 0:
        Y[:-s] = X[s:]
    elif s < 0:
        Y[-s:] = X[:s]
    else:
        Y = X.copy()
    return Y


def section_a(P, B, team):
    n = len(B)
    g = graded_mask(n)
    seen = np.isfinite(B[:, 0])
    inside = seen & (B[:, 0] > 1) & (B[:, 0] < PITCH[0] - 1) & (B[:, 1] > 1) & (B[:, 1] < PITCH[1] - 1)
    vb = vel(B)
    sb = np.linalg.norm(vb, axis=1)
    d, k = nearest(P, B)
    say("== A  tactical ball vs players ==")
    say("frames", n, "| graded", int(g.sum()), "| ball seen on graded {:.3f}".format(seen[g].mean()))
    say("players seen per frame (graded): team0 {:.2f} team1 {:.2f}".format(
        np.isfinite(P[g][:, team == 0, 0]).sum(1).mean(), np.isfinite(P[g][:, team == 1, 0]).sum(1).mean()))

    say("-- A1 nearest-player distance (m), ball inside the pitch --")
    base = g & inside & np.isfinite(d)
    say("all frames            ", q(d[base]))
    for lo, hi in ((0, 0.5), (0.5, 2), (2, 6), (6, 12), (12, 40)):
        m = base & (sb >= lo) & (sb < hi)
        say("ball speed {:>4}-{:<4} m/s".format(lo, hi), q(d[m]))
    say("share of frames with nearest player < 1.0 / 1.5 / 2.5 m: {:.3f} / {:.3f} / {:.3f}".format(
        *(np.mean(d[base] < r) for r in (1.0, 1.5, 2.5))))

    # frames where a player plausibly carries the ball: ball moving at running pace, a player
    # within 4 m. The mask is fixed at zero shift so every shift is scored on the same frames.
    carry = base & (sb > 1.5) & (sb < 8) & (d < 4)
    say("-- A2 time shift of the ball track (frames; + means the ball track is LATE) --")
    say("carry frames used:", int(carry.sum()))
    best = None
    for s in range(-15, 16):
        ds, _ = nearest(P[carry], shift(B, s)[carry])
        ds = ds[np.isfinite(ds)]
        med, mean = np.median(ds), np.mean(np.minimum(ds, 6))
        say("shift {:+3d}  median {:.3f}  mean(capped 6 m) {:.3f}".format(s, med, mean))
        if best is None or med < best[1]:
            best = (s, med)
    say("best shift by median: {:+d} frames ({:+.2f} s), median {:.3f} m".format(best[0], best[0] / FPS, best[1]))

    say("-- A3 velocity lag: correlation of ball velocity (shifted) with the nearest player's --")
    idx = np.flatnonzero(carry)
    vp_all = vel(P.reshape(n, -1)).reshape(P.shape)
    vp = vp_all[idx, k[idx]]
    bestc = None
    for s in range(-15, 16):
        vs = shift(vb, s)[idx]
        ok = np.isfinite(vs).all(1) & np.isfinite(vp).all(1)
        a, b = vs[ok], vp[ok]
        c = float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))
        say("shift {:+3d}  corr {:.4f}".format(s, c))
        if bestc is None or c > bestc[1]:
            bestc = (s, c)
    say("best shift by velocity correlation: {:+d} frames, corr {:.4f}".format(*bestc))

    say("-- A4 spatial offset ball - nearest player (m), ball slower than 2 m/s, player < 3 m --")
    slow = base & (sb < 2) & (d < 3)
    idx = np.flatnonzero(slow)
    off = B[idx] - P[idx, k[idx]]
    say("frames", len(idx), "| mean dx {:+.3f} dy {:+.3f} | median dx {:+.3f} dy {:+.3f} | |offset| median {:.3f}".format(
        off[:, 0].mean(), off[:, 1].mean(), np.median(off[:, 0]), np.median(off[:, 1]),
        np.median(np.linalg.norm(off, axis=1))))
    bx = B[idx, 0] / PITCH[0]
    by = B[idx, 1] / PITCH[1]
    say("by pitch zone (x third, y third): n, mean dx, mean dy")
    for i in range(3):
        for j in range(3):
            m = (bx >= i / 3) & (bx < (i + 1) / 3) & (by >= j / 3) & (by < (j + 1) / 3)
            if m.sum() >= 50:
                say("  x{} y{}  n={:6d}  dx {:+.3f}  dy {:+.3f}".format(i, j, int(m.sum()), off[m, 0].mean(), off[m, 1].mean()))
    # linear model: offset = a + b * (position - centre)
    Xd = np.column_stack([np.ones(len(idx)), bx - 0.5, by - 0.5])
    for name, col in (("dx", 0), ("dy", 1)):
        coef = np.linalg.lstsq(Xd, off[:, col], rcond=None)[0]
        say("  {} = {:+.3f} {:+.3f}*(x-0.5) {:+.3f}*(y-0.5)   (x, y normalised)".format(name, *coef))

    say("-- A5 noise and availability --")
    acc = np.linalg.norm(B[2:] - 2 * B[1:-1] + B[:-2], axis=1) * FPS * FPS
    say("ball acceleration (m/s2), graded:", q(acc[g[1:-1]]))
    pacc = np.linalg.norm(P[2:] - 2 * P[1:-1] + P[:-2], axis=2) * FPS * FPS
    say("player acceleration (m/s2), graded:", q(pacc[g[1:-1]].ravel()[::7]))
    say("ball speed (m/s), graded:", q(sb[g]))
    miss = (~seen) & g
    e = np.diff(np.r_[0, miss.astype(int), 0])
    L = np.flatnonzero(e == -1) - np.flatnonzero(e == 1)
    say("ball gaps on graded frames: count {} | length frames {}".format(len(L), q(L.astype(float))))
    say("share of missing frames in gaps <=25 / 26-125 / >125 frames: {:.3f} / {:.3f} / {:.3f}".format(
        L[L <= 25].sum() / max(1, L.sum()), L[(L > 25) & (L <= 125)].sum() / max(1, L.sum()),
        L[L > 125].sum() / max(1, L.sum())))
    return d


def xcorr_terms(a, ma, s, ms, nfft):
    """sum over chunk of ma*ms*(a - s)^2 and of ma*ms, for every start position of the chunk in s."""
    A0 = np.fft.rfft(np.where(ma, 1.0, 0.0)[::-1], nfft)
    A1 = np.fft.rfft(np.where(ma, a, 0.0)[::-1], nfft)
    A2 = np.fft.rfft(np.where(ma, a * a, 0.0)[::-1], nfft)
    S0 = np.fft.rfft(np.where(ms, 1.0, 0.0), nfft)
    S1 = np.fft.rfft(np.where(ms, s, 0.0), nfft)
    S2 = np.fft.rfft(np.where(ms, s * s, 0.0), nfft)
    m = len(a)
    f = lambda X, Y: np.fft.irfft(X * Y, nfft)[m - 1:m - 1 + len(s) - m + 1]
    ssd = f(A2, S0) - 2 * f(A1, S1) + f(A0, S2)
    cnt = f(A0, S0)
    return ssd, cnt


def section_b(P, B, team, Pb, Bb, teamb, d_tac):
    n, nb = len(B), len(Bb)
    say("")
    say("== B  broadcast tracking ==")
    seen_b = np.isfinite(Bb[:, 0])
    say("broadcast frames", nb, "| ball seen {:.3f} | players seen per frame {:.2f}".format(
        seen_b.mean(), np.isfinite(Pb[:, :, 0]).sum(1).mean()))
    seen_t = np.isfinite(B[:, 0])

    say("-- B2 alignment from the ball track (chunks of 1500 broadcast frames) --")
    chunk = 1500
    nfft = 1 << int(np.ceil(np.log2(n + chunk)))
    flips = {"same": (1, 1), "x mirrored": (-1, 1), "y mirrored": (1, -1), "both mirrored": (-1, -1)}
    results = []
    for a0 in range(0, nb - chunk + 1, chunk):
        mb = seen_b[a0:a0 + chunk]
        if mb.sum() < 400:
            continue
        best = None
        for fname, (fx, fy) in flips.items():
            cb = Bb[a0:a0 + chunk].copy()
            if fx < 0:
                cb[:, 0] = PITCH[0] - cb[:, 0]
            if fy < 0:
                cb[:, 1] = PITCH[1] - cb[:, 1]
            sx, c = xcorr_terms(np.nan_to_num(cb[:, 0]), mb, np.nan_to_num(B[:, 0]), seen_t, nfft)
            sy, _ = xcorr_terms(np.nan_to_num(cb[:, 1]), mb, np.nan_to_num(B[:, 1]), seen_t, nfft)
            ok = c >= 300
            if not ok.any():
                continue
            rms = np.sqrt(np.maximum(sx + sy, 0) / np.maximum(c, 1))
            rms[~ok] = np.inf
            t0 = int(np.argmin(rms))
            srt = np.sort(rms[np.isfinite(rms)])
            # runner-up outside +-50 frames of the best
            far = rms.copy()
            far[max(0, t0 - 50):t0 + 51] = np.inf
            second = float(far.min())
            if best is None or rms[t0] < best[2]:
                best = (fname, t0, float(rms[t0]), int(c[t0]), second)
            del srt
        if best:
            results.append((a0, best[1] - a0) + best)
    if not results:
        say("no chunk could be aligned")
        return
    good = [r for r in results if r[4] < 3.0 and r[6] > 1.5 * r[4]]
    say("chunks tried {} | confidently aligned {} (rms < 3 m and runner-up 1.5x worse)".format(len(results), len(good)))
    say("orientation of confident chunks:", {f: sum(1 for r in good if r[2] == f) for f in flips})
    offs = np.array([r[1] for r in good])
    if len(offs):
        say("offset tactical_frame - broadcast_frame: min {} max {} | distinct (rounded to 25) {}".format(
            offs.min(), offs.max(), len(set((offs // 25).tolist()))))
        say("rms of aligned chunks (m):", q(np.array([r[4] for r in good])))
        say("first 60 confident chunks: broadcast_start, offset, rms, overlap")
        for r in good[:60]:
            say("  b={:6d} off={:6d} rms={:.2f} n={}".format(r[0], r[1], r[4], r[5]))

    say("-- B3 the two balls on aligned frames --")
    fname = max(flips, key=lambda f: sum(1 for r in good if r[2] == f))
    fx, fy = flips[fname]
    map_t = np.full(nb, -1, np.int64)
    for r in good:
        a0, off = r[0], r[1]
        t = np.arange(a0, a0 + chunk) + off
        okk = (t >= 0) & (t < n)
        map_t[a0:a0 + chunk][okk] = t[okk]
    bi = np.flatnonzero(map_t >= 0)
    ti = map_t[bi]
    Bx = Bb[bi].copy()
    Px = Pb[bi].copy()
    if fx < 0:
        Bx[:, 0] = PITCH[0] - Bx[:, 0]
        Px[:, :, 0] = PITCH[0] - Px[:, :, 0]
    if fy < 0:
        Bx[:, 1] = PITCH[1] - Bx[:, 1]
        Px[:, :, 1] = PITCH[1] - Px[:, :, 1]
    g = graded_mask(n)[ti]
    both = g & np.isfinite(Bx[:, 0]) & seen_t[ti]
    say("aligned broadcast frames {} | on graded tactical frames {} ({:.1%} of graded)".format(
        len(bi), int(g.sum()), g.sum() / graded_mask(n).sum()))
    say("both balls seen:", int(both.sum()), "| distance between the two balls (m):", q(np.linalg.norm(Bx[both] - B[ti[both]], axis=1)))
    only_b = g & np.isfinite(Bx[:, 0]) & ~seen_t[ti]
    only_t = g & ~np.isfinite(Bx[:, 0]) & seen_t[ti]
    say("graded aligned frames with broadcast ball only: {} | tactical ball only: {} | neither: {}".format(
        int(only_b.sum()), int(only_t.sum()), int((g & ~np.isfinite(Bx[:, 0]) & ~seen_t[ti]).sum())))
    inside = (Bx[:, 0] > 1) & (Bx[:, 0] < PITCH[0] - 1) & (Bx[:, 1] > 1) & (Bx[:, 1] < PITCH[1] - 1)
    m = both & inside
    d_bb, _ = nearest(Px[m], Bx[m])
    d_bt, _ = nearest(P[ti[m]], Bx[m])
    say("on frames where both balls are seen, nearest-player distance (m):")
    say("  tactical ball  vs tactical players ", q(d_tac[ti[m]]))
    say("  broadcast ball vs broadcast players", q(d_bb))
    say("  broadcast ball vs tactical players ", q(d_bt))
    for r_ in (1.0, 1.5):
        say("  share < {} m: tactical/tactical {:.3f} | broadcast/broadcast {:.3f} | broadcast/tactical {:.3f}".format(
            r_, np.mean(d_tac[ti[m]] < r_), np.mean(d_bb < r_), np.mean(d_bt < r_)))
    # player agreement between feeds: broadcast players to nearest tactical player of the same team
    sub = np.flatnonzero(g)[::25]
    dd = []
    for j in sub:
        for t_ in (0, 1):
            a = Px[j][teamb == t_]
            a = a[np.isfinite(a[:, 0])]
            b = P[ti[j]][team == t_]
            b = b[np.isfinite(b[:, 0])]
            if len(a) and len(b):
                dd.append(np.linalg.norm(a[:, None] - b[None], axis=2).min(1))
    if dd:
        say("broadcast player to nearest same-team tactical player (m):", q(np.concatenate(dd)))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("MTAH_DATA_DIR", os.path.join(here, "..", "init", "data")))
    a = ap.parse_args()
    t = os.path.join(a.data_dir, "tactical", "BAR-ATH_tactical_atd")
    b = os.path.join(a.data_dir, "broadcast", "BAR-ATH_broadcast_atd")
    if not os.path.isfile(t + ".txt"):
        sys.exit("tactical ATD not found under " + a.data_dir)
    P, B, team = read_atd(t + ".txt", t + ".xml")
    d = section_a(P, B, team)
    if os.path.isfile(b + ".txt"):
        Pb, Bb, teamb = read_atd(b + ".txt", b + ".xml")
        section_b(P, B, team, Pb, Bb, teamb, d)
    else:
        say("")
        say("== B  broadcast tracking: not found at", b + ".txt", "==")
    with open(os.path.join(here, "tracking_diagnostics_report.txt"), "w") as f:
        f.write("\n".join(OUT) + "\n")
    print("\nreport written to", os.path.join(here, "tracking_diagnostics_report.txt"))


if __name__ == "__main__":
    main()
