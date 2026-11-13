"""Possession from the trajectories themselves: a small recurrent network over every frame.

The boosted models (crossgame.py) see each frame through window means: good at "who has had the
ball around here", blunt about the instant it changes hands, and almost blind to a possession
that lasts a second. This model reads the raw 25 Hz geometry instead - where the three nearest
players of each team are relative to the ball and how they move against it - forwards and
backwards in time, and says for every frame

  p_home    P(home is in possession), trained on the live frames of other games
  p_change  P(no change / possession passes to home / to away within 6 frames of here)

Like the boosted models it is trained only on bench games (never the game it labels, never a
label of the challenge match) and it is symmetric: every training window is also shown with the
teams swapped and with the pitch mirrored, and predictions average the two team orders.

Needs PyTorch (CPU works, a GPU is much faster). Layout of a feature row:

  [ home block | away block | neutral ]      swap = exchange the two blocks, flip the labels
"""

import os
import time

import numpy as np

K_NEAR = 3
PER_PLAYER = 6                   # distance, dx, dy, relative vx, vy, player speed
TEAM_EXTRA = 8
BLOCK = K_NEAR * PER_PLAYER + TEAM_EXTRA
NEUTRAL = 7
DIM = 2 * BLOCK + NEUTRAL
CHANGE_WINDOW = 6
HIDDEN = 128
LAYERS = 3
WINDOW = 500                     # frames per training window (20 s)
BATCH = 64
STEPS = 6000                     # ~6 min on a laptop GPU; 1500 / 3000 / 6000 steps were compared on
                                 # two held-out games and the longest was best on both
LR = 2e-3
PIECE = 4000                     # frames predicted at a time ...
MARGIN = 750                     # ... with this much context either side (30 s)


def features(z):
    """(n, DIM) float32 from a layer-0 cache. Ball-relative parts are 0 where the ball is unknown."""
    xy = z["player_xy"].astype(np.float32)
    vel = z["player_vel"].astype(np.float32)
    team = z["player_team"]
    ball = z["ball_xy"].astype(np.float32)
    bvel = z["ball_vel"].astype(np.float32)
    half = z["half"].astype(int)
    L, W = (float(v) for v in z["pitch"])
    sign = z["attack_sign"][half].astype(np.float32)          # (n, 2): +1 attacks +x
    n = len(half)
    known = ~np.isnan(ball[:, 0])
    b = np.where(known[:, None], ball, 0.0)
    bv = np.where(known[:, None] & ~np.isnan(bvel[:, :1]), bvel, 0.0)
    out = np.zeros((n, DIM), np.float32)
    rows = np.arange(n)[:, None]
    with np.errstate(all="ignore"):
        for t in (0, 1):
            mine = (team == t) & ~np.isnan(xy[..., 0])
            rel = xy - b[:, None, :]
            d = np.where(mine, np.hypot(rel[..., 0], rel[..., 1]), np.inf)
            order = np.argsort(d, axis=1)[:, :K_NEAR]
            dk = d[rows, order]
            ok = np.isfinite(dk) & known[:, None]
            s = sign[:, t][:, None]                            # this team's attacking direction
            relk = rel[rows, order]
            pv = np.nan_to_num(vel[rows, order])
            blk = np.zeros((n, K_NEAR, PER_PLAYER), np.float32)
            blk[..., 0] = np.minimum(dk, 30.0) / 10.0
            blk[..., 1] = np.clip(relk[..., 0] * s, -30, 30) / 10.0
            blk[..., 2] = np.clip(relk[..., 1], -30, 30) / 10.0
            blk[..., 3] = np.clip((pv[..., 0] - bv[:, None, 0]) * s, -20, 20) / 5.0
            blk[..., 4] = np.clip(pv[..., 1] - bv[:, None, 1], -20, 20) / 5.0
            blk[..., 5] = np.minimum(np.hypot(pv[..., 0], pv[..., 1]), 12.0) / 5.0
            blk = np.where(ok[..., None], blk, 0.0)
            # dy's sign is arbitrary (which touchline): keep it, mirroring is a training augmentation
            a = t * BLOCK
            out[:, a:a + K_NEAR * PER_PLAYER] = blk.reshape(n, -1)
            e = a + K_NEAR * PER_PLAYER
            dn = np.where(mine, d, np.inf)
            out[:, e + 0] = np.where(known, np.sum(dn < 2.0, axis=1), 0) / 3.0
            out[:, e + 1] = np.where(known, np.sum(dn < 5.0, axis=1), 0) / 5.0
            out[:, e + 2] = np.where(known, (b[:, 0] - L / 2) * sign[:, t], 0.0) / (L / 2)
            out[:, e + 3] = np.where(known, bv[:, 0] * sign[:, t], 0.0) / 10.0
            xs = np.where(mine, xy[..., 0], np.nan)
            vx = np.where(mine, vel[..., 0], np.nan)
            sp = np.where(mine, np.hypot(vel[..., 0], vel[..., 1]), np.nan)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[:, e + 4] = np.nan_to_num((np.nanmean(xs, axis=1) - L / 2) * sign[:, t]) / (L / 2)
                out[:, e + 5] = np.nan_to_num(np.nanmean(vx, axis=1) * sign[:, t]) / 3.0
                out[:, e + 6] = np.nan_to_num(np.nanmean(sp, axis=1)) / 3.0
                out[:, e + 7] = np.nan_to_num(np.nanstd(xs, axis=1)) / 20.0
        speed = np.hypot(bv[:, 0], bv[:, 1])
        c = 2 * BLOCK
        out[:, c + 0] = known
        out[:, c + 1] = np.minimum(speed, 40.0) / 10.0
        out[:, c + 2] = np.clip(np.r_[0.0, np.diff(speed)], -5, 5) * known
        out[:, c + 3] = np.clip(bv[:, 1], -30, 30) / 10.0
        out[:, c + 4] = np.where(known, (b[:, 1] - W / 2) / (W / 2), 0.0)
        out[:, c + 5] = np.minimum(z["ball_frames_since_seen"], 250) / 250.0
        out[:, c + 6] = np.minimum(z["ball_frames_until_seen"], 250) / 250.0
    return np.nan_to_num(out)


# columns whose sign flips when the pitch is mirrored across its long axis (y -> W - y)
_MIRROR = [t * BLOCK + k * PER_PLAYER + j for t in (0, 1) for k in range(K_NEAR) for j in (2, 4)] \
    + [2 * BLOCK + 3, 2 * BLOCK + 4]


def swap_teams(X):
    return np.concatenate([X[..., BLOCK:2 * BLOCK], X[..., :BLOCK], X[..., 2 * BLOCK:]], axis=-1)


def targets(truth):
    """(possession 1 home / 0 away / -1 not live, change 0 none / 1 to home / 2 to away / -1 n.a.)"""
    n = len(truth)
    poss = np.where(truth == 0, 1, np.where(truth == 1, 0, -1)).astype(np.int64)
    ch = np.zeros(n, np.int64)
    ha = np.isin(truth, [0, 1])
    for i in np.flatnonzero(ha[1:] & ha[:-1] & (truth[1:] != truth[:-1])) + 1:
        ch[max(0, i - CHANGE_WINDOW):i + CHANGE_WINDOW + 1] = 1 if truth[i] == 0 else 2
    ch[~ha] = -1
    return poss, ch


def build_net():
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(DIM, HIDDEN), nn.ReLU())
            self.gru = nn.GRU(HIDDEN, HIDDEN, LAYERS, batch_first=True, bidirectional=True,
                              dropout=0.1)
            self.poss = nn.Linear(2 * HIDDEN, 1)
            self.change = nn.Linear(2 * HIDDEN, 3)

        def forward(self, x):
            h, _ = self.gru(self.inp(x))
            return self.poss(h).squeeze(-1), self.change(h)

    return Net()


def spell_weights(truth, half, power=0.5):
    """Training-only weights: shorten the influence gap between long and short spells.

    A 10-second possession has weight one. Weights are bounded so a noisy
    single-frame annotation cannot dominate training. Never used at inference.
    """
    truth, half = np.asarray(truth), np.asarray(half)
    out = np.zeros(len(truth), np.float32)
    bounds = np.r_[0, np.flatnonzero((truth[1:] != truth[:-1]) |
                                    (half[1:] != half[:-1])) + 1, len(truth)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        if half[a] > 0 and truth[a] in (0, 1):
            out[a:b] = np.clip((250.0 / (b - a)) ** power, 0.25, 4.0)
    return out


def train(games, seed=0, steps=STEPS, verbose=False, spell_balance=0.0):
    """games: [(X (n, DIM), truth (n,) 0 H / 1 A / 2 D / -1, half (n,))]. Returns a state dict."""
    import torch
    import torch.nn.functional as Fn
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True     # same seed, same machine -> same weights
    torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(seed)
    started = time.monotonic()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps)
    data = []
    for X, truth, half in games:
        poss, ch = targets(truth)
        sw = spell_weights(truth, half, spell_balance)
        for h in (1, 2):
            idx = np.flatnonzero(half == h)
            if len(idx) > WINDOW:
                a, b = idx[0], idx[-1] + 1
                data.append((X[a:b], poss[a:b], ch[a:b], sw[a:b]))
    sizes = np.array([len(d[0]) for d in data], float)
    weight = torch.tensor([0.3, 1.0, 1.0], device=dev)       # changes are ~4% of live frames
    net.train()
    for step in range(steps):
        xb, pb, cb, wb = [], [], [], []
        for g in rng.choice(len(data), BATCH, p=sizes / sizes.sum()):
            X, poss, ch, sw = data[g]
            s = int(rng.integers(0, len(X) - WINDOW))
            x, p, c = X[s:s + WINDOW], poss[s:s + WINDOW], ch[s:s + WINDOW]
            if rng.random() < 0.5:                             # the other team order
                x = swap_teams(x)
                p = np.where(p >= 0, 1 - p, -1)
                c = np.where(c == 1, 2, np.where(c == 2, 1, c))
            if rng.random() < 0.5:                             # the mirrored pitch
                x = x.copy()
                x[:, _MIRROR] *= -1
            xb.append(x), pb.append(p), cb.append(c)
            wb.append(sw[s:s + WINDOW])
        x = torch.from_numpy(np.stack(xb)).to(dev)
        p = torch.from_numpy(np.stack(pb)).to(dev)
        c = torch.from_numpy(np.stack(cb)).to(dev)
        lp, lc = net(x)
        live = p >= 0
        if spell_balance:
            w = torch.from_numpy(np.stack(wb)).to(dev)[live]
            ploss = (Fn.binary_cross_entropy_with_logits(lp[live], p[live].float(),
                                                        reduction="none") * w).sum() / w.sum()
        else:
            ploss = Fn.binary_cross_entropy_with_logits(lp[live], p[live].float())
        loss = ploss + Fn.cross_entropy(lc[c >= 0], c[c >= 0], weight=weight)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if verbose and (step % 250 == 0 or step == steps - 1):
            elapsed = time.monotonic() - started
            eta = elapsed * (steps - step - 1) / (step + 1)
            print("step {:5d}/{:d} loss {:.4f} elapsed {:.0f}s ETA {:.0f}s".format(
                step + 1, steps, float(loss), elapsed, eta), flush=True)
    return {k: v.cpu() for k, v in net.state_dict().items()}


def predict(state, X, half):
    """(p_home (n,), p_change (n, 3)) for one game; 0.5 / no-change outside the halves."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_net().to(dev)
    net.load_state_dict(state)
    net.eval()
    n = len(X)
    p_home = np.full(n, 0.5, np.float32)
    p_change = np.tile(np.array([1.0, 0.0, 0.0], np.float32), (n, 1))
    with torch.no_grad():
        for h in (1, 2):
            idx = np.flatnonzero(half == h)
            if not len(idx):
                continue
            a, b = idx[0], idx[-1] + 1
            # in overlapping pieces (cuDNN limits the length of one sequence); each piece keeps
            # only its middle, so every kept frame has MARGIN frames of context on both sides
            for s0 in range(a, b, PIECE):
                lo, hi = max(a, s0 - MARGIN), min(b, s0 + PIECE + MARGIN)
                x = torch.from_numpy(np.ascontiguousarray(X[lo:hi])[None]).to(dev)
                xs = torch.from_numpy(np.ascontiguousarray(swap_teams(X[lo:hi]))[None]).to(dev)
                lp, lc = net(x)
                lps, lcs = net(xs)
                ph = 0.5 * (torch.sigmoid(lp) + 1 - torch.sigmoid(lps))[0].cpu().numpy()
                pc = 0.5 * (torch.softmax(lc, -1) + torch.softmax(lcs, -1)[..., [0, 2, 1]])[0].cpu().numpy()
                e = min(b, s0 + PIECE)
                p_home[s0:e] = ph[s0 - lo:e - lo]
                p_change[s0:e] = pc[s0 - lo:e - lo]
    return p_home, p_change
