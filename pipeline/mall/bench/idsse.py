"""Read the seven open DFL / IDSSE Bundesliga matches into the bench's plain arrays.

    Bassek, Rein, Weber & Memmert (2025), "An integrated dataset of spatiotemporal and event data
    in elite soccer", Scientific Data. CC-BY 4.0. https://doi.org/10.6084/m9.figshare.28196177

    python -m pipeline.mall.bench.idsse --download      # ~2.6 GB into $IDSSE_SRC (default data/idsse)
    python -m pipeline.mall.bench.idsse                 # read every match and print its label shares

Why these games: the official position feed carries, on every frame, the ball status (in play /
dead) and the team in possession. That is a frame-exact H / A / D answer key made by the league's
data operation - no event-log reconstruction - on seven full matches. Unlike the Metrica games
the ball is tracked through stoppages, as it often is in the challenge match. These are other
games, which RULES.md allows for training and validation.

load(match) -> the metrica.load() dict plus
  label    (n,) int8 0 home, 1 away, 2 dead
  ball_z   (n,) metres (used only to decide where a simulated optical tracker loses the ball)
"""

import argparse
import os
import re
import urllib.request

import numpy as np

from ..common import ROOT

SRC = os.environ.get("IDSSE_SRC") or os.path.join(ROOT, "data", "idsse")
# match id -> (figshare file id of the match information, of the positions)
FILES = {
    "J03WPY": (51643487, 51643526), "J03WN1": (51643472, 51643517),
    "J03WMX": (51643475, 51643514), "J03WOH": (51643478, 51643520),
    "J03WQQ": (51643484, 51643529), "J03WOY": (51643481, 51643523),
    "J03WR9": (51643490, 51643532),
}
MATCHES = list(FILES)
URL = "https://ndownloader.figshare.com/files/{}"

_ATTR = re.compile(r'(\w+)="([^"]*)"')


def paths(match):
    return (os.path.join(SRC, match + "_meta.xml"), os.path.join(SRC, match + "_tracking.xml"))


def download(matches=MATCHES):
    os.makedirs(SRC, exist_ok=True)
    for m in matches:
        for path, fid in zip(paths(m), FILES[m]):
            if os.path.isfile(path) and os.path.getsize(path) > 1000:
                continue
            print("downloading", os.path.basename(path), flush=True)
            urllib.request.urlretrieve(URL.format(fid), path)


def _meta(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    env = dict(_ATTR.findall(re.search(r"<Environment[^>]*>", text).group(0)))
    gen = dict(_ATTR.findall(re.search(r"<General[^>]*>", text).group(0)))
    teams = {}
    for block in re.findall(r"<Team .*?</Team>", text, flags=re.S):
        head = dict(_ATTR.findall(block[:block.index(">")]))
        teams[head["TeamId"]] = head["TeamName"]
    home, away = gen["HomeTeamId"], gen["GuestTeamId"]
    return {"pitch": (float(env["PitchX"]), float(env["PitchY"])), "home": home, "away": away,
            "names": (teams.get(home, "Home"), teams.get(away, "Away"))}


def load(match):
    meta_path, track_path = paths(match)
    M = _meta(meta_path)
    L, W = M["pitch"]
    sets = []                 # (team 0/1/'ball'/None, section, frames N, values)
    cur = None
    with open(track_path, encoding="utf-8") as f:
        for line in f:
            i = line.find("<Frame ")
            if i >= 0 and cur is not None:
                a = dict(_ATTR.findall(line))
                if cur[0] == "ball":
                    cur[3].append((int(a["N"]), float(a["X"]), float(a["Y"]), float(a.get("Z", 0)),
                                   int(a.get("BallPossession", 0)), int(a.get("BallStatus", 0))))
                else:
                    cur[3].append((int(a["N"]), float(a["X"]), float(a["Y"])))
                continue
            i = line.find("<FrameSet")
            if i >= 0:
                a = dict(_ATTR.findall(line))
                tid = a.get("TeamId", "")
                who = "ball" if tid.upper() == "BALL" else (
                    0 if tid == M["home"] else 1 if tid == M["away"] else None)
                cur = None
                if who is not None:
                    cur = (who, a["GameSection"], a.get("PersonId"), [])
                    sets.append(cur)
    section_no = {"firstHalf": 1, "secondHalf": 2}
    # frame numbering: each half has its own contiguous N range, taken from the ball
    spans = {}
    for who, sec, pid, rows in sets:
        if who == "ball" and sec in section_no:
            N = np.array([r[0] for r in rows])
            spans[sec] = (int(N.min()), int(N.max()))
    offset, n = {}, 0
    for sec in ("firstHalf", "secondHalf"):
        offset[sec] = n - spans[sec][0]
        n += spans[sec][1] - spans[sec][0] + 1
    persons = sorted({(who, pid) for who, sec, pid, _ in sets if who != "ball" and sec in section_no},
                     key=lambda t: (t[0], t[1]))
    col = {p: i for i, p in enumerate(persons)}
    players = np.full((n, len(persons), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    ball_z = np.full(n, np.nan)
    label = np.full(n, 2, np.int8)
    period = np.zeros(n, np.int8)
    for who, sec, pid, rows in sets:
        if sec not in section_no or not rows:
            continue
        r = np.array(rows, float)
        idx = r[:, 0].astype(int) + offset[sec]
        ok = (idx >= 0) & (idx < n)
        idx, r = idx[ok], r[ok]
        xy = np.column_stack([r[:, 1] / L + 0.5, r[:, 2] / W + 0.5])
        if who == "ball":
            ball[idx] = xy
            ball_z[idx] = r[:, 3]
            poss, status = r[:, 4].astype(int), r[:, 5].astype(int)
            label[idx] = np.where(status == 1, np.where(poss == 2, 1, 0), 2)
            period[idx] = section_no[sec]
        else:
            players[idx, col[(who, pid)]] = xy
    team_of_slot = np.array([p[0] for p in persons], np.int8)
    return {"players": players, "ball": ball, "ball_z": ball_z, "team_of_slot": team_of_slot,
            "period": period, "label": label, "events": [], "names": M["names"],
            "pitch": M["pitch"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--download", action="store_true")
    p.add_argument("--matches", nargs="+", default=MATCHES)
    a = p.parse_args()
    if a.download:
        download(a.matches)
        return
    for m in a.matches:
        G = load(m)
        lab, per = G["label"], G["period"]
        g = per > 0
        print("{}  {} v {}  frames {}  H {:.1%} A {:.1%} D {:.1%}  ball seen {:.1%}".format(
            m, G["names"][0], G["names"][1], int(g.sum()), np.mean(lab[g] == 0),
            np.mean(lab[g] == 1), np.mean(lab[g] == 2), np.mean(~np.isnan(G["ball"][g, 0]))))


if __name__ == "__main__":
    main()
