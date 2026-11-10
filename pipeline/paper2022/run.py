"""Inputs, native paper output, and an explicit H/A/D viewer adapter."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .core import Parameters, detect, runs
from .events import classify, infer_goalkeepers
from .setpieces import detect as detect_restarts
from .sequences import native_sequences


def viewer_labels(core, half, dead, restarts, rejected):
    """Challenge-specific adapter, NOT an algorithm specified by the paper.

    Hold the last unambiguous accepted gain through travel/duels. Duel gains
    become usable at the first unambiguous frame for that player. No lookback
    across a loose-ball interval. Explicit fallback is needed where no owner exists.
    """
    labels = np.full(len(half), 2, np.int8)
    decisions = []
    previous_team, previous_half = None, None
    for h in (1, 2):
        for start, end in runs((half == h) & ~dead):
            start, end = int(start), int(end)
            changes = []
            for restart in restarts:
                confirmed = restart.get('reason') in ('complete_trigger_and_pattern',
                            's1_stationary_executor_rechecked_at_release', 's1_incomplete_trigger_missing_executor')
                if restart['frame'] == start and restart['team'] in (0, 1) and confirmed:
                    changes.append((start, restart['team'], 'restart_executor', restart['slot']))
            for gain in core['gains']:
                f, slot = gain['frame'], gain['slot']
                if not start <= f < end or (f, slot) in rejected:
                    continue
                if core['duel'][f]:
                    possible = np.flatnonzero(~core['duel'][f:gain['end']] & core['pz'][f:gain['end'], slot])
                    if not len(possible):
                        continue
                    f += int(possible[0])
                changes.append((f, gain['team'], 'verified_player_gain', slot))
            changes.sort()
            by_frame = {}
            for f, team, why, slot in changes:
                by_frame.setdefault(f, []).append((team, why, slot))
            starting = {row[0] for row in by_frame.get(start, [])}
            initial = starting.pop() if len(starting) == 1 else None
            reason = 'verified_gain_or_restart_at_start'
            if initial is None and previous_team is not None and previous_half == h:
                initial, reason = previous_team, 'fallback_last_team_unresolved_restart'
            if initial is None:
                d = core['distances'][start]
                valid = np.isfinite(d)
                if valid.any():
                    slot = int(np.nanargmin(d))
                    initial = int(core['teams'][start, slot])
                    reason = 'fallback_nearest_observation_no_verified_gain'
                else:
                    initial, reason = 0, 'fallback_home_no_observation'
            labels[start:end] = initial
            decisions.append(dict(frame=start, end=end, team=int(initial), reason=reason))
            # Conflicting simultaneous gains remain unresolved rather than choosing a slot ID.
            for f, group in sorted(by_frame.items()):
                teams = {row[0] for row in group}
                if len(teams) == 1:
                    team = teams.pop()
                    labels[f:end] = team
                    decisions.append(dict(frame=f, team=int(team), slot=group[0][2], reason=group[0][1]))
            previous_team, previous_half = int(labels[end - 1]), h
    return labels, decisions


def reproduce(players, teams, ball, half, pitch, signs, dead, params=Parameters(), keepers=None):
    teams = np.broadcast_to(teams, players.shape[:2]).copy() if np.ndim(teams) == 1 else np.asarray(teams).copy()
    teams[~np.isfinite(players).all(axis=2)] = -1
    dead = np.asarray(dead, bool) | (half == 0)
    core = detect(players, teams, ball, half, dead, params)
    core['teams'] = teams
    if keepers is None:
        keepers, keeper_metadata = infer_goalkeepers(players, teams, half, signs, pitch)
        keeper_source = 'inferred_from_median_defensive_position'
    else:
        keeper_metadata, keeper_source = [], 'supplied_player_roles'
    restarts = detect_restarts(players, teams, half, signs, pitch, dead, core)
    passing, receiving, rejected = classify(core, players, teams, half, signs, pitch, dead, restarts, keepers)
    player, control, sequences, loose = native_sequences(core, half, dead, restarts, passing, rejected)
    core.update(player_owner=player, ball_control=control)
    labels, decisions = viewer_labels(core, half, dead, restarts, rejected)
    trace = dict(schema=1, fps=params.fps, parameters=core['parameters'],
                 keeper_source=keeper_source, goalkeeper_assignments=keeper_metadata,
                 gains=core['gains'], loss_candidates=core['losses'],
                 passing_events=passing, receiving_events=receiving, restarts=restarts,
                 control_intervals=core['intervals'], viewer_decisions=decisions,
                 control_sequences=sequences, no_possession_sequences=loose,
                 rejected_goalkeeper_gains=[list(key) for key in sorted(rejected)])
    return labels, core, trace


def read_status(path, n):
    with Path(path).open() as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['frame', 'in_play']:
            raise ValueError('Status CSV must have frame,in_play columns; not ownership labels')
        rows = list(reader)
    if [int(r['frame']) for r in rows] != list(range(n)) or any(r['in_play'] not in ('0', '1') for r in rows):
        raise ValueError('Status CSV needs every ordered frame and binary 0/1 in_play values')
    return np.array([r['in_play'] == '0' for r in rows])


def tactical(output, profile='bc', status='missing', status_file=None):
    from ..mall.common import CACHE, load_layer
    from ..run import write_csv
    z = load_layer('layer0')
    # Layer 0 is an input adapter here: only its original raw coordinates,
    # source teams, pitch, half markers and attack-direction metadata are read.
    ball, players, half = z['ball_xy_raw'], z['player_xy_raw'], z['half']
    if status == 'provided':
        if status_file is None:
            raise ValueError('--status provided requires --status-file')
        dead = read_status(status_file, len(half))
    elif status == 'mall':
        dead = load_layer('layer3')['dead'].astype(bool)
    else:
        dead = ~np.isfinite(ball).all(axis=1)
    labels, core, trace = reproduce(players, z['team_of_slot'], ball, half, z['pitch'], z['attack_sign'],
                                   dead, Parameters.profile(profile))
    directory = Path(CACHE).parent / 'paper2022'
    directory.mkdir(parents=True, exist_ok=True)
    code = b''.join(p.read_bytes() for p in sorted(Path(__file__).parent.glob('*.py')))
    trace.update(game='BAR-ATH', status_input=status, profile=profile,
                 source_fingerprint=hashlib.sha256(code).hexdigest(),
                 source='https://doi.org/10.1007/s12283-022-00381-6',
                 ownership_adapter='last unambiguous verified gain; carry last team at unresolved restart, no future-prefix backdating')
    trace_path = directory / f'tactical-{profile}-{status}.events.json'
    trace_path.write_text(json.dumps(trace, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(directory / f'tactical-{profile}-{status}.npz',
                        label=labels, zone_state=core['state'], duel=core['duel'],
                        player_owner=core['player_owner'], ball_control=core['ball_control'])
    write_csv(labels, half, output)
    print(f"Paper reproduction: {len(trace['gains'])} gains, {len(trace['passing_events'])} passing events, "
          f"{len(trace['restarts'])} restarts; profile={profile}, status={status}", flush=True)
