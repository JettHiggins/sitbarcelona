"""Explain scored/rejected frame evidence without reading possession truth."""
import argparse
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True, help='Inclusive last frame')
    args = parser.parse_args()
    z = np.load('pipeline/cache/touches/tactical.evidence.npz')
    trace = json.load(open('pipeline/cache/touches/tactical.touches.json'))
    if not 0 <= args.start <= args.end < len(z['valid_window']):
        parser.error('Frame range outside match')
    accepted = {e['frame']: e for e in trace['touches']}
    rejected = {e['frame']: e for e in trace['rejected']}
    for f in range(args.start, args.end+1):
        scores = z['actor_scores'][f]
        slot = int(scores.argmax())
        if not z['valid_window'][f]:
            reason = 'gap/half-boundary/jump in 25-frame window'
        elif not z['candidate'][f]:
            reason = 'stationary/no observable impulse'
        elif scores.max() < trace['parameters']['threshold']:
            reason = 'score below threshold (or no actor within radius)'
        elif f in accepted:
            reason = f"TOUCH team={accepted[f]['team']} distance={accepted[f]['distance_m']:.2f}m"
        elif f in rejected:
            reason = 'ambiguous actor'
        else:
            reason = 'not a separated per-actor score peak'
        print(f'{f}: actor={slot} score={scores[slot]:.3f} impulse={z["impulse_m_s"][f]:.3f}m/s {reason}')


if __name__ == '__main__':
    main()
