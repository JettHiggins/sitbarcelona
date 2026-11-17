"""Inventory other-game videos before any supervised visual experiment.

This reads filenames and optional ffprobe metadata only. It never reads challenge
frames or ground truth, and never infers that a video is synchronized from its name.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess


def probe(path):
    if not shutil.which('ffprobe'):
        return dict(status='ffprobe unavailable')
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=r_frame_rate,nb_frames,duration',
               '-of', 'json', str(path)]
    try:
        data = json.loads(subprocess.check_output(command, text=True, timeout=30))
        return dict(status='probed', streams=data.get('streams', []))
    except (subprocess.SubprocessError, ValueError) as error:
        return dict(status='probe failed', error=str(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video-root', type=Path, required=True,
                        help='Local directory containing permitted OTHER-game videos')
    parser.add_argument('--alignment-manifest', type=Path,
                        help='Optional JSON with verified tracking/video sync and label provenance')
    parser.add_argument('--output', type=Path,
                        default=Path('pipeline/reports/wes/video-audit.json'))
    args = parser.parse_args()
    if not args.video_root.is_dir():
        parser.error(f'video root does not exist: {args.video_root}')
    manifest = {}
    if args.alignment_manifest:
        manifest = json.loads(args.alignment_manifest.read_text())
    report = dict(video_root=str(args.video_root.resolve()),
                  alignment_manifest=str(args.alignment_manifest.resolve()) if
                  args.alignment_manifest else None,
                  note='Names/durations do not establish synchronization or usable labels.',
                  games={})
    for game in (1, 2, 3):
        matches = sorted(p for p in args.video_root.rglob('*')
                         if p.is_file() and p.suffix.lower() in ('.mp4', '.mov', '.mkv')
                         and (f'game{game}' in p.stem.lower().replace(' ', '') or
                              f'game_{game}' in p.stem.lower().replace(' ', '')))
        report['games'][f'bench{game}'] = dict(
            videos=[dict(path=str(p.resolve()), bytes=p.stat().st_size, metadata=probe(p))
                    for p in matches],
            manifest=manifest.get(f'bench{game}'),
            ready_for_supervised_training=bool(matches and
                                               (manifest.get(f'bench{game}') or {}).get('sync_verified') and
                                               (manifest.get(f'bench{game}') or {}).get('labels_usable')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
