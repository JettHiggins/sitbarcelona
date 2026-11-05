"""Verify a frame,label CSV against a local video's frame count using ffprobe."""
import argparse
import csv
import json
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video')
    parser.add_argument('csv')
    args = parser.parse_args()
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=nb_frames,nb_read_frames,avg_frame_rate', '-of', 'json']
    stream = json.loads(subprocess.check_output(command + [args.video]))['streams'][0]
    count = stream.get('nb_frames', 'N/A')
    if count == 'N/A':
        print('Frame count is absent from metadata; scanning video…', flush=True)
        stream = json.loads(subprocess.check_output(command + ['-count_frames', args.video]))['streams'][0]
        count = stream['nb_read_frames']
    count = int(count)
    with open(args.csv, newline='', encoding='utf-8-sig') as source:
        rows = list(csv.DictReader(source))
    frames = [int(row['frame']) for row in rows]
    valid = (frames == list(range(count)) and
             all(row['label'] in {'H', 'A', 'D', 'X'} for row in rows))
    print(f'Video: {count:,} frames ({stream["avg_frame_rate"]} fps)')
    print(f'CSV:   {len(rows):,} rows; {len(set(frames)):,} unique frames')
    print('MATCH: one valid label per video frame, in order.' if valid else
          'FAIL: expected ordered frames 0 through video frame count minus one, with valid labels.')
    return 0 if valid else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, IndexError, subprocess.CalledProcessError) as error:
        print(f'Could not verify: {error}', file=sys.stderr)
        sys.exit(1)
