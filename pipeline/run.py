"""Run an isolated pipeline and export predictions for the possession viewer."""
import argparse
import importlib
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MALL_LAYERS = ('layer1_features', 'layer2_touches', 'layer3_dead_ball')
BENCH_TAGS = ('bench1', 'bench2', 'bench3') + tuple(f'dfl{i}' for i in range(1, 8))


def run_module(name, game='tactical'):
    print(f'Running {name} ({game})', flush=True)
    env = dict(os.environ, POSSESSION_GAME=game)
    subprocess.run([sys.executable, '-m', name], cwd=ROOT, env=env, check=True)


def write_csv(labels, half, path, expected=199319):
    import numpy as np
    labels, half = np.asarray(labels), np.asarray(half)
    if labels.shape != (expected,) or half.shape != labels.shape:
        raise ValueError(f'Expected {expected} predictions and half markers')
    if not np.isin(labels, [0, 1, 2]).all():
        raise ValueError('Expected numeric H/A/D labels (0/1/2)')
    letters = np.array(['H', 'A', 'D'])[labels.astype(int)]
    letters[half == 0] = 'X'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        handle.write('frame,label\n')
        handle.writelines(f'{i},{label}\n' for i, label in enumerate(letters))
    print(f'{path}: {len(letters)} frames')


def mall(mode, output, reuse_cache=False):
    import numpy as np
    from .mall.common import CACHE, cache_path, load_layer
    from .mall.layer4_possession_model.run import bench_tags
    if mode in ('model', 'interaction') and not set(BENCH_TAGS).issubset(bench_tags()):
        raise ValueError('Mall model needs all ten bench games. Run python -m pipeline.run prepare-mall --bench-src PATH first, or use --mode events.')
    # Rebuild match stages on every run; do not silently reuse stale caches.
    cache_dir = Path(CACHE)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not reuse_cache:
        run_module('pipeline.mall.layer0_clean.run')
        for layer in MALL_LAYERS:
            run_module(f'pipeline.mall.{layer}.run')
    if mode in ('model', 'interaction'):
        # Layer 1 was rebuilt above; its cached design must be rebuilt too.
        if not reuse_cache:
            (cache_dir / 'models/design_tactical.npz').unlink(missing_ok=True)
            run_module('pipeline.mall.layer4_possession_model.run')
            run_module('pipeline.mall.layer5_sequence_decoder.run')
        if mode == 'interaction':
            from .mall.interaction.model import TRAIN_GAMES, predict, save_diagnostics, train
            trained = train(TRAIN_GAMES)
            result, events, legacy = predict('tactical', trained)
            np.savez_compressed(cache_path('layer6_interaction'), **result)
            output.parent.mkdir(parents=True, exist_ok=True)
            save_diagnostics(output.with_suffix('.interactions.json'), 'tactical', trained,
                             result, events, legacy)
            labels = result['label']
        else:
            labels = load_layer('layer5')['label']
    else:
        from .mall import config as config
        from .mall.layer5_sequence_decoder.decoder import decode
        # Explicit ablation: retain Mall's event decoder and disable learned evidence.
        config.SEQ_W_FULL = config.SEQ_W_PLAYERS = config.SEQ_W_CP = 0.0
        z = load_layer('layer0')
        labels = decode(z, load_layer('layer2'), load_layer('layer3'),
                        {'p_home_players': np.full(len(z['half']), .5)})
    write_csv(labels, load_layer('layer0')['half'], output)


def event_first(output, reuse_cache=False):
    from .event_first.run import build
    from .mall.common import CACHE
    if not reuse_cache:
        run_module('pipeline.mall.layer0_clean.run')
        for layer in MALL_LAYERS:
            run_module(f'pipeline.mall.{layer}.run')
    labels, half, trace = build(diagnostics=Path(CACHE).parent / 'event_first/tactical.events.json')
    write_csv(labels, half, output)
    print(f"Event-first: {len(trace['events'])} events; {len(trace['decisions'])} ownership decisions; no learned possession evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='pipeline', required=True)
    for name, choices, default in [('lenny', ['nearest', 'dead', 'decoder', 'model'], 'model'),
                                   ('mall', ['events', 'model', 'interaction'], 'model')]:
        p = subs.add_parser(name)
        p.add_argument('--mode', choices=choices, default=default)
        p.add_argument('--output', type=Path)
        p.add_argument('--data-dir', type=Path, help='Data bundle root or tactical folder')
        if name == 'mall':
            p.add_argument('--reuse-cache', action='store_true',
                           help='Interaction mode only: explicitly reuse existing Mall tracking/model caches')
    p = subs.add_parser('prepare-mall', help='Build the ten training games and their features')
    p.add_argument('--bench-src', type=Path, required=True, help='Local metrica-sports/sample-data checkout')
    p = subs.add_parser('event-first', help='Independent contact/recovery ownership state machine')
    p.add_argument('--output', type=Path)
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--reuse-cache', action='store_true', help='Explicitly reuse existing tracking/features/dead-ball caches')
    p = subs.add_parser('paper', help='Reconstruct the published 2022 possession/event algorithm')
    p.add_argument('--output', type=Path)
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--reuse-cache', action='store_true', help='Reuse raw tracking arrays already parsed into layer 0')
    p.add_argument('--profile', choices=['a', 'bc'], default='bc', help='Published provider parameter profiles (BC by default; our provider is not identified as B/C)')
    p.add_argument('--status', choices=['missing', 'provided', 'mall'], default='missing', help='Explicit game-status source; missing follows the paper feed convention')
    p.add_argument('--status-file', type=Path, help='Full frame,in_play binary CSV for --status provided')
    p = subs.add_parser('touches', help='Learn player touches; switch immediately to each touching team')
    p.add_argument('--bench-src', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--reuse-cache', action='store_true', help='Reuse parsed raw tracking arrays only')
    p.add_argument('--skip-evaluation', action='store_true', help='Skip held-out sample-game evaluation')
    args = parser.parse_args()
    if args.pipeline == 'mall' and args.reuse_cache and args.mode != 'interaction':
        parser.error('--reuse-cache is only supported with mall --mode interaction')
    os.environ['POSSESSION_GAME'] = 'tactical'
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
    if args.pipeline == 'prepare-mall':
        os.environ['BENCH_SRC'] = str(args.bench_src.resolve())
        # Three Metrica games (local checkout) and seven open DFL games (downloaded once).
        subprocess.run([sys.executable, '-m', 'pipeline.mall.bench.idsse', '--download'],
                       cwd=ROOT, check=True)
        subprocess.run([sys.executable, '-m', 'pipeline.mall.bench.build', '--games', '1', '2', '3',
                        '--dfl', '1', '2', '3', '4', '5', '6', '7'], cwd=ROOT, env=dict(os.environ),
                       check=True)
        # Training reads the bench games' features and answer keys only (layers 1-2); their
        # own dead-ball layer is run by bench/score.py when they are scored.
        for tag in BENCH_TAGS:
            for layer in MALL_LAYERS[:2]:
                run_module(f'pipeline.mall.{layer}.run', tag)
        from .mall.common import CACHE
        model_dir = Path(CACHE) / 'models'
        for pattern in ('design_*.npz', 'deadp_design_*.npz', 'frame_*.pkl', 'cp_*.pkl', 'deadp_*.pkl'):
            for path in model_dir.glob(pattern):
                path.unlink()
        return
    if args.data_dir:
        data = args.data_dir.resolve()
        os.environ['MTAH_DATA_DIR'] = str(data.parent if data.name == 'tactical' else data)
    if args.pipeline == 'event-first':
        output = (args.output or ROOT / 'possession-viewer/data/event-first.csv').resolve()
        event_first(output, args.reuse_cache)
        return
    if args.pipeline == 'touches':
        if not args.reuse_cache:
            run_module('pipeline.mall.layer0_clean.run')
        from .touches.run import build
        output = (args.output or ROOT / 'possession-viewer/data/touches.csv').resolve()
        build(args.bench_src.resolve(), output, not args.skip_evaluation)
        return
    if args.pipeline == 'paper':
        if args.status == 'provided' and not args.status_file:
            parser.error('--status provided requires --status-file')
        if args.status_file and args.status != 'provided':
            parser.error('--status-file requires --status provided')
        if not args.reuse_cache:
            run_module('pipeline.mall.layer0_clean.run')
        from .paper2022.run import tactical
        output = (args.output or ROOT / 'possession-viewer/data/paper.csv').resolve()
        tactical(output, args.profile, args.status, args.status_file)
        return
    output = (args.output or ROOT / f'possession-viewer/data/{args.pipeline}-{args.mode}.csv').resolve()
    if args.pipeline == 'mall':
        mall(args.mode, output, args.reuse_cache)
    else:
        from .lenny import config
        config.OUTPUT = output
        # The original writer captures OUTPUT at import; set it before importing.
        module = importlib.import_module('.lenny.run_all', __package__)
        sys.argv = [sys.argv[0], '--stage', args.mode]
        module.main()


if __name__ == '__main__':
    main()
