"""Run an isolated pipeline and export predictions for the possession viewer."""
import argparse
import importlib
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MALL_LAYERS = ('layer1_features', 'layer2_touches', 'layer3_dead_ball')


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
    if mode in ('model', 'interaction') and not {'bench1', 'bench2', 'bench3'}.issubset(bench_tags()):
        raise ValueError('Mall model needs all three bench games. Run python -m pipeline.run prepare-mall --bench-src PATH first, or use --mode events.')
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
    p = subs.add_parser('prepare-mall', help='Build all three training games and their features')
    p.add_argument('--bench-src', type=Path, required=True, help='Local metrica-sports/sample-data checkout')
    p = subs.add_parser('event-first', help='Independent contact/recovery ownership state machine')
    p.add_argument('--output', type=Path)
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--reuse-cache', action='store_true', help='Explicitly reuse existing tracking/features/dead-ball caches')
    args = parser.parse_args()
    if args.pipeline == 'mall' and args.reuse_cache and args.mode != 'interaction':
        parser.error('--reuse-cache is only supported with mall --mode interaction')
    os.environ['POSSESSION_GAME'] = 'tactical'
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
    if args.pipeline == 'prepare-mall':
        os.environ['BENCH_SRC'] = str(args.bench_src.resolve())
        run_module('pipeline.mall.bench.build')
        for tag in ('bench1', 'bench2', 'bench3'):
            for layer in MALL_LAYERS:
                run_module(f'pipeline.mall.{layer}.run', tag)
        from .mall.common import CACHE
        model_dir = Path(CACHE) / 'models'
        for pattern in ('design_*.npz', 'frame_*.pkl', 'cp_*.pkl'):
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
