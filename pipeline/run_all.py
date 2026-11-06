"""Run from repository root: python -m pipeline.run_all [--stage nearest]."""
import argparse
import warnings
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'pipeline'
from . import config as c

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['nearest','dead','decoder','model'],default='model')
    args=parser.parse_args()
    c.initialize()
    from .layer0_clean import build as clean
    from .layer1_features import build as features, nearest_labels
    from .layer6_output import write
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        track=clean()
        f, distances, nearest=features(track)
        labels=nearest_labels(f)
        if args.stage!='nearest':
            from .layer3_dead_ball import build as dead_ball
            dead,restart,score=dead_ball(track,f)
            labels[dead]='D'
            if args.stage in ('decoder','model'):
                from .layer2_touches import build as touches
                from .layer5_decoder import build as decode
                events=touches(track,f,distances,nearest)
                probability=None
                if args.stage=='model':
                    from .layer4_possession_model import build as model
                    probability=model(track,f,events,dead)
                labels=decode(track,f,events,dead,restart,probability)
        write(labels)

if __name__=='__main__':
    main()
