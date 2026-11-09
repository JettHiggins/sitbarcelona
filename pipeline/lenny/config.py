"""All algorithm settings. Chosen from physical rules, never public labels."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHALLENGE = Path(os.environ.get('MTAH_CHALLENGE_DIR', ROOT / 'pipeline/vendor/challenge'))
DATA = Path(os.environ.get('MTAH_DATA_DIR', ROOT / 'data'))
if DATA.name == 'tactical':
    DATA = DATA.parent
CACHE = ROOT / 'pipeline/cache/lenny'
OUTPUT = ROOT / 'possession-viewer/data/lenny-model.csv'
SEED = 2027
FPS = 25
N = 199319
PITCH = (105.3, 68.0)
HALVES = ((11185, 90652), (110808, 190198))
MAX_BALL_SPEED = 40.
MAX_GAP = 25  # strictly shorter than one second
SMOOTH_WINDOW, SMOOTH_ORDER = 7, 2
REACH, NEAR = 1.5, 2.
RECEPTION_FRAMES, CERTAIN_FRAMES = 8, 25
COMOTION = 6.
KICK_ACCEL, KICK_TURN, FAST_BALL = 4., 0.8, 10.
GLITCH_TURN = 2.5
MIN_DEAD, MIN_PLAY = 75, 50
SLOW_PLAYERS, REST_BALL = 1., 0.8
DEAD_THRESHOLD = 2.2
TAG_WEIGHT, OUT_WEIGHT, SLOW_WEIGHT, SPOT_WEIGHT, FORMATION_WEIGHT = 1.5, 2., 0.7, 1., 1.
RESTART_RADIUS, CENTRE_RADIUS = 5., 3.
MODEL_WINDOWS = (75, 250, 750)
MODEL_BLOCK = 7500
MODEL_PURGE = 750
TRAIN_STRIDE = 10
MODEL_ITER, MODEL_LEAVES, MODEL_RATE = 100, 15, 0.07
SWITCH_COST, RECEPTION_REWARD, KICK_REWARD = 4., 7., 2.
MIN_POSSESSION, SHORT_PENALTY = 50, 4.
MODEL_WEIGHT, CONTROL_WEIGHT = 0.12, 2.

def initialize():
    import random
    import sys
    import numpy as np
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ['OMP_NUM_THREADS'] = '1'
    sys.path.insert(0, str(CHALLENGE))
    CACHE.mkdir(parents=True, exist_ok=True)

def runs(values):
    import numpy as np
    edges = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
    return zip(edges[:-1], edges[1:])
