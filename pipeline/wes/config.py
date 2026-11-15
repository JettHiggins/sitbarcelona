"""Settings for the single Wes dead-ball decoder.

Compare changes on permitted other games with `python -m pipeline.wes.evaluate`;
never select values on the BAR–ATH public labels.
"""

MAX_BALL_OUTSIDE_M = 0.5
CONTROL_MAX_CONTEST = 0.25
JOINT_MIN_TRIGGER_FRAMES = 8
JOINT_CONTEXT_FRAMES = 25
JOINT_BOUNDARY_GRID = 5
