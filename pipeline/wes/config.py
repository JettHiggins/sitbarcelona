"""Wes-only experimental settings; Mall's baseline config remains fixed.

Treat these as hypotheses. Compare every change on the three other games with
`python -m pipeline.wes.evaluate`; never select values on BAR-ATH public labels.
"""

MIN_REOPEN_FRAMES = 8
DEAD_SWITCH_COST = 8.0
MAX_BALL_OUTSIDE_M = 0.5
CONTROL_MAX_CONTEST = 0.25
TEAM_CREDIT_WEIGHT = 0.1
TEAM_CREDIT_CAP = 2.0

TOUCH_ATTRIB_RADIUS = 2.5
RECOVERY_LOOKAHEAD_FRAMES = 50
RECOVERY_BLEND = 0.25
CONTACT_CHANGE_PRIOR = 0.25
CONTACT_BONUS_WEIGHT = 1.0
STRONG_CP_PROBABILITY = 0.7
