# Route 1: train on tracking that looks like the match's

Models trained on clean outside games gain a lot on other clean games and fail on the challenge
match, because the match's ball track is less precise and disappears during stoppages. Here the
outside games (seven public DFL matches) are degraded to the statistics measured on the match's
automatic tracking (`tracking_diagnostics_report.txt`), and the models are trained on that.

| file | what it does |
|---|---|
| `degrade.py` | hides the ball during stoppages, adds a slow drift, smooths the ball, applies the measured offset and stretch, drops players out |
| `stats.py` | the diagnostics' summary numbers on any game, used to calibrate `degrade.py` |
| `prepare.py` | builds layers 1-3 and model inputs for each outside game, clean and degraded |
| `experiment.py` | trains frame, change-point and dead-ball models on four degraded games; picks the blend on a fifth; tests on two more |
| `predict.py` | inference on the challenge match; writes `possession-viewer/data/mall-matched.csv` |
| `run/` | trained models, frozen setting, validation and test scores |

No reference label of the challenge match is read anywhere. Only its diagnostics report (aggregate
tracking statistics) informed the degradation.

## Results on held-out degraded games (spellF1 / macroF1)

| game | current pipeline | models trained on clean outside games | route 1 |
|---|---|---|---|
| J03WR9 (selection) | 0.470 / 0.779 | 0.483 / 0.687 | 0.594 / 0.841 |
| J03WMX (test) | 0.555 / 0.885 | 0.493 / 0.727 | 0.605 / 0.922 |
| J03WN1 (test) | 0.418 / 0.796 | 0.447 / 0.649 | 0.582 / 0.885 |

Frozen setting: route-1 frame model, route-1 change-point model, dead-ball evidence fully from the
route-1 model at weight 0.1. Decoder settings unchanged.

Limits: the degradation is a simulation; it reproduces the match's macroF1 collapse for the
clean-trained models (0.687 simulated, 0.680 on the public clip) but is milder on stoppages. The
two test games were used in earlier experiments. The referee-motion evidence is not included.
