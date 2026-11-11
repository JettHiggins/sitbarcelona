# Touch-only tracking model

An independent experiment: detect an observable player contact, attribute it to
a player, and immediately assign that player's team. Same-team contacts are
logged too. No possession probabilities, decoder, minimum possession duration,
confirmation pass, retrospective recovery rule, or dead-ball classifier is used.
Unknown until the first accepted touch of each half; hold the last touching team
through missing tracking and stoppages. This is deliberately a different target
from the challenge's control/dead-ball labels. Keep Mall as the stronger current
possession model.

## Research and input fit

| Primary paper | Match to our inputs | What this implementation borrows |
| --- | --- | --- |
| [Biermann et al., 2023, Synchronization of passes in event and spatiotemporal soccer data](https://pmc.ncbi.nlm.nih.gov/articles/PMC10518005/) | 25 Hz, 2D ball and player coordinates: close match | Short player-specific distance windows, polynomial distance features, ball kinematics, supervised event scoring. Their task synchronizes known passes using event timestamps/actors; ours must spot contacts across an entire match without an event feed. We use gradient boosting rather than SwiftEvent. Their pass synchronization results are not all-touch accuracy. |
| [Vidal-Codina et al., 2022, Automatic event detection in football using tracking data](https://link.springer.com/article/10.1007/s12283-022-00381-6) | 25 Hz, 2D coordinates: close match | Proximity plus changes in ball movement as contact evidence. Its control intervals and possession-first event trees are not used here. It does not establish that every physical deflection can be recovered from these inputs. |
| [Wang et al., 2025, Semi-automated last touch detection for out-of-bounds possession decisions in football](https://link.springer.com/article/10.1007/s12283-025-00518-3) | Richer inputs: 50 Hz 3D ball and 29 player joints, instrumented-ball timing supervision | Separate touch timing from attribution; use before/after ball velocity direction and magnitude as features. We cannot reproduce their skeletal distances, aerial-contact features, or sensor supervision with 2D player centres. Their duel decision accuracy does not transfer to full-match touch spotting. |

This is a paper-inspired adaptation, not a faithful reproduction of any of the
three papers. No video, ball height, skeleton, player limbs, or IMU data is used.

## Model

1. Read raw observed ball/player coordinates in metres. Reuse the existing input
   parser and player validity, duplicate and team repair masks. Do not use its
   interpolated ball, possession features, dead-ball labels, or model predictions.
2. Require a complete 25-frame ball window in one half, with no missing sample or
   adjacent jump above 60 m/s. Never bridge a dropout to generate a contact.
3. Score actors within 3 m of the ball using a 25-sample distance window (degree-5
   polynomial, distances, minimum location and local slope), plus before/after
   ball velocity, vector change, cosine similarity and normalized speed change.
   A 0.05 m/s vector-change floor rejects stationary numerical motion; it is not
   a forceful-kick requirement. Small/glancing contacts can still be missed.
4. A HistGradientBoosting classifier scores actor/frame contacts. Accept local
   score peaks >=0.65, requiring a 0.10 margin over other actors. Scores are not
   calibrated real-world probabilities. Suppress peaks only within three frames
   for the same actor; there is no team-switch cooldown.
5. Apply the accepted actor's team at that exact detected frame. The detector is
   offline, using 12 future frames (0.48 s); output timing is its score peak,
   rather than the beginning of a prior proximity/control interval.

Training uses original, undegraded Metrica sample games 1–3: sender contact proxies
at pass/shot/recovery/set-piece starts, receiver proxies at pass ends, deduplicated
by actor/frame. Positive windows are within two frames; timing negatives are
8/12 frames away, and other nearby actors at annotated contacts supply attribution
negatives. Do not treat every unannotated frame as negative. These are still **weak
labels**: pass endpoints can be imprecise; unlogged contacts can contaminate local
negatives; headers, dribbles and deflections are not comprehensively annotated.
Challenge possession truth is never read by training or inference.

## Run and inspect

From the repository root, after installing `pipeline/requirements.txt`:

```bash
git clone https://github.com/metrica-sports/sample-data.git /tmp/metrica-sample-data
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 pipeline/.venv/bin/python -m pipeline.run touches \
  --bench-src /tmp/metrica-sample-data --data-dir /path/to/challenge/data
# If raw tracking has already been parsed, add --reuse-cache.

pipeline/.venv/bin/python -m pipeline.touches.inspect --start 15320 --end 15380
pipeline/.venv/bin/python -m unittest pipeline.touches.test_touches
```

The default run trains three leave-one-game-out models for evaluation, then a
final model on all three games for the challenge match. `--skip-evaluation` skips
only the held-out runs. Parsed raw tracking is the only reused Mall cache.

Outputs (generated locally, ignored by Git):

- `possession-viewer/data/touches.csv`: H/A/X, all 199,319 video frames.
- `pipeline/cache/touches/tactical.touches.json`: accepted/ambiguous contacts,
  actor, team, score, runner-up, distance, impulse, parameters, source checksums.
- `pipeline/cache/touches/tactical.evidence.npz`: every actor/frame score and
  observation, jump, window-validity and motion-candidate masks.
- `pipeline/cache/touches/benchmark.json`: held-out endpoint matching and proposal
  coverage. One-to-one actor matching at 3/12/25 frames (0.12/0.48/1 seconds).
- `pipeline/cache/touches/model.pkl`: local trained model; load only trusted files.

Serve `python3 possession-viewer/server.py`, open http://127.0.0.1:8081/ and choose
`touches.csv`. The prediction API discovers it without a viewer code change.

## Measured results, 2026-11-11

Sample data checkout: `e706dd506b360d69d9d123d5b8026e7294b13996`.
Thresholds/model settings were not selected by maximizing public possession scores.

| Held-out game | Detected contacts | Endpoint proxy precision @3f | Recall @3f | F1 @3f | Proposal coverage @3f |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 2268 | 0.6287 | 0.8144 | 0.7096 | 0.8350 |
| 2 | 2597 | 0.6350 | 0.8328 | 0.7206 | 0.8576 |
| 3 | 3063 | 0.5811 | 0.8073 | 0.6758 | 0.8295 |

Unmatched detections are not automatically false physical touches: event logs are
incomplete for this task. Conversely, matching an endpoint proxy does not prove
correct contact timing on video. Proposal coverage includes quality/radius/motion
gates and is an upper bound before classification, not detector recall.

Barcelona–Athletic: **847 accepted contacts**, 3 ambiguous peak frames; 90,452
frames have complete valid ball windows. Missing windows block new contacts,
while their output team is carried forward. Export has H=80,641, A=77,811,
D=0, X=40,867 (405 initial unknown frames inside the published halves).

| Public 5-minute possession evaluation | Spell F1 | Macro F1 |
| --- | ---: | ---: |
| Touch-only | 0.3429 | 0.4676 |
| Mall | 0.5375 | 0.8503 |

Touch-only team accuracy restricted to ground-truth H/A frames is **76.52%**;
all-class accuracy is 61.53%. Absence of D is deliberate, but the H/A-only errors
also show contact spotting/attribution are insufficient. This is not just a
dead-ball scoring penalty and is not an improvement over Mall.

In the opening sequence it detects BAR at frame 11473 and ATH at 11574/11577,
but misses the earlier BAR deflection and keeper recovery. It accepts no touch
in 15300–15400 or 15720–15830, missing the user's short-flip examples. The scored
evidence permits checking weak scores separately from tracking gaps and actor
ambiguity; do not explain these misses as a possession decoder resisting switches.

Opening-play check: the first half starts at 11185, but the first accepted contact
is not until 11473. Thus **11185–11472 are X: 288 frames / 11.52 seconds**,
including Athletic's opening play. Only the first two ball samples are absent;
152 of the opening 166 frames have valid feature windows. Missing data alone
does not explain this failure. The strongest Athletic actor score in 11185–11350
is 0.262 at frame 11264, below the 0.65 acceptance threshold. This is a missed
contact/generalization problem, not an absent CSV, playback UI issue, or a
possession switch penalty. The first 12 frames after the missing samples are
also ineligible under the complete-window gate, which can lose boundary contacts.

The important remaining validation is a small, separately annotated set of actual
touch times/actors, including negative examples and deflections. With only these
event proxies, physical all-touch precision/recall cannot be established. 2D
proximity also cannot resolve an aerial ball passing above a player's centre.
