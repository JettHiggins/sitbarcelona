# Mall possession improvements — 11 November 2026

> Integration status: the supplied prediction CSV is available in the viewer.
> Several inference dependencies were absent from the source branch. Read
> [INTEGRATION.md](INTEGRATION.md) before using the historical commands below;
> `pipeline.run mall --mode improved` is not available in this checkout.

The current recommended candidate is **`C:\Users\mallw\Documents\mtah\possession-viewer\data\mall-referee-supervised.csv`**. The default `run_mall.ps1` and `pipeline.run mall --mode improved` now use this candidate. Older candidates remain available for comparison.

## Current result on Barcelona's public five-minute clip

| Metric | Original hysteresis | Earlier calibration | Supervised motion |
|---|---:|---:|---:|
| Spell F1 | 0.537500 | 0.550000 | **0.586667** |
| Macro F1 | 0.850348 | 0.851075 | **0.882565** |
| Frame accuracy | 0.864933 | 0.865733 | **0.892000** |
| Dead-ball F1 | 0.787333 | — | **0.845414** |

This is +4.92 percentage points of spell F1 and +3.22 points of macro F1 versus the hysteresis baseline. The output has 13 predicted live-possession spells in this clip, compared with 15 previously and 17 reference spells. The score increase reflects fewer spurious predicted spells and one additional match at the strictest evaluated overlap threshold. It does not mean all reference spells are recovered.

## What the selected model changes

A supervised dead-ball model sees main-referee speed, the slower/faster assistant speeds, and player movement summaries. It learns from official live/dead labels in the first halves of two public IDSSE matches. Source referee tracks are sampled at 5 Hz to match the existing Barcelona vision-derived motion cache. Missing-referee augmentation uses unlabeled target visibility patterns. It recognizes movement patterns, **not whistle-blowing, advantage gestures, or arm signals**.

The model has 60 features, 120 boosting iterations, at most 15 leaves per tree and a minimum leaf size of 100. Six evidence-blend settings were compared using leave-one-match-out predictions on the external first halves. The selected setting combines 50% original dead-ball evidence with 50% learned evidence scaled by 0.1. The final model trains only on those first halves; second-half labels are excluded.

The control radius remains **1.5 m**. Possession decoder switch cost stays **4.0**, duration weight **0.30**, hysteresis strength **0.30**, and decay **25 frames**. The gain comes from better stoppage evidence, not a more complicated Preisach model.

## Rechecks and limits

On the external second halves, mean spell F1 improved from 0.614921 to 0.622269. A matched player-only model scored 0.620773, so referee motion adds a small benefit beyond player movement alone. The paired block-bootstrap interval for the baseline-to-motion spell gain is **−0.0086 to +0.0240**. This crosses zero: the external gain is not statistically established.

Barcelona has only 17 public reference spells and has been checked repeatedly. No target possession labels were used to fit this model or select its numeric settings, but the final recommendation includes target sanity-check results. This is a practically better candidate on the available clip, **not proof of a global optimum or full-match improvement**.

Other new candidates were tested and rejected for Barcelona: clean IDSSE possession/dead-ball models, tracking-gap adaptation, bounded stoppage retiming, and a combination of new possession models with motion evidence. Their logs, frozen settings and scores are preserved. Larger external gains did not reliably transfer to Barcelona.

## Run

From this folder:

```powershell
.\run_mall.ps1
.\run_mall.ps1 -Verify
```

The first writes the recommended CSV in the possession viewer's data folder. The second checks byte-identical replay. `-Candidate calibrated` and `-Candidate robust` retain the earlier options. `run_improved.ps1` writes the same selected model to `mall-improved.csv`; `-Rebuild` also rebuilds the ordinary Mall stages.

Scientific dependencies are copied under `mall-improvement/dependencies`; the runner no longer depends on the generated Codex workspace. It uses the existing Python runtime. The match feature caches and the existing referee motion cache are required.

Evidence: `experiments/expanded_final_report.json`, `experiments/referee_supervised/{frozen,test,ablation,deployment}.json`, and the final replay verification report. Training scripts and downloaded data are all under mtah. The following sections preserve earlier experiment history.

---

# Earlier calibration experiments — 10 November 2026

The best candidate at that earlier stage was `possession-viewer/data/mall-calibrated.csv` in the parent mtah folder. `mall-robust.csv` contains the same predictions: the robust-model comparison selected the same calibration policy. The original prediction files remain available.

## Result on Barcelona's public five-minute check

| Metric | Existing hysteresis baseline | Automatic calibration |
|---|---:|---:|
| Spell F1 | 0.537500 | 0.550000 |
| Macro F1 | 0.850348 | 0.851075 |
| Frame accuracy | 0.864933 | 0.865733 |
| Predicted possession spells | 15 | 15 |
| Reference possession spells | 17 | 17 |

This is a modest improvement, not evidence of a significant general improvement or a global optimum. One additional spell matches at the 0.4 IoU threshold. The match counts at the other four thresholds are unchanged. The frame-accuracy improvement amounts to six frames in this check.

The earlier `mall-radius-tuned.csv` had a higher public-clip score, but its radius was selected using that same clip. It is not an independent validation result and was not used to select this candidate.

## What changed

The current match has larger ball-to-player distances than the public training games. During observed live play, the nearest-player median is about 1.75 m in Barcelona, versus 0.54–0.55 m in two benchmark games. That alone does not prove tracking error, so the calibration was also checked under simulated tracking noise.

For each match, estimate the median nearest-player distance on observed, predicted-live frames where the ball moves at 0.5–6 m/s, the nearest player has relative speed below 2 m/s, and the distance is below 3.5 m. Use only automatic tracking, without reference possession labels. The reference is the median of the corresponding estimates from the external training games.

`control_radius = clip(1.5 + max(0, match_estimate - external_reference), 1.5, 3.0)`

Barcelona's estimate is 0.799024 m and the external reference is 0.127129 m. Its resulting control radius is **2.171895 m**, replacing 1.5 m for control segments and dwell features. Touch events and all ball-related model input columns are regenerated before inference.

The original dead-ball mask is retained exactly. Kick attribution remains at 1.5 m and touch attribution at 2.5 m. Decoder switch cost remains 4.0 and duration-prior weight 0.30. Hysteresis retains its three relays, strength 0.30, and 25-frame decay. This is an inference calibration, not a separately retrained model for every possible radius.

## Approaches tested and rejected

- **Training-label correction.** The earlier hysteresis experiment included dead-ball frames as away possession when training its two-game benchmark models. Corrected models exclude dead and ungraded frames and have content-dependent cache keys. The original three-game model used for Barcelona already had the correct exclusions. All new comparisons used the corrected benchmark baseline.
- **Turnover penalties near stoppages, kick reliability, evidence weights and stronger hysteresis.** No substantial consistent improvement.
- **Learned dead-ball detector.** Reserved external-game second-half spell F1 improved from 0.665386 to 0.725501 and macro F1 from 0.886175 to 0.940341. Its paired block-bootstrap spell gain interval was approximately +0.038 to +0.082. It failed to transfer to Barcelona, where spell F1 fell to 0.462500 and macro F1 to 0.709566. Saved only as `mall-external-deadball.csv`; not recommended.
- **Stronger rewards for short receptions.** Did not beat the selection baseline.
- **Spell-weighted training.** Gave short spells more weight but did not improve the average selection score.
- **Training on clean and noisy tracking.** Did not beat automatic calibration on the combined clean/noisy selection objective.
- **Higher uncertainty quantiles.** The 75th and 90th percentiles did not beat median-based calibration. No additional Barcelona thresholds were selected from these alternatives.

## Evaluation and uncertainty

Models predicting an external game were trained on the other games. First halves selected settings. Second halves checked the frozen choices. These three matches have also been used in previous experiments, so they are not a fresh independent dataset.

The robustness test perturbed ball positions with Gaussian offsets of 0.75 m standard deviation at 50-frame knots, interpolated in time, and updated velocities and features. This is a simulated sensor disturbance, not a claim that Barcelona's tracking follows that distribution. The original dead-ball mask was held fixed to isolate possession robustness.

| External second-half test | Baseline spell F1 | Calibration spell F1 |
|---|---:|---:|
| Original tracking | 0.665386 | 0.666263 |
| New simulated-noise realization | 0.642365 | 0.649943 |

The noisy-test paired five-minute block-bootstrap 95% interval for the spell gain is approximately **−0.0047 to +0.0186**. It crosses zero. The clean gain is also very small. The benchmark answer keys are reconstructed from event data and are not equivalent to corrected frame-level tracking annotations.

Barcelona labels were used for post-selection checks, not model fitting or choosing calibration quantiles. Repeated checks on the same 17 reference spells are still insufficient to establish generalization. Further substantial improvement needs independent validation data with comparable tracking quality. Increasing search size on the existing public clip would not resolve that limitation.

## Run and verify

From this directory in PowerShell:

```powershell
.\run_mall.ps1 -Candidate calibrated
.\run_mall.ps1 -Candidate calibrated -Verify
```

The first command writes to the parent project's `possession-viewer\data\mall-calibrated.csv`. Verification writes a separate replay CSV under `experiments` and checks byte equality against the frozen result. `-Candidate robust` reproduces the same selected predictions under `mall-robust.csv`.

The local runner uses the existing Python runtime and dependency directory. All experiment scripts, copied Mall source, training caches, models, reports and newly generated prediction files are within mtah. Derived data are excluded from version control by this folder's `.gitignore`.

`pipeline.run mall --mode improved --output PATH` is also available after the normal Mall preparation. Prediction requires the prepared tracking caches; no reference possession CSV is read by the inference runner.

## Evidence files

Machine-readable records under `experiments` include:

- `baseline_audit.json` and hashed live-only model metadata.
- `screen.json`, `dead_screen.json`, `joint_refine.json`, `heldout_report.json`.
- `reception_screen.json`, `weighted_screen.json`, `robust_screen.json`.
- `calibration_screen.json`, `calibration_frozen.json`, `robust_frozen.json`.
- `robust_heldout.json`, `calibration_public_check.json`, `robust_deployment.json`.
- `verification.json`, produced after successful replay and CSV contract checks.

Scripts and logs preserve both successful and rejected approaches. No hand-edited possession labels or frame-specific corrections were applied.

## Expanded external-data experiment (November 10)

With permission to broaden research, downloaded all seven IDSSE matches (21 XML files, about 2.63 GB) into `external/idsse`. No FAQ.md was read. The source is the public PySport mirror of the IDSSE dataset, pinned to commit `a715a38dfbaf5f58e431727c2b78d174101a703c`; `source_manifest.json` and `downloads.json` record provenance and verified download hashes. Dataset paper: https://doi.org/10.1038/s41597-025-04505-y (DFL, CC BY 4.0).

The adapter checks 25 Hz timing, maps coordinates to Mall pitch coordinates, separates official BallStatus/BallPossession labels from features, and leaves the halftime gap ungraded. All seven matches have possession labels for every live frame. Referee positions are present only in J03WMX and J03WN1. These are position tracks, not gesture annotations or licensed video.

Training: J03WOH, J03WOY, J03WPY, J03WQQ. Setting selection: J03WR9. Reserved testing: J03WMX and J03WN1. The four training games supply approximately 3.5 hours of labeled live play. Home/away swapping remains in frame-model training. Source event tags are converted into simulated broad SmartTag windows; these are not actual challenge SmartTag annotations.

The first experiment retrained possession and dead-ball classifiers. Validation selected the new possession model and a 50% dead-ball evidence blend with learned evidence scale 0.5. Radius stayed 1.5 m; hysteresis and the possession decoder remained unchanged.

| Evaluation | Baseline spell F1 | Selected spell F1 | Baseline macro F1 | Selected macro F1 |
|---|---:|---:|---:|---:|
| Selection match J03WR9 | 0.511675 | 0.657329 | 0.809331 | 0.870919 |
| Reserved J03WMX | 0.598063 | 0.653547 | 0.909350 | 0.943814 |
| Reserved J03WN1 | 0.483450 | 0.663955 | 0.837305 | 0.925188 |
| Barcelona public sanity check | 0.537500 | 0.522222 | 0.850348 | 0.680135 |

The reserved-match mean spell gain was about 0.1180, with a paired five-minute block-bootstrap interval of +0.0905 to +0.1478. This is conditional on two matches and does not establish target generalization. Barcelona dead-ball recall fell from 0.8632 to 0.2491. Therefore **mall-idsse.csv is an experimental, rejected target candidate**, not a replacement for mall-calibrated.csv. It was replayed byte-identically with reference-label reading disabled.

The next experiment (`idsse_adapt.py`) addresses the tracking mismatch: raw Barcelona ball availability is 59.3%, compared with nearly continuous IDSSE ball tracks. It independently samples blocks of target availability and adds 0.75 m temporally correlated position noise to external matches, retaining external labels. Clean and perturbed external validation conditions choose settings; no Barcelona reference labels enter training or selection. The original two test matches are reused for this follow-up and are no longer a fresh independent dataset. Results are recorded under `experiments/idsse_adapt/experiment`.
