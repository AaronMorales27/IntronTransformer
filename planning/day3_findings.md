# Day 3 Findings (Baseline + Threshold Sweep)

## Goal for Day 3
- Build intuition with a minimal downstream classifier setup:
  - `pretrained_frozen_head_only`
  - `random_frozen_head_only`
- Compare both on `accuracy`, `F1`, and `ROC-AUC`.
- Keep the experiment simple and interpretable before any fine-tuning.

Summary:
Day 3 was about isolating representation quality with the simplest possible downstream setup before adding complexity.

## Questions I focused on (this chat + forked thread)
- What does **frozen backbone + trainable head only** actually mean?
- Why can `ROC-AUC` look good while `F1`/accuracy look bad?
- Can we pick a better threshold than `0.5` from validation?
- Is this enough to move to Day 4 pipeline cleanup?

Summary:
Most of the work centered on understanding geometry vs threshold effects, then deciding if the pipeline was mature enough to refactor.

## What I implemented
- `scripts/train_classifier.py` now:
  - loads split CSV data (`train/val/test`)
  - builds a frozen transformer backbone
  - trains only a linear head on sequence embeddings
  - runs pretrained vs random-init comparison
  - logs metrics to CSV
- Added threshold sweep option:
  - `--sweep_thresholds`
  - sweeps thresholds on validation probabilities to maximize validation F1
  - applies selected threshold to both val and test
  - writes `selected_threshold` and `threshold_source` to CSV

Summary:
The script now supports both a fixed-threshold baseline and a calibrated-threshold evaluation path with explicit traceability in output CSV.

## What I observed
### Before threshold sweep (`seed=42`)
- Pretrained had stronger ranking (`ROC-AUC`) but weak thresholded metrics:
  - val AUC high, but val F1/accuracy low at default threshold `0.5`
- Random looked better on F1/accuracy in that run, which suggested threshold/calibration effects.

Summary:
At threshold 0.5, pretrained signal existed but was not converted into good hard class decisions.

### After threshold sweep (`seed=42`)
- Selected thresholds:
  - pretrained: `0.43`
  - random: `0.05`
- Pretrained F1 and accuracy improved substantially after threshold selection.
- Core insight: a fixed `0.5` threshold hid useful pretrained signal.

Summary:
Validation-based threshold selection unlocked better downstream behavior, especially for pretrained.

## Interpretation
- With frozen backbones, this setup behaves like a probe of representation quality.
- `ROC-AUC` reflects ranking quality (threshold-free).
- `F1`/accuracy reflect hard decisions (threshold-dependent).
- Therefore, threshold tuning is not "cheating" here; it is necessary calibration for fair comparison.

Summary:
AUC answers "are examples ranked well?", while F1/accuracy answer "did we pick a good cutoff for decisions?"

## Limitations (important)
- Dataset is tiny and synthetic, so metrics can vary a lot across seeds.
- Validation set is small; selected threshold may overfit validation.
- This is an intuition experiment, not final biological performance.

Summary:
This is strong for learning and method development, but not yet strong for biological claims.

## Decision for next step
- Day 3 objective is met: reproducible baseline + calibration insight.
- Next logical move is Day 4:
  - move training logic into `src/pipeline/*`
  - add `configs/default.yaml`
  - keep one-command reproducibility in `README.md`

Summary:
We should now prioritize structure and reproducibility cleanup (Day 4), not new modeling complexity yet.

