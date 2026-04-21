# Day 6B Partial Unfreeze Quickstart

This runbook extends the existing splice donor workflow with a **partial unfreeze ladder**.
It is written as a first-time guide.

## What changed

The runner now supports three transfer-learning controls:

- `tune_mode`:
  - `frozen_head_only`: freeze backbone, train only linear head (existing behavior)
  - `partial_unfreeze`: train head + top `unfreeze_top_n` transformer blocks
- `unfreeze_top_n`: number of top backbone blocks to train when using `partial_unfreeze`
- `ladder_top_n`: comma-separated list like `"1,2,4"` to run a full ladder in one command

## Intuition (why this helps)

1. **Frozen head-only** asks: "Are pretrained representations already useful?"
2. **Partial unfreeze** asks: "If I let top layers adapt to splice donor data, do metrics improve?"
3. **Ladder (`1,2,4`)** finds the best tradeoff between:
   - stronger adaptation (higher N)
   - training stability / overfitting risk
   - runtime

## Files involved

- Data prep: `scripts/prepare_splice_data.py`
- Training runner: `scripts/train_classifier.py`
- Main configs:
  - `configs/splice.yaml` (single run baseline/full run)
  - `configs/splice_smoke.yaml` (small smoke run)
  - `configs/splice_ladder.yaml` (frozen + top-1/top-2/top-4 + random anchor)

## Step-by-step workflow

### Step 1 — Prepare data once

```bash
conda activate curvebio_nt
cd /Users/aaronmorales/Curvebio
python scripts/prepare_splice_data.py
```

Expected output file:

- `data/processed/splice_donors_nt_revised.csv`

### Step 2 — Optional smoke sanity check

```bash
python scripts/prepare_splice_data.py --smoke
python scripts/train_classifier.py --config configs/splice_smoke.yaml
```

Use this to validate env/device/metrics quickly before full compute.

### Step 3 — Run frozen baseline (single mode)

```bash
python scripts/train_classifier.py --config configs/splice.yaml
```

This produces:

- `results/splice_metrics_baseline_vs_pretrained.csv`

### Step 4 — Run a single top-N partial unfreeze (no baseline rerun)

If you already completed frozen baseline, run only the next experiment:

```bash
python scripts/train_classifier.py \
  --config configs/splice.yaml \
  --tune_mode partial_unfreeze \
  --unfreeze_top_n 1 \
  --out_csv results/splice_partial_unfreeze_top1.csv
```

By default this runs **pretrained top-N only**. It does not rerun frozen baseline
or random anchors.

To include random anchor in the same call:

```bash
python scripts/train_classifier.py \
  --config configs/splice.yaml \
  --tune_mode partial_unfreeze \
  --unfreeze_top_n 1 \
  --include_random_anchor \
  --out_csv results/splice_partial_unfreeze_top1_with_random.csv
```

### Step 5 — Run the full partial-unfreeze ladder

```bash
python scripts/train_classifier.py --config configs/splice_ladder.yaml
```

This one command runs, in order:

1. `pretrained_frozen_head_only`
2. `pretrained_partial_unfreeze_top1`
3. `pretrained_partial_unfreeze_top2`
4. `pretrained_partial_unfreeze_top4`
5. `random_frozen_head_only` (sanity anchor; enabled in ladder config)

Output:

- `results/splice_metrics_comparison.csv`

## How to read the output

For each run, compare `val` and `test` on:

- `roc_auc` (primary ranking metric)
- `f1` (threshold-dependent quality)
- `accuracy` (helpful, but less robust than AUC)

Also check:

- `selected_threshold`
- `threshold_source` (should be `val_f1_sweep` when enabled)
- `notes` (records mode and unfreeze setting)

## Common first-pass settings

- Fast scan: `epochs: 1`, `ladder_top_n: "1,2,4"`
- Confirmed best setting: rerun with `epochs: 2` or `3`
- Stability: rerun best 1-2 settings with 2-3 seeds

## Troubleshooting tips

- If training looks "stuck", runner now prints batch heartbeats.
- If memory is tight, lower `batch_size`.
- If overfitting appears (val up, test flat/down), reduce `unfreeze_top_n` or epochs.
