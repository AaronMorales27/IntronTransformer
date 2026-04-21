# Genomic Foundation Model Notes and Experiments

## Why this exists

This repo is my running lab notebook for understanding genomic foundation models by actually using them end-to-end, not just reading summaries or papers.

The goal is to keep both intuition and execution in the same place:

1. build concepts I can explain in plain language,
2. run experiments I can reproduce from scratch,
3. track what changed, what improved, and what still feels uncertain.

This is not meant to be a polished benchmark or production reference.
It is a structured record of understanding:
smoke tests -> frozen baselines -> threshold calibration -> probing -> partial unfreeze.

## Quickstart (first successful run)

From repo root:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install torch transformers scikit-learn numpy pyyaml
python3 scripts/prepare_data.py --out data/processed/promoters_demo.csv --n_total 200 --seed 42
python3 scripts/train_classifier.py --config configs/default.yaml
```

If the config path is not ready in your environment, run the script directly:

```bash
python3 scripts/train_classifier.py \
  --data_csv data/processed/promoters_demo.csv \
  --model_id InstaDeepAI/nucleotide-transformer-v2-50m-multi-species \
  --epochs 3 \
  --batch_size 8 \
  --lr 1e-3 \
  --max_length 256 \
  --seed 42 \
  --device auto \
  --out_csv results/metrics_baseline_vs_pretrained.csv
```

## How to read outcomes

Two metrics are doing different jobs, so I treat them differently:

- `ROC-AUC` tells me whether positives are ranked above negatives across thresholds.
- `F1` and `accuracy` tell me how good hard decisions are at one chosen threshold.

This matters because I can see a run where pretrained has stronger ranking (`ROC-AUC`) but weaker default-threshold (`0.5`) F1.
That does not automatically mean the representation is worse; it can mean calibration is off for this task/data split.

So I use validation threshold selection (`--sweep_thresholds`) as a calibration step:

- choose threshold on validation (e.g., best val F1),
- lock that threshold,
- report test metrics under that fixed choice.

What this tells me:

- representation quality and decision calibration are related but not identical,
- fixed threshold comparisons can hide useful pretrained signal,
- threshold tuning should be documented clearly so results stay interpretable.

## Results at a glance (splice task)

- Frozen pretrained beats frozen random on test ROC-AUC (`0.823787` vs `0.521263`) in `results/splice_metrics_baseline_vs_pretrained.csv`.
- Moving from frozen pretrained to partial unfreeze top-1 raises test ROC-AUC from `0.823787` to `0.967238`.
- Partial unfreeze top-4 also improves strongly (`0.956999` ROC-AUC), but in this run top-1 is slightly better than top-4.
- Short version: pretraining helps even when frozen, and limited top-layer adaptation helps even more.

## Limitations

- Early runs are small-scale and sometimes synthetic, useful for method intuition but not strong biological claims by themselves.
- Validation sets are limited, so selected thresholds can be noisy and seed-sensitive.
- Frozen-head and probe experiments show decodability/readability of signal, not full task-optimal usage.
- Real task conclusions should prioritize leakage-safe splits and multi-seed reporting before any strong claim.

## Documentation map

Public-facing summaries (recommended first):
- Story overview: `docs/story.md`
- Results overview (including synthetic -> splice transition): `docs/results.md`

Internal working notes (for my own process tracking):
- `planning/` contains day-by-day findings and rough quickstarts.

