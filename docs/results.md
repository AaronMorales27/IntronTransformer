# Results Overview (Splice First)

This is the public-facing summary of the splice-site results.
Internal day-by-day notes stay in `planning/`.

## Why I switched from synthetic to splice data

I used synthetic promoter-style data first to debug the pipeline quickly (tokenization, training loop, metrics, threshold calibration).
After that machinery was stable, I moved to published splice-site data so the comparisons would be grounded in a real annotated task.

So the synthetic phase was for method validation.
The splice phase is the main evidence.

## Headline comparison 1: pretrained frozen vs random frozen

Source: `results/splice_metrics_baseline_vs_pretrained.csv` (seed 42, val-threshold sweep)

- **Pretrained frozen (test):**
  - accuracy: `0.713333`
  - F1: `0.751013`
  - ROC-AUC: `0.823787`
- **Random frozen (test):**
  - accuracy: `0.493667`
  - F1: `0.660407`
  - ROC-AUC: `0.521263`

How I read this:
- The pretrained backbone is doing meaningful representational work even when frozen.
- The strongest gap is on ROC-AUC, which supports better ranking quality than random frozen.

## Headline comparison 2: frozen vs partial unfreeze (top layers)

Sources:
- frozen baseline: `results/splice_metrics_baseline_vs_pretrained.csv`
- partial unfreeze top-1: `results/splice_partial_unfreeze_top1.csv`
- partial unfreeze top-4: `results/splice_partial_unfreeze_top4_combined.csv`

Test split comparison (pretrained runs):

- **Frozen head-only**
  - accuracy: `0.713333`
  - F1: `0.751013`
  - ROC-AUC: `0.823787`
- **Partial unfreeze top-1**
  - accuracy: `0.906667`
  - F1: `0.906292`
  - ROC-AUC: `0.967238`
- **Partial unfreeze top-4**
  - accuracy: `0.892667`
  - F1: `0.894357`
  - ROC-AUC: `0.956999`

How I read this:
- Allowing adaptation of top transformer layers gives a large gain over frozen-only.
- In this run, top-1 is slightly stronger than top-4, which suggests "more unfreezing" is not automatically better.
- This motivates a ladder search instead of assuming monotonic improvement with deeper unfreezing.

## Biological interpretation of ROC-AUC around 0.95

When test ROC-AUC is around `0.95` on splice-site classification, a practical interpretation is:
- the model is usually ranking true splice examples above non-splice examples across many possible thresholds,
- so the representation has captured sequence patterns that are strongly associated with splice identity.

What it is likely picking up:
- canonical splice-related motifs and local context statistics (for example donor-like core signals and flanking composition),
- dependencies between nearby positions that are hard to capture with purely local heuristics.

What this does **not** prove by itself:
- that the model learned a full causal mechanism of splicing,
- that performance will hold unchanged across all species/tissues/data-generation pipelines,
- that every high-confidence prediction is biologically correct without external validation.

So I treat ROC-AUC ~0.95 as strong evidence of discriminative sequence signal capture on this dataset, not as final biological truth.

## Notes on confidence and scope

- These are strong directional results, but still from limited seed coverage in the splice runs shown here.
- Best next validation step is repeating top settings across multiple seeds and reporting mean/std.
- Threshold selection is val-based (`val_f1_sweep`) and then locked on test, which keeps the evaluation logic explicit.

## Artifact index

- Main splice baseline comparison:
  - `results/splice_metrics_baseline_vs_pretrained.csv`
- Partial unfreeze runs:
  - `results/splice_partial_unfreeze_top1.csv`
  - `results/splice_partial_unfreeze_top4_combined.csv`
- Optional smoke reference:
  - `results/splice_smoke_metrics_batch=8.csv`
