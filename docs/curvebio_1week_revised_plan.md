# Curve Bio AI Research Intern - 1 Week Revised Plan

## Goal
Build interview-ready intuition for genomic foundation models and complete one compact, reproducible experiment pipeline you can confidently discuss.

## Time Budget
- Total: ~10 hours
- Cadence: 7 days
- Target output: one mini-project repo structure + one results summary + interview talking points

## Deliverables by End of Week
1. A runnable mini-pipeline for genomic sequence classification (pretrained vs random-init baseline).
2. One probing-style analysis figure (layer-wise or embedding-level comparison).
3. One robustness/ablation figure (mutation/noise sensitivity or motif masking).
4. A short technical writeup (1-2 pages) and a 6-8 slide interview deck.
5. A concise prep sheet with answers to common interview questions.

---

## Day-by-Day Plan

## Day 1 (1.5h): Transformer + Genomics Core Intuition
- Read/watch:
  - Illustrated Transformer (focus: Q/K/V, multi-head attention, positional signal).
  - 3Blue1Brown attention video (focus: geometric intuition).
- Write a 1-page note:
  - "VAE latent space vs transformer representations"
  - "Reconstruction objective vs masked language modeling"
  - "Why probing helps interpret representation quality"

Output:
- `docs/intuition_notes.md`

## Day 2 (1.5h): Model Setup and Data Path
- Set up environment (`torch`, `transformers`, `datasets`, `sklearn`, `matplotlib`, `umap-learn`).
- Load one genomic model checkpoint from HuggingFace (DNABERT-2 or Nucleotide Transformer).
- Prepare one simple task dataset:
  - Promoter prediction preferred (or splice classification).
- Verify first forward pass and tokenization.

Output:
- `scripts/smoke_test_model.py`
- `scripts/prepare_data.py`

## Day 3 (1.5h): Baseline + Fine-Tune
- Train/evaluate:
  - Baseline A: random-init classifier.
  - Baseline B: pretrained encoder + classifier head.
- Track metrics:
  - Accuracy, F1, ROC-AUC (if binary).
- Save metrics to CSV.

Output:
- `scripts/train_classifier.py`
- `results/metrics_baseline_vs_pretrained.csv`

## Day 4 (1.5h): Reproducible Pipeline Cleanup
- Convert ad hoc notebook flow into reproducible structure:
  - Config file for model/data/training args.
  - Separate modules for data, model, train, eval.
- Add one-command run instructions.

Output:
- `configs/default.yaml`
- `src/pipeline/*` modules
- `README.md` with reproduce steps

## Day 5 (1.5h): Probing Experiment
- Extract embeddings from multiple hidden layers.
- Train logistic regression probes on frozen embeddings.
- Plot performance vs layer depth.

Output:
- `scripts/run_probing.py`
- `results/fig_probe_layerwise.png`
- `results/probe_scores.csv`

## Day 6 (1.5h): Ablation/Robustness Experiment
- Run one controlled robustness test:
  - Option A: random mutation noise rates (0%, 1%, 5%, 10%).
  - Option B: motif-region masking/shuffling.
- Plot degradation curve and brief interpretation.

Output:
- `scripts/run_ablation.py`
- `results/fig_ablation_robustness.png`

## Day 7 (1.5h): Interview Packaging
- Write a short technical summary:
  - Problem, setup, model choice, metrics, findings, limitations, next steps.
- Build 6-8 slides:
  - Motivation, pipeline, results, probing, ablation, what you learned.
- Prepare concise answers for likely questions:
  - Why pretrained beats random-init?
  - What did probing reveal?
  - What would you test next for biological validity?

Output:
- `docs/technical_summary.md`
- `docs/interview_slides_outline.md`
- `docs/interview_qa.md`

---

## Most Applicable Hands-On Experiments (Priority Order)
1. Pretrained vs random-init comparison on one genomics task.
2. Layer-wise probing of embeddings for interpretable signal localization.
3. Mutation/masking robustness ablation to test reliance on sequence features.

If time gets tight, finish in this order and skip extras.

## Minimal "Good Enough" Scope (If You Only Have ~6-7 Hours)
- Complete Days 1-4 only.
- Run one short training pass and produce one comparison table.
- Write a 1-page summary and 3-4 strong interview stories from your process.

## Stretch Scope (If You Have Extra Time)
- Add attention heatmap visualization for a known motif-containing sequence.
- Add second dataset task to test generalization.
- Add simple experiment tracker logging (CSV is enough; W&B optional).

---

## Interview Framing Script (Use This Structure)
- Context: "I built a compact benchmark to test how genomic foundation model representations transfer to a disease-relevant sequence task."
- Method: "I compared random-init vs pretrained, then used probing and ablation to inspect what the model learned."
- Result: "Pretraining improved downstream performance and robustness; probing suggested signal concentrated in mid/late layers."
- Reflection: "Next I would validate biological realism with motif-level controls and better negative sampling."

## Weekly Success Criteria
- You can explain attention, pretraining, probing, and ablation clearly in plain language.
- You can walk through your pipeline from data prep to metrics without looking at notes.
- You can defend one design choice and one limitation of your experiment.
- You have files/plots you can show if asked for concrete evidence of work.

