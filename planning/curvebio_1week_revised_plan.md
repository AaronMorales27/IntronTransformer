# Curve Bio AI Research Intern - 1 Week Revised Plan

## Goal

Build interview-ready intuition for genomic foundation models and complete one compact, reproducible experiment pipeline you can confidently discuss.

## Time Budget

- Total: ~10 hours baseline; **Day 6 splice track adds ~3–5 hours** if you run full data prep + fine-tune ladder + robustness.
- Cadence: 7 days
- Target output: one mini-project repo structure + one results summary + interview talking points

## Plan evolution (do not start a new doc)

- **Days 1–5**: Pipeline validation, synthetic sanity data, probing, and analysis mechanics — already done or in progress.
- **Day 6 onward**: **Track B — real biological task** (splice site classification). Reuse the same `src/pipeline` runner, configs, metrics, and probing; swap **data prep + split policy** and extend training to **non-frozen** fine-tuning where appropriate.
- Keep this single `curvebio_1week_revised_plan.md` as the source of truth; extend days below rather than maintaining a parallel plan file.

## Deliverables by End of Week

1. A runnable mini-pipeline for genomic sequence classification (pretrained vs random-init baseline).
2. One probing-style analysis figure (layer-wise or embedding-level comparison).
3. **Real-data splice track:** prepared CSV + non-leaky splits + metrics CSV comparing frozen vs partial (or full) fine-tune; one robustness figure (splice-context masking preferred).
4. A short technical writeup (1-2 pages) and a 6-8 slide interview deck (**splice-first**, synthetic phase as supporting context).
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

## Day 6 (heavier ~3–5h): Track B — Real splice-site classification

**Why splice (vs other real tasks here):** GENCODE/Ensembl-style splice labels are widely cited, binary-friendly, and match Nucleotide Transformer strengths (local motif + flanking context). Interview story: synthetic plumbing → **real annotated biology** → frozen vs fine-tuned → calibration + one robustness check.

### 6A — Data + splits (~1–1.5h)

- Add `scripts/prepare_splice_data.py` (or equivalent) that emits `sequence,label,split` compatible with existing loaders.
- Labels: donor vs non-donor and/or acceptor vs non-acceptor windows (pick one binary task first to ship).
- **Splits:** chromosome-held-out or transcript-held-out — avoid random row shuffle if the same gene appears in train and test.
- Output: `data/processed/splice_*.csv` + one-line row counts / class balance in logs.

### 6B — Training ladder (~1–2h)

- Reuse `configs/default.yaml` pattern; add `configs/splice.yaml` pointing at the new CSV.
- Run comparison rows to CSV (same schema style as Day 3/4):
  - Frozen backbone + trainable head (baseline).
  - **Partial unfreeze:** top N transformer layers + head.
  - Optional: short full-model fine-tune if VRAM/time allow (document LR and steps).
- Track: accuracy, F1, ROC-AUC; val threshold sweep then locked test evaluation (match probing discipline).

### 6C — Inference + calibration (~30–45m)

- Single-sequence or mini-batch inference script or documented CLI one-liner.
- Save val-selected threshold(s) alongside test metrics for reproducibility.

### 6D — Robustness / ablation (~45m–1h)

- **Motif-context ablation:** mask or shuffle nucleotides in the conserved splice region (e.g. GT/AG core and immediate flank) vs random positions; plot degradation.
- *Original Day 6 idea (random mutation rates)* can stay as **stretch** or move to early Day 7 if splice Day 6 runs long.

Output (target):

- `scripts/prepare_splice_data.py`
- `configs/splice.yaml`
- `results/splice_metrics_comparison.csv` (or extend existing metrics CSV with `task=splice` column)
- `results/splice_robustness.csv` + `results/fig_splice_robustness.png` (or reuse ablation figure path with clear filename)
- `docs/day6_splice_quickstart.md` (run order + split rationale in 1 page)

**Deferred from original Day 6 (optional):** generic `scripts/run_ablation.py` mutation sweep — still valuable; schedule on Day 7 morning or as stretch after 6D.

## Day 7 (1.5–2h): Interview packaging (splice-first narrative)

- Write a short technical summary:
  - **Lead with splice:** problem, data source, split policy, model, metrics, frozen vs fine-tuned findings, limitations.
  - **Then one slide** on synthetic phase: why it existed (pipeline only), not biological claims.
- Build 6-8 slides:
  - Motivation → real task definition → pipeline diagram → pretrained vs random / frozen vs unfrozen → probing (where signal sits) → splice robustness (masking) → limitations & next steps.
- Prepare concise answers for likely questions:
  - Why pretrained beats random-init?
  - What did probing reveal on synthetic vs (if run) on splice embeddings?
  - How did you prevent leakage across chromosomes/transcripts?
  - What would you test next (more tissues, longer flanks, multi-species)?

Output:

- `docs/technical_summary.md`
- `docs/interview_slides_outline.md`
- `docs/interview_qa.md`
- Optional: `docs/interview_story_splice.md` (5–8 bullet “elevator” tied to filenames in `results/`)

---

## Most Applicable Hands-On Experiments (Priority Order)

1. Pretrained vs random-init comparison on one genomics task (synthetic first, **real splice for claims**).
2. Layer-wise probing of embeddings for interpretable signal localization.
3. **Splice-context masking / mutation robustness** on real labels (supersedes pure synthetic ablation for interviews).

If time gets tight, finish in this order and skip extras.

## Minimal "Good Enough" Scope (If You Only Have ~6-7 Hours)

- Complete Days 1-4 only.
- Run one short training pass and produce one comparison table.
- Write a 1-page summary and 3-4 strong interview stories from your process.

## Stretch Scope (If You Have Extra Time)

- Add attention heatmap visualization for a known motif-containing sequence.
- Add second dataset task to test generalization (e.g. promoter ChIP-seq after splice ships).
- Add simple experiment tracker logging (CSV is enough; W&B optional).
- Run full mutation-rate ablation (`scripts/run_ablation.py`) on splice windows in addition to masking.

---

## Interview Framing Script (Use This Structure)

- Context: "I built a reproducible pipeline on a synthetic check, then moved to **real splice-site labels** from public annotations so metrics reflect biology-grounded signal, not toy motif injection."
- Method: "I compared random-init vs pretrained, used layer-wise probing for representation structure, then fine-tuned with held-out chromosomes/transcripts and ran a motif-flank ablation."
- Result: "Pretraining and/or partial unfreeze improved splice metrics; probing showed where linear signal concentrates; masking the splice core hurt performance, supporting reliance on known biology."
- Reflection: "Next I’d expand negatives (hard negatives), flank lengths, and optionally multi-species or tissue-specific data."

## Weekly Success Criteria

- You can explain attention, pretraining, probing, and ablation clearly in plain language.
- You can walk through your pipeline from data prep to metrics without looking at notes.
- You can defend one design choice and one limitation of your experiment.
- You have files/plots you can show if asked for concrete evidence of work.