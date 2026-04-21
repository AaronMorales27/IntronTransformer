# Day 3 Quickstart (Beginner-First)

This guide is for your first run of **Decision 1**:
- frozen backbone
- trainable head only

You will produce:
- `results/metrics_baseline_vs_pretrained.csv`

---

## 0) What this run is doing in plain English

You run two experiments on the same dataset:

1. `pretrained_frozen_head_only`
   - Transformer backbone starts from learned genomic weights.
   - Backbone does not update.
   - Only the small linear classifier head updates.

2. `random_frozen_head_only`
   - Same backbone architecture, but random weights.
   - Backbone does not update.
   - Only the small linear classifier head updates.

This is a quick representation-quality probe:
- If pretrained geometry is useful, the pretrained run should perform better.

---

## 1) Install packages (first time only)

From the project root:

```bash
python -m pip install --upgrade pip
python -m pip install torch transformers scikit-learn numpy
```

What each package is for:
- `torch`: training loop, tensors, neural net modules.
- `transformers`: loading tokenizer + genomic transformer backbone.
- `scikit-learn`: accuracy, F1, ROC-AUC metrics.
- `numpy`: reproducibility helpers.

---

## 2) Optional: regenerate demo data

If you want to rebuild the toy dataset:

```bash
python scripts/prepare_data.py --out data/processed/promoters_demo.csv --n_total 200 --seed 42
```

---

## 3) Run the Day 3 script

Minimal run:

```bash
python scripts/train_classifier.py
```

Recommended explicit run (clearer for learning):

```bash
python scripts/train_classifier.py \
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

Run with threshold sweep (recommended after first baseline):

```bash
python scripts/train_classifier.py \
  --data_csv data/processed/promoters_demo.csv \
  --epochs 3 \
  --batch_size 8 \
  --lr 1e-3 \
  --max_length 256 \
  --seed 42 \
  --device auto \
  --sweep_thresholds \
  --out_csv results/metrics_baseline_vs_pretrained.csv
```

---

## 4) How to read training logs

You will see lines like:

```text
[train] epoch=1 loss=... val_acc=... val_f1=... val_auc=...
```

Interpretation:
- `loss` down over epochs is usually good.
- `val_acc`, `val_f1`, `val_auc` up is usually good.
- Compare the final pretrained vs random rows in output CSV.

---

## 5) Inspect metrics file

```bash
python -c "import csv; p='results/metrics_baseline_vs_pretrained.csv'; print(open(p).read())"
```

Each row includes:
- `run_name` (`pretrained_frozen_head_only` or `random_frozen_head_only`)
- `split` (`val` or `test`)
- `accuracy`, `f1`, `roc_auc`
- `selected_threshold` and `threshold_source` (`fixed_0.5` or `val_f1_sweep`)
- your run settings (`epochs`, `batch_size`, `lr`, `max_length`, `seed`)

Why threshold sweep matters:
- `roc_auc` measures ranking quality and ignores threshold choice.
- `accuracy` and `f1` require hard class decisions (`0` vs `1`), so threshold matters.
- If pretrained has better ranking but weak `f1` at threshold `0.5`, sweeping threshold on validation can reveal the true benefit.

---

## 6) Common first-time issues

1. **`ModuleNotFoundError`**  
   Install missing package with `python -m pip install <name>`.

2. **Hugging Face download/auth/network issues**  
   Retry on stable internet. If needed, run once while logged in to HF CLI.

3. **Slow run on CPU**  
   Lower `--max_length` (e.g. 128) and `--batch_size` (e.g. 4).

4. **MPS/CUDA device issue**  
   Force CPU:
   ```bash
   python scripts/train_classifier.py --device cpu
   ```

---

## 7) Why this helps before light fine-tuning

This run isolates one question:
- "How good is the frozen representation space?"

After this, light fine-tuning adds one more question:
- "If we let top transformer layers move, does task performance improve further?"

