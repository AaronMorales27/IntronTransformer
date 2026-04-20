# Day 4 Quickstart (Pipeline + Config, Beginner-First)

This guide helps you run the Day 4 pipeline while keeping Day 3 behavior recognizable.

You now have two equivalent ways to run:
- familiar wrapper script: `scripts/train_classifier.py`
- modular runner: `python -m src.pipeline.runner`

Both paths produce the same style of metrics CSV.

---

## 0) What changed from Day 3

Day 3 had most logic in one script.

Day 4 keeps your Day 3 entrypoint, but moves internals into:
- `src/pipeline/data.py`
- `src/pipeline/modeling.py`
- `src/pipeline/train.py`
- `src/pipeline/evaluate.py`
- `src/pipeline/runner.py`
- `src/pipeline/config.py`

And adds a default config:
- `configs/default.yaml`

Why this helps:
- easier to maintain
- easier to test and extend
- easier to reproduce runs from config

---

## 1) Install packages (first time only)

From project root:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install torch transformers scikit-learn numpy pyyaml
```

Package notes:
- `pyyaml` is needed for `--config` YAML support.

---

## 2) Optional: regenerate demo data

```bash
python3 scripts/prepare_data.py --out data/processed/promoters_demo.csv --n_total 200 --seed 42
```

---

## 3) Run with the familiar Day 3 entrypoint

### Minimal

```bash
python3 scripts/train_classifier.py
```

### Explicit arguments (same style as Day 3)

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

### With threshold sweep

```bash
python3 scripts/train_classifier.py \
  --data_csv data/processed/promoters_demo.csv \
  --epochs 3 \
  --batch_size 8 \
  --lr 1e-3 \
  --max_length 256 \
  --seed 42 \
  --device auto \
  --sweep_thresholds \
  --out_csv results/metrics_baseline_vs_pretrained_seed_42_swept.csv
```

---

## 4) Run with Day 4 config file

### Use defaults from config

```bash
python3 -m src.pipeline.runner --config configs/default.yaml
```

### Use config + targeted overrides

```bash
python3 -m src.pipeline.runner \
  --config configs/default.yaml \
  --seed 42 \
  --sweep_thresholds \
  --out_csv results/metrics_baseline_vs_pretrained_seed_42_swept.csv
```

Tip:
- CLI overrides win over YAML values.

---

## 5) Expected output

You should see:
- setup logs (`device`, split sizes)
- train logs per epoch (`loss`, `val_acc`, `val_f1`, `val_auc`)
- threshold log per run
- final line with output CSV path

Default output file:
- `results/metrics_baseline_vs_pretrained.csv`

Each row includes:
- `run_name`, `split`
- `accuracy`, `f1`, `roc_auc`
- `selected_threshold`, `threshold_source`
- run settings (`epochs`, `batch_size`, `lr`, `max_length`, `seed`)

---

## 6) How to inspect results quickly

```bash
python3 -c "print(open('results/metrics_baseline_vs_pretrained.csv').read())"
```

For swept run:

```bash
python3 -c "print(open('results/metrics_baseline_vs_pretrained_seed_42_swept.csv').read())"
```

---

## 7) Common issues

1. **`ModuleNotFoundError: No module named 'torch'`**
   - Install dependencies from section 1.

2. **`ImportError` for YAML support**
   - Install `pyyaml`.

3. **Hugging Face download/auth/network issues**
   - retry with stable internet
   - optionally run once with HF CLI auth

4. **Slow on CPU**
   - reduce `--max_length` (e.g. 128)
   - reduce `--batch_size` (e.g. 4)

5. **MPS/CUDA issue**
   - force CPU:
   ```bash
   python3 scripts/train_classifier.py --device cpu
   ```

---

## 8) Suggested Day 4 workflow

1. Run baseline with fixed threshold.
2. Run with `--sweep_thresholds`.
3. Compare pretrained vs random on:
   - threshold-free signal (`roc_auc`)
   - thresholded decisions (`f1`, `accuracy`)
4. Keep outputs under `results/` with clear filenames per seed/settings.

