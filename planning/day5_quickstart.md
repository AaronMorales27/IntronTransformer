# Day 5 Quickstart (Layer-Wise Probing, Intuition-First)

This day answers one practical question:

**"Which transformer layers carry the most linearly accessible signal for my task?"**

## Why probing?

You already compared pretrained vs random (Day 3/4). Probing adds a new lens:

- Keep the backbone frozen.
- Extract embeddings from each hidden layer.
- Train a tiny linear classifier (logistic regression) per layer.
- Compare layer scores.

If a simple probe performs well on a layer, that layer's representation is likely organized in a way that makes the label easy to separate.

## Design decisions (and why they teach intuition)

1. **Frozen backbone**
   - Prevents the model from adapting during probing.
   - Keeps the question focused on "what is already encoded?" not "what can be learned with fine-tuning?"

2. **Linear probe**
   - Low-capacity classifier.
   - High score means representation quality is doing most of the work.

3. **Layer-wise curve**
   - Converts abstract "hidden states" into an interpretable performance profile.
   - Helps build stories like "mid layers peak" or "final layers are best for this task."

4. **CSV as source of truth**
   - `results/probe_scores.csv` is the canonical artifact.
   - Plot is a convenience layer on top.

## Run it

From project root:

```bash
python3 scripts/run_probing.py \
  --config configs/default.yaml \
  --out_csv results/probe_scores.csv \
  --out_png results/fig_probe_layerwise.png \
  --threshold_mode val_sweep \
  --metric f1
```

### Single-seed baseline (quick sanity check)

```bash
python3 scripts/run_probing.py \
  --config configs/default.yaml \
  --seed 42 \
  --threshold_mode fixed
```

### Recommended: per-layer val-threshold sweep

```bash
python3 scripts/run_probing.py \
  --config configs/default.yaml \
  --seed 42 \
  --threshold_mode val_sweep \
  --metric f1
```

### Recommended for stability: multi-seed + val sweep

```bash
python3 scripts/run_probing.py \
  --config configs/default.yaml \
  --seeds 42,43,44,45,46 \
  --threshold_mode val_sweep \
  --metric f1
```

Useful speed overrides:

```bash
python3 scripts/run_probing.py \
  --config configs/default.yaml \
  --batch_size 4 \
  --max_length 128 \
  --device cpu
```

## Outputs

- `results/probe_scores.csv`: includes per-seed test rows (`split=test`) and aggregated rows (`split=test_mean`).
- `results/fig_probe_layerwise.png`: metric vs layer depth curve.

CSV columns now include:
- `selected_threshold`: threshold chosen per layer (`0.50` for fixed mode, data-driven for val sweep).
- `seed`: integer for per-seed rows, `all` for aggregate rows.

## How to read the result

- **Increasing curve**: later layers become more task-aligned.
- **Middle-layer peak**: middle representations capture most transferable signal.
- **Flat curve**: either signal is distributed or the dataset/task is too simple/noisy to separate layer quality.

Interpretation tip:
- Trust `split=test_mean` rows most when using multiple seeds; they reduce one-off randomness.

## Interview framing

Try this concise structure:

- "I froze the genomic transformer and trained linear probes at each layer."
- "This isolates representation quality from fine-tuning effects."
- "The best layer was X by F1, suggesting useful task signal concentrates there."
- "Next, I'd test whether the same layer trend holds under mutation/masking robustness."
