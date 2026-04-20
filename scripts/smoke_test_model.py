#!/usr/bin/env python3
"""
Day 2 — Smoke test: tokenizer + pretrained genomic Transformer (HF) + first forward pass.

Why this script exists
----------------------
Before you train anything, you want a *thin*, reproducible proof that:
  1) Your Python environment has compatible versions of torch + transformers.
  2) You can download the checkpoint (or use a local cache).
  3) The tokenizer maps raw DNA strings → integer token IDs the model understands.
  4) A forward pass runs and tensor shapes look sane.

Key concepts (read these once, then skim the code)
--------------------------------------------------
**Hugging Face `transformers`**
  A library of model implementations + loading utilities. It wraps PyTorch modules
  (and others) so you can do `AutoModel.from_pretrained("org/name")`.

**Checkpoint vs architecture**
  *Architecture* = the math (attention layers, dimensions).
  *Checkpoint* = architecture + learned weights + tokenizer files.
  Loading `from_pretrained` pulls config + weights + tokenizer vocab from the Hub
  (or a local folder).

**Tokenizer**
  Turns each sequence into a list of integer IDs (and often special tokens like
  [CLS], [SEP], PAD). Genomic models may use k-mers (e.g. 6-mers) instead of
  single letters—always read the model card for the exact convention.
  InstaDeep Nucleotide Transformer checkpoints may still report an ``EsmTokenizer``
  class name (ESM-derived remote code) while operating on DNA; trust the model card,
  not the class name alone.

**Tensors**
  PyTorch `torch.Tensor` objects = multidimensional arrays on CPU/GPU with
  autograd support. For inference-only smoke tests we wrap in `torch.no_grad()`.

**`trust_remote_code=True`**
  Some models ship custom Python (subclass definitions) alongside weights.
  Hugging Face requires an explicit opt-in to run that code. You still use
  PyTorch under the hood; this flag only controls whether custom model code
  from the repo is executed.

**Attention mask**
  A tensor of 0/1 (or bool) telling the model which positions are real tokens vs
  padding. Padding exists so batches can be rectangular even when sequences
  differ in length.

**Hidden states**
  With `output_hidden_states=True`, transformers return a tuple: one tensor per
  layer (embedding layer + each block). The *last* hidden state is often used
  for downstream sequence representations.

**Forward call (this script)**
  The default ``--model_id`` is InstaDeep's Nucleotide Transformer; its remote
  code often expects ``encoder_attention_mask`` in addition to ``attention_mask``.
  If you point ``--model_id`` at another MLM model and get ``TypeError: unexpected
  keyword argument 'encoder_attention_mask'``, remove that keyword from the
  ``model(...)`` call in ``main()`` and keep only ``attention_mask=``.

References
----------
Model card (example used below):
  https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species

Remote code vs ``transformers`` version
----------------------------------------
Checkpoints with ``trust_remote_code=True`` download Python modules into your HF
cache. If you see ``ImportError`` from ``modeling_esm.py`` inside that cache,
your **installed** ``transformers`` is newer or older than the **Hub module**
expects (API drift). Align versions per the model card or clear the cached
``transformers_modules/.../InstaDeepAI/...`` folder after changing installs.

macOS note (OpenMP / libomp)
---------------------------
If you see ``OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib
already initialized`` and the process aborts, two libraries in your env each
linked OpenMP (often PyTorch + NumPy/BLAS via conda). Setting
``KMP_DUPLICATE_LIB_OK=TRUE`` *before* ``import torch`` avoids that abort; this
script sets it on Darwin by default. For a stricter fix, align OpenMP in conda
(e.g. ``llvm-openmp``) or export the variable in your shell before running.
"""

from __future__ import annotations

import argparse
import os
import sys

# Must run before importing torch (or numpy that pulls OpenMP). Conda + macOS often hits duplicate libomp.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--model_id",
        type=str,
        default="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
        help="Hugging Face hub model id (org/name).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Where to run tensors. 'auto' picks cuda > mps > cpu when available.",
    )
    return p.parse_args()


def pick_device(name: str) -> str:
    """Return a torch device string. 'mps' is Apple Silicon GPU; 'cuda' is NVIDIA."""
    import torch

    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    args = parse_args()
    # Import here so `--help` works even if torch isn't installed yet.
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    device = pick_device(args.device)
    print(f"[smoke_test] device: {device}")

    # 1) Tokenizer: DNA strings -> input_ids + attention_mask (padding/truncation to one length).
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    max_length = min(tokenizer.model_max_length, 512)
    sequences = [
        "ATTCCGATTCCGATTCCG",
        "ATTTCTCTCTCTCTCTGAGATCGATCGATCGAT",
    ]
    batch = tokenizer(
        sequences,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    print(f"[smoke_test] input_ids:        {tuple(input_ids.shape)}  (batch, seq_len)")
    print(f"[smoke_test] attention_mask:   {tuple(attention_mask.shape)}")

    # 2) Model weights + 3) one forward pass (no gradients).
    model = AutoModelForMaskedLM.from_pretrained(args.model_id, trust_remote_code=True)
    model.eval()
    model.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True,
        )

    print(f"[smoke_test] logits:           {tuple(outputs.logits.shape)}  (batch, seq_len, vocab)")
    hs = outputs.hidden_states
    print(f"[smoke_test] hidden_states:    {len(hs)} tensors (embedding + transformer layers)")
    print(f"[smoke_test] last_hidden:      {tuple(hs[-1].shape)}  (batch, seq_len, hidden_dim)")

    print("[smoke_test] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
