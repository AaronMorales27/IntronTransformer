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

References
----------
Model card (example used below):
  https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species


## Summarization of the program

## Smoke test: what actually happens

### Inputs (two DNA strings → one batch)
- You pass **two raw nucleotide strings**. The **tokenizer** turns each character (or k-mer chunk, depending on the model) into **integer token IDs** (`input_ids`).
- **Padding / truncation** only exists so both rows have the **same length** (here up to 512). Shorter sequences get **PAD** tokens at the end; longer ones are **cut off**. The **attention mask** marks real bases vs padding so the model does not treat padding as sequence.
- This script runs **one forward pass** (inference-style), **not training**: there is **no loss**, **no backward pass**, and **no weight updates**. The same `input_ids` format is what you would feed during training, but training also needs labels and an optimizer loop.

### After the model is loaded: forward pass (intuition + geometry)
- **Embedding step:** each of the 512 positions is mapped from a discrete token ID to a **vector** in $(\mathbb{R}^{d})$ (here \(d = 512\) for the last layer’s hidden size). Think of a **curve** (or walk) of 512 points in a high-dimensional space—one point per position.
- **Transformer layers (×12):** at each layer, **self-attention** lets every position gather information from other positions (weighted mix), then **feed-forward** blocks reshape those vectors. Geometrically, the sequence of 512 vectors is **repeatedly transformed**; the network builds **context-aware** representations so each position “knows” about relevant parts of the sequence.
- **Head at the end:** the **masked language modeling (MLM) head** maps each position’s final hidden vector back to a **score vector over the whole vocabulary** (length ≈ 4107 here).

### What `logits` are
- **`logits[b, i, :]`** = for batch index `b`, sequence position `i`, a **raw score for every possible token** the model could emit at that position (before softmax).
- They answer: “If this were an MLM task, **which token** would the model favor here?” Training compares these to **masked** targets; your smoke test **does not** apply random masking or a cross-entropy loss—it just **runs the stack** so you see sane shapes.

### What this smoke test did **not** compute
- **Training:** no masked labels, **no loss**, **no gradients**, **no optimizer step**.
- **Inference extras:** no softmax/probabilities, no “best token” argmax, no sampling.
- **Downstream head:** no classifier on top of `[CLS]` / mean pooling / last layer for your promoter task.
- **Data pipeline:** no CSV loading, shuffling, or epochs—only two hard-coded strings.