# Project Story: From Setup to Adaptation

This document is the coherent thread across the experiments and iterative commits.
The point is not "final benchmark performance."
The point is: can I explain what I did, why each step existed, what each step taught me, and what should happen next.

## 1) Starting point: prove the stack works

Before training anything, I needed a thin proof that the environment and model path were real:
- tokenizer loads,
- checkpoint loads,
- forward pass runs,
- tensor shapes make sense.

This stage is mostly about removing setup ambiguity.
If this fails, later training results are noise because the pipeline is not trustworthy yet.

## 2) First real question: is pretrained geometry useful when frozen?

After setup, I moved to the simplest downstream comparison:
- pretrained frozen backbone + trainable linear head,
- random-init frozen backbone + trainable linear head.

Why this specific setup:
- freezing isolates representation quality from optimization complexity,
- linear head keeps capacity low so the backbone signal has to do most of the work.

This is not "best achievable performance."
It is a controlled probe: do pretrained embeddings make the task easier to separate?

## 3) Calibration insight: AUC and F1 answer different questions

A key turning point was seeing that one model can have stronger `ROC-AUC` but weaker F1/accuracy at threshold `0.5`.
That is not automatically contradictory.

Interpretation:
- `ROC-AUC`: ranking quality across thresholds,
- `F1` / `accuracy`: quality of one hard decision threshold.

So I added validation threshold sweep (`--sweep_thresholds`) to separate:
1. representation ranking quality
2. threshold calibration quality

This changed how I interpret "better" models:
better ranking can be hidden by a poor fixed cutoff.

## 4) Engineering cleanup: move from script-heavy to reproducible pipeline

Once baseline behavior was understood, I moved logic from one training script into modular pipeline components and config-driven runs.

Why this matters:
- easier reruns,
- easier extension to new tasks,
- less accidental drift between experiments.

This step was less about new model insight and more about making future insight reliable.

## 5) Representation analysis: where is signal linearly readable?

Next, I ran layer-wise probing:
- freeze backbone,
- extract hidden states by layer,
- train simple probes per layer,
- compare performance across depth.

This gives a layer-wise story instead of treating the model as a black box.
High probe performance at a layer suggests task signal is decodable there with low-capacity readout.

Important caution:
probing demonstrates decodability, not necessarily that the model uses that exact feature set in end-to-end optimization.

## 6) Task transition: from synthetic promoters to published splice-site data

I started with synthetic promoter-style data for one reason: fast iteration while building trust in the pipeline.
That made it easier to debug tokenization, training loops, metric logging, and threshold behavior without expensive data setup.

After the pipeline became stable, I moved to published splice-site data so the experiments would be biologically grounded.
This transition changed the meaning of results:
- synthetic phase: method/pipeline validation and intuition building,
- splice phase: evidence on a real annotated task with stronger external relevance.

This was a deliberate decision, not a random dataset swap.
The progression was "first make the machinery reliable, then test it on data that matters more."

## 7) Adaptation step: partial unfreeze ladder on splice track

After frozen comparisons and probing, the next question became:
"If I let only the top N transformer blocks adapt, can I improve task metrics without fully unfreezing everything?"

That motivated the partial-unfreeze ladder:
- frozen baseline,
- top-1 unfreeze,
- top-2 unfreeze,
- top-4 unfreeze,
- optional random anchor for sanity.

This is a tradeoff search:
- more adaptation can improve task fit,
- but can increase instability/overfitting/runtime cost.

## 8) What I can currently claim vs what I cannot yet claim

Reasonable current claims:
- I have a reproducible path from data prep to metrics.
- I can explain why AUC and F1 can disagree and how calibration addresses that.
- I can run frozen/probing/partial-unfreeze workflows and compare outputs consistently.

Claims I should avoid (for now):
- strong biological conclusions from small or synthetic-heavy runs,
- broad generalization claims without stronger split policy + multi-seed stability,
- claiming causal biological mechanism from probing alone.

## 9) Logical next steps

1. Stabilize top configurations with multi-seed runs and aggregate reporting.
2. Strengthen non-leaky split strategy for real biological tasks.
3. Add robustness checks (motif-context masking / perturbation) and document degradation patterns.
4. Promote final artifacts into a cleaner `docs/` layout after content wording is finalized.

## 10) Reading order

If someone is new to this repo, this order is the shortest path:
1. `README.md`
2. `docs/story.md`
3. `docs/results.md`
