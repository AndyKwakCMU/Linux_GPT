# pretrain/

A from-scratch GPT pretraining pipeline: tokenizer training, data packing,
a hand-written transformer (no `transformers`/`AutoModel`), and a training
loop, run entirely on your own GPU. Standalone -- doesn't import anything
from `src/linuxgpt/`, doesn't get run by `linuxgpt ingest`/`ask`, and isn't
a dependency of the RAG system in the rest of this repo.

## Why this exists

This is a learning project, not (yet) a production model. The goal is to
actually understand pretraining mechanics -- tokenization, batching,
attention, the loss curve, overfitting -- by running the real thing on real
kernel source, sized for one GPU (RTX 5000 Ada, Laptop, 16GB) instead of a
GPU cluster. See the top-level `CLAUDE.md`/conversation history for the
fuller context: the plan is to run this, learn from it, and *later* combine
whatever comes out of it with the citation-verified RAG system already
built in this repo -- not to replace it.

## Read this before starting a multi-day run: the corpus is small

`.kernel_src/` (F2FS + a curated set of kernel docs, per `kernel_manifest.yaml`
in the repo root) is roughly **1-1.5M tokens**. A competent open model is
pretrained on *trillions* of tokens; this corpus is roughly a millionth of
that. Concretely, with the default 8-layer/512-dim (~30-40M param) config:

- The model will see the **entire corpus multiple times per hour**, not
  per day. Training for multiple days on this corpus alone means training
  for hundreds/thousands of epochs over the same ~1.5M tokens.
- Expect `train_loss` to keep dropping while `val_loss` flattens and then
  *rises* -- classic overfitting/memorization, not a bug. This is actually
  the most instructive thing to watch: `checkpoints/log.csv` records both
  every `eval_interval`, plot them against each other.
- A memorized model on this corpus is not "a kernel expert" -- it's a model
  that has memorized ~1.5M tokens of text. That's a fine, real learning
  outcome (you'll see exactly *how* memorization looks in a loss curve and
  in sampled output), but don't mistake it for the eventual goal.

**To do a "real" run instead of a memorization demo:** point `--corpus-dir`
at something much bigger before training for days -- e.g. a full kernel
checkout (`git clone --depth 1 https://github.com/torvalds/linux`) instead
of just `.kernel_src/`, and/or mix in a general-text corpus so the model
also has non-kernel English/code to learn from. `tokenizer_train.py` and
`prepare_data.py` both take `--corpus-dir`, so this doesn't require editing
code -- just point them elsewhere and retrain the tokenizer on the bigger
corpus (vocab_size in `config.py` should go up too, e.g. 32k-50k, once
there's enough data to justify a bigger vocabulary).

## Setup

Separate venv from the RAG project's `.venv` on purpose -- this pulls in
`torch`, which the RAG side never needs.

```bash
cd pretrain
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# install torch per pytorch.org for your CUDA version if the requirements.txt
# pin doesn't match what's on your machine -- e.g.:
# pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Verify the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# should print: True NVIDIA RTX 5000 Ada Generation Laptop GPU
```

## Running it

```bash
# 1. Train a byte-level BPE tokenizer on the corpus (defaults to ../.kernel_src)
python tokenizer_train.py

# 2. Tokenize + pack into data/train.bin, data/val.bin
python prepare_data.py

# 3. Train (Ctrl-C any time; resume picks up from the last checkpoint)
python train.py
python train.py --resume

# 4. Sample from a checkpoint to see what it's learned so far
python sample.py --prompt "static int f2fs_"
```

`config.py` holds every hyperparameter (model size, batch size, LR
schedule, eval/checkpoint intervals) -- no CLI flag sprawl, just edit the
dataclasses there. `TrainConfig.batch_size`/`grad_accum_steps` are the first
things to tune if you hit a CUDA OOM or want to use more of the 16GB.

## What's in here

```
config.py          # GPTConfig (model) + TrainConfig (training) dataclasses
model.py           # GPT: RMSNorm, rotary embeddings, SwiGLU MLP, causal
                    #   self-attention via F.scaled_dot_product_attention
tokenizer_train.py # trains + saves a byte-level BPE tokenizer
prepare_data.py    # tokenizes the corpus -> data/train.bin, data/val.bin
train.py           # the training loop (bf16, grad accum, cosine LR,
                    #   checkpoint/resume, logs to checkpoints/log.csv)
sample.py          # generate text from a checkpoint
```

## Sizing reference (16GB, Ada, bf16)

The default config (8 layers, 8 heads, 512 dim, ~30-40M params,
block_size=1024) leaves plenty of headroom on 16GB even before tuning batch
size up. Rough parameter-count formula for this architecture:
`12 * n_layer * n_embd^2` (attention + SwiGLU dominate; embedding table is
`vocab_size * n_embd`, small at vocab_size=8192). If you scale up
`n_layer`/`n_embd` toward GPT-2-small territory (12 layers, 768 dim,
~110-125M params) it still comfortably fits 16GB at this batch size --
that's a reasonable target once the corpus is big enough to justify it (see
above).
