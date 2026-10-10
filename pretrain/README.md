# pretrain/

From-scratch pretraining of small GPTs whose weights are trained directly on
Linux kernel source -- one **piece** per subsystem, starting with F2FS.
Tokenizer, data packing, a hand-written transformer (no
`transformers`/`AutoModel`) and the training loop are all here, sized for
one 16GB laptop GPU (RTX 5000 Ada, Laptop).

Standalone: own venv, doesn't import `src/linuxgpt/`, isn't run by
`linuxgpt ingest`/`ask`.

## Why this exists

Two goals at once, on purpose:

1. **Learn pretraining by doing it**: tokenization, batching, attention,
   loss curves, overfitting, all on real data you can read.
2. **Give the tool intuition that comes from the source itself.** The RAG
   side (`linuxgpt ask`) has a general model (Qwen) *reading* retrieved
   chunks. A piece is different: its weights were shaped by nothing but
   F2FS's C, comments and the kernel's own docs. General pretrained models
   keep the jobs they're good at -- reading English questions, writing
   English answers, orchestrating -- and talk to the pieces (see Roadmap).

The citation verifier stays the source of truth. **Nothing a piece
produces ever counts as a citation or bypasses verification**; at most it
steers what gets retrieved, or is shown labeled as unverified model
intuition.

## Concepts

- **Piece** = one small model + its corpus manifest, `pieces/<name>.yaml`:
  exactly which files of the pinned v6.12 tarball its weights train on.
  `pieces/f2fs.yaml` = `fs/f2fs/`, F2FS's headers outside it
  (`include/linux/f2fs_fs.h`, `include/trace/events/f2fs.h`), and the
  curated kernel docs from `kernel_manifest.yaml`. The RAG side's
  hand-authored lessons are excluded: they're synthesis, not source.
- **Shared, frozen tokenizer** = `tokenizers/kernel-v6.12-16k.json`, trained
  once on the *whole* kernel tree (minus 296 generated >256KB register
  headers) and committed. Every piece uses it, which is what makes pieces
  stitchable later: token id 1234 means the same thing in every piece.
  Never retrain it casually -- every checkpoint records its sha256 and
  `train.py`/`sample.py`/`eval_probes.py` refuse a mismatch.
  It uses a C-aware pre-tokenizer (identifiers stay whole, so BPE learns
  `spin_lock_irqsave`, ` f2fs`, `_sb_info` as tokens); GPT-2's default
  splits at every `_` and digit. On F2FS that's 3.03 chars/token vs 2.65.
- **Split by top-level definition, not by file.** F2FS is ~30 C files; a
  by-file split made val one random file. `build_corpus.py` cuts code at
  column-0 `}` / `)` (one function/struct/TRACE_EVENT per unit, with its
  leading comment) and docs at RST section titles, then hashes each unit
  into train (~90%) or val (~10%). Every unit keeps `path:start-end` line
  provenance.

## The corpus is small -- size your expectations to it

F2FS piece, measured: **46 files, 1,980 units, 502,849 train / 57,305 val
tokens.** A competent open model sees trillions of tokens; this is half a
million. So:

- The laptop is for **many short experiments**, not one multi-day run.
  `tiny` (~3M non-embedding params) passes over the whole corpus every ~15
  steps. Compare `tiny` vs `small`, vary dropout, watch where val_loss
  bottoms out. That comparison *is* the lesson.
- Expect `train_loss` to keep dropping while `val_loss` flattens then rises
  -- memorization. `ckpt_best.pt` keeps the lowest-val_loss checkpoint, which
  is the one to use; `ckpt.pt` is just the latest (for `--resume`).
- `eval_probes.py` tells you whether the model learned *structure* (beats a
  bigram baseline on held-out functions it never saw) or just memorized.
- Multi-day runs belong to the rented-GPU stage with much bigger pieces.

## Setup (on the NVIDIA laptop)

```bash
git clone <this repo> && cd Linux_GPT
# the pinned tarball: either let the RAG side download + verify it ...
python -m venv .venv && source .venv/bin/activate && pip install -e . && linuxgpt ingest
# ... or copy .cache/linux-6.12.tar.xz over from another machine.

cd pretrain
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# install torch for your CUDA version if the default wheel doesn't match, e.g.:
# pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# should print: True NVIDIA RTX 5000 Ada Generation Laptop GPU
```

Separate venv on purpose -- this pulls in `torch`, which the RAG side never
needs. The tarball's sha256 is checked against kernel.org on every run
(`--skip-verify` for offline; the local hash is still recorded).

## Running it

```bash
# 0. Tokenizer: already trained and committed. Only re-run to deliberately
#    start a new, incompatible tokenizer generation (~8GB RAM peak).
# python tokenizer_train.py

# 1. Cut the piece's corpus into units with provenance + train/val split
python build_corpus.py --piece f2fs

# 2. Tokenize -> data/f2fs/train.bin, val.bin, data_meta.json
python prepare_data.py --piece f2fs

# 3. Train (Ctrl-C any time; --resume continues from checkpoints/f2fs/ckpt.pt)
python train.py --piece f2fs --preset tiny
python train.py --piece f2fs --preset tiny --resume

# 4. Look at what it learned
python sample.py --piece f2fs --prompt "static int f2fs_"
python eval_probes.py --piece f2fs --show 10
```

To run `small` after `tiny`, move `checkpoints/f2fs/` aside first (resume
refuses a checkpoint from a different preset).

## What's in here

```
pieces/f2fs.yaml   # piece manifest: which tarball files this piece trains on
kernel_tarball.py  # tarball sha256 verification + streaming (mirrors linuxgpt's fetch.py)
tokenizer_train.py # trains the shared C-aware byte-level BPE on the whole tree
tokenizers/        # the frozen shared tokenizer + its .sha256 (committed)
build_corpus.py    # piece -> data/<piece>/corpus.jsonl (units, provenance, split)
prepare_data.py    # corpus.jsonl -> train.bin / val.bin / data_meta.json
config.py          # GPTConfig + TrainConfig + presets (tiny / small / gpt2s)
model.py           # GPT: RMSNorm, rotary embeddings, SwiGLU MLP, SDPA attention
train.py           # training loop: bf16, grad accum, cosine LR, best/latest
                   #   checkpoints with provenance, log.csv
sample.py          # generate text from a checkpoint
eval_probes.py     # held-out function probes vs unigram/bigram baselines
```

## Presets (16GB Ada, bf16)

| preset  | shape (layers/dim/heads, ctx) | non-embedding | + tied embedding | where |
|---------|-------------------------------|---------------|------------------|-------|
| `tiny`  | 4 / 256 / 4, 512              | ~3M           | +4.2M            | laptop default |
| `small` | 6 / 384 / 6, 512              | ~10.6M        | +6.3M            | laptop, expect earlier overfit |
| `gpt2s` | 12 / 768 / 12, 1024           | ~85M          | +12.6M           | rented GPU, big pieces only |

Non-embedding params ≈ `12 * n_layer * n_embd^2`. If you hit CUDA OOM,
lower `batch_size` in the preset's overrides in `config.py` and raise
`grad_accum_steps` to keep the effective batch the same.

## Roadmap

1. **F2FS piece** (now): train `tiny`/`small` on the laptop, read the
   curves, run the probes.
2. **Bridge stage 1 -- retrieval.** `serve.py`: a stdlib `http.server`
   process in this venv exposing `/embed`, `/score`, `/complete` for a
   checkpoint. The RAG side calls `/embed` as an optional second retriever
   or reranker next to `nomic-embed-text`, so F2FS-trained weights decide
   *what Qwen reads*. Verification is unaffected by construction -- it only
   changes which chunks are candidates.
   - Honest caveat: a piece trained mostly on C won't embed English
     questions well out of the box. Step one is measuring mean-pooled
     hidden states against nomic on a small question -> expected-symbol set.
     Step two is a small contrastive head trained on the kernel's own
     (comment, function body) pairs -- English<->F2FS-code embeddings whose
     weights still come only from the source.
3. **Bridge stage 2 -- tools.** Qwen calls `/complete` ("what usually
   follows this?") and `/score` ("how surprising is this code to a model
   trained on F2FS?") as tools. Output is labeled *unverified model
   intuition* in the CLI and never counts as a citation.
4. **More pieces on rented GPUs**: VFS, block, mm, each a
   `pieces/<name>.yaml`, all on the same frozen tokenizer. Start with a
   router that picks which piece(s) to consult; the shared tokenizer is
   what leaves weight-level options (distilling pieces into one model,
   mixture-of-experts style stitching) open later.
