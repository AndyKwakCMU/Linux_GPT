# pretrain/

From-scratch pretraining of small GPTs whose weights are trained directly on
Linux kernel source: a whole-kernel base, fine-tuned down to the filesystem
stack, then to F2FS. Tokenizer, data packing, a hand-written transformer (no
`transformers`/`AutoModel`) and the training loop are all here, sized for
one 16GB laptop GPU. It has run on a Quadro RTX 5000 Max-Q (Turing, fp16);
it was written for an RTX 5000 Ada (bf16), and `train.py` picks whichever
the GPU supports.

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

- **Piece** = one corpus manifest, `pieces/<name>.yaml`: exactly which
  files of the pinned v6.12 tarball a model's weights train on. Three exist,
  each a superset of the next:

  | piece | contents | train / val tokens |
  |---|---|---|
  | `kernel` | every `.c`/`.h`/`.rst` in the tree, minus 293 generated files >256KB | 258.7M / 29.3M |
  | `fs` | F2FS, VFS core, iomap/fscrypt/fsverity/quota/jbd2, local + flash filesystems, `block/`, `mm/`, their headers + docs | 10.1M / 1.1M |
  | `f2fs` | `fs/f2fs/`, its two outside headers, the curated docs from `kernel_manifest.yaml` | 0.50M / 0.06M |

  The RAG side's hand-authored lessons are in none of them: they're
  synthesis, not source.
- **Training chain**: pretrain on `kernel`, then `train.py --init-from`
  fine-tunes on `fs`, then on `f2fs`. Each step measurably helps (see
  Results). Pieces share the tokenizer, so any checkpoint can initialize
  any piece's run at the same preset.
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
  provenance. The hash is of (path, unit text), so a unit is val in every
  piece that contains it: nothing a downstream probe tests on is ever seen
  upstream in the chain.

## Results (2026-10, Quadro RTX 5000 Max-Q)

All numbers are `eval_probes.py` on the **same 123 held-out F2FS
functions** (21,493 body tokens) -- loss / perplexity / top-1 next-token
accuracy. Baselines on that set: bigram 3.166 / 23.7 / 36.2%, unigram
6.127 / 458 / 8.4%.

| training | preset | F2FS probe loss | ppl | top-1 | identifier top-1 |
|---|---|---|---|---|---|
| `f2fs` only (best of tiny/small, dropout, schedule sweeps) | small | 1.815 | 6.1 | 65.0% | 52.5% |
| `fs` -> `f2fs` | small | 1.021 | 2.8 | 75.4% | 67.6% |
| `fs` -> `f2fs` | large | 1.012 | 2.8 | 76.1% | 68.4% |
| `kernel` -> `fs` -> `f2fs` | small | 0.939 | 2.6 | 76.6% | 68.7% |
| `kernel` -> `fs` -> `f2fs` | medium | 0.887 | 2.4 | 77.8% | 70.2% |
| `kernel` -> `fs` -> `f2fs` | **large** | **0.854** | **2.3** | **78.8%** | **71.7%** |

What it taught:

- **On 0.5M tokens, nothing but more data matters.** The `f2fs` piece
  overfits after ~400 steps (~12 epochs) at every size, dropout and
  schedule tried; `ckpt_best.pt` (lowest val_loss) is the one to use,
  `ckpt.pt` is just the latest for `--resume`.
- **Neighboring code is the data.** Training on `fs` (20x the tokens) cut
  F2FS perplexity from 6.1 to 3.0 *before* any F2FS fine-tune -- ext4,
  btrfs, VFS, mm and block teach F2FS.
- **Size only pays once data stops being the limit.** On `fs` (10M
  tokens), small/medium/large all plateau at val_loss ~1.6. On `kernel`
  (259M), F2FS loss falls 0.94 -> 0.89 -> 0.85 with size.
- **Each chain step helps at every size.** `kernel -> fs -> f2fs` beats
  `kernel -> f2fs` by 0.035-0.055 loss; the `kernel` base alone (no
  fine-tune) is *worse* on F2FS than any `fs`-trained model (1.24-1.47).
- **`large` has roughly used up v6.12.** Its `kernel` val_loss bottoms at
  iter 14,500 (~1.8 epochs) and starts rising; `medium` was still
  improving at 16,000. Bigger needs more data (e.g. git history), not
  more epochs.
- Fine-tunes peak fast: 50-225 iters on `f2fs`, 1,500-2,000 on `fs`.

Best checkpoint: `checkpoints/f2fs/ft_kernel_fs_large/ckpt_best.pt`;
`ft_kernel_fs_medium` is 0.03 behind at 2.3x fewer params.

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
# pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# e.g.: True Quadro RTX 5000 with Max-Q Design
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

`--run NAME` writes to `checkpoints/<piece>/NAME/` so experiments don't
overwrite each other; `--dropout`, `--weight-decay`, `--lr` (min_lr =
lr/10), `--warmup-iters`, `--eval-interval`, `--max-iters` override the
preset. Fine-tuning from another piece's checkpoint (same preset):

```bash
python train.py --piece f2fs --preset small --run ft --init-from checkpoints/fs/small5000/ckpt_best.pt \
    --lr 1e-4 --warmup-iters 20 --max-iters 400 --eval-interval 25
python eval_probes.py --piece f2fs --checkpoint checkpoints/f2fs/ft/ckpt_best.pt
```

The whole chain (`kernel` pretrain -> `fs` -> `f2fs`, a `kernel -> f2fs`
control, probes for every stage) is `pipeline.sh`; run it detached, it
takes hours (`small` ~1.5h, `medium` ~3h, `large` ~5.5h on the Max-Q):

```bash
python build_corpus.py --piece kernel && python prepare_data.py --piece kernel   # ~2 min
python build_corpus.py --piece fs && python prepare_data.py --piece fs
setsid nohup bash pipeline.sh medium large > logs/pipeline.nohup 2>&1 &
# results: logs/kernel_<preset>_summary.txt
```

## What's in here

```
pieces/*.yaml      # piece manifests (kernel, fs, f2fs): which tarball files train the weights
kernel_tarball.py  # tarball sha256 verification + streaming (mirrors linuxgpt's fetch.py)
tokenizer_train.py # trains the shared C-aware byte-level BPE on the whole tree
tokenizers/        # the frozen shared tokenizer + its .sha256 (committed)
build_corpus.py    # piece -> data/<piece>/corpus.jsonl (units, provenance, split)
prepare_data.py    # corpus.jsonl -> train.bin / val.bin / data_meta.json
config.py          # GPTConfig + TrainConfig + presets (tiny / small / medium / large / gpt2s)
model.py           # GPT: RMSNorm, rotary embeddings, SwiGLU MLP, SDPA attention
train.py           # training loop: bf16 (fp16 + GradScaler on pre-Ampere), grad accum,
                   #   cosine LR, --init-from fine-tuning, best/latest checkpoints
                   #   with provenance, log.csv
pipeline.sh        # kernel -> fs -> f2fs pretrain + fine-tunes + probes, per preset
sample.py          # generate text from a checkpoint
eval_probes.py     # held-out function probes vs unigram/bigram baselines
```

## Presets

Throughput measured on the 16GB Quadro RTX 5000 Max-Q (fp16, torch.compile).

| preset   | shape (layers/dim/heads, ctx) | non-embedding | + tied embedding | tok/s | where |
|----------|-------------------------------|---------------|------------------|-------|-------|
| `tiny`   | 4 / 256 / 4, 512              | ~3M           | +4.2M            | ~225k | quick experiments |
| `small`  | 6 / 384 / 6, 512              | ~10.6M        | +6.3M            | ~110k | laptop |
| `medium` | 8 / 512 / 8, 512              | ~25M          | +8.4M            | ~63k  | laptop, `kernel`/`fs` |
| `large`  | 12 / 640 / 10, 512            | ~59M          | +10.5M           | ~31k  | laptop upper end (5GB VRAM) |
| `gpt2s`  | 12 / 768 / 12, 1024           | ~85M          | +12.6M           | --    | rented GPU |

Non-embedding params ≈ `12 * n_layer * n_embd^2`. If you hit CUDA OOM,
lower `batch_size` in the preset's overrides in `config.py` and raise
`grad_accum_steps` to keep the effective batch the same.

## Roadmap

1. ~~**F2FS piece**~~ done, and extended to the `kernel -> fs -> f2fs`
   chain -- see Results.
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
4. **More pieces / more data on rented GPUs.** The chain result suggests
   specialists should be *fine-tunes of one shared base* (VFS, block, mm
   each `kernel -> <piece>`), not independent from-scratch models. A
   bigger base needs more than v6.12's 259M tokens: git history (patches
   + commit messages) is the obvious next source. The shared tokenizer
   keeps routing / distillation / MoE-style stitching open.
