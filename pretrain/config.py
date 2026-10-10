"""Model + training hyperparameters: two dataclasses plus named presets. No
framework config magic -- edit values here; train.py's only overrides are
--piece, --preset and --max-iters.

Sizing note: the F2FS piece (pieces/f2fs.yaml) is only ~0.5M tokens. A
GPT-2-small-sized model memorizes that in well under an hour, so on the
laptop the useful work is many short runs at `tiny`/`small` (compare their
train/val curves, vary dropout) rather than one multi-day run. `gpt2s` is
for the rented-GPU stage, once a piece's corpus is big enough to justify it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

PRETRAIN_ROOT = Path(__file__).resolve().parent
DEFAULT_TOKENIZER = PRETRAIN_ROOT / "tokenizers" / "kernel-v6.12-16k.json"


@dataclass
class GPTConfig:
    vocab_size: int = 16384  # must match the shared tokenizer (tokenizer_train.py --vocab-size)
    block_size: int = 512  # context length (tokens)
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1  # this corpus gets seen hundreds of times; some regularization delays memorization
    bias: bool = False  # no bias terms in Linear/Norm layers (LLaMA-style, matches modern practice, fewer params)
    rope_theta: float = 10000.0


@dataclass
class TrainConfig:
    piece: str = "f2fs"
    tokenizer_path: Path = DEFAULT_TOKENIZER

    # batching
    batch_size: int = 32  # per-step micro-batch; tune down if you hit CUDA OOM on 16GB
    grad_accum_steps: int = 2  # effective batch size = batch_size * grad_accum_steps

    # schedule
    max_iters: int = 5_000
    warmup_iters: int = 200
    lr: float = 1e-3
    min_lr: float = 1e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # logging / eval / checkpointing
    eval_interval: int = 100
    eval_iters: int = 50  # batches averaged per eval, for both train and val
    log_interval: int = 10
    checkpoint_interval: int = 500

    device: str = "cuda"
    dtype: str = "bfloat16"  # Ada Lovelace has native bf16 tensor cores -- no fp16 loss-scaling complexity needed
    compile: bool = True  # torch.compile -- meaningful speedup on Ada, first iter will be slow (graph capture)

    seed: int = 1337

    @property
    def data_dir(self) -> Path:
        return PRETRAIN_ROOT / "data" / self.piece

    @property
    def out_dir(self) -> Path:
        return PRETRAIN_ROOT / "checkpoints" / self.piece


# name -> (model shape, TrainConfig overrides). Rough non-embedding param
# count for this architecture is 12 * n_layer * n_embd^2; the tied
# embedding table adds vocab_size * n_embd on top.
PRESETS: dict[str, tuple[GPTConfig, dict]] = {
    # ~3M non-embedding + 4.2M embedding. Laptop default.
    "tiny": (GPTConfig(n_layer=4, n_head=4, n_embd=256, block_size=512), {}),
    # ~10.6M non-embedding + 6.3M embedding. Expect it to overfit sooner than tiny.
    "small": (GPTConfig(n_layer=6, n_head=6, n_embd=384, block_size=512), {"lr": 6e-4, "min_lr": 6e-5}),
    # GPT-2-small shape, ~85M non-embedding. Rented-GPU stage, big pieces only.
    "gpt2s": (
        GPTConfig(n_layer=12, n_head=12, n_embd=768, block_size=1024),
        {"lr": 3e-4, "min_lr": 3e-5, "batch_size": 12, "grad_accum_steps": 4, "max_iters": 60_000, "warmup_iters": 1_000, "eval_interval": 250},
    ),
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
