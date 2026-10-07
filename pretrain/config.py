"""Model + training hyperparameters. Two dataclasses, no framework config
magic -- edit values here or override via CLI flags on train.py, that's it.

Sizing note: defaults target a single 16GB laptop GPU (RTX 5000 Ada
Generation, Laptop variant) running for multiple days. See pretrain/README.md
for why the *default* model is deliberately smaller than what would fit in
16GB -- the current corpus (.kernel_src/, F2FS + curated docs) is only ~1-1.5M
tokens, and a 124M-param GPT-2-scale model will memorize that in well under a
day, not spend days learning anything general. Scale n_layer/n_head/n_embd up
once the corpus is bigger (see prepare_data.py --corpus-dir)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PRETRAIN_ROOT = Path(__file__).resolve().parent


@dataclass
class GPTConfig:
    vocab_size: int = 8192  # matches tokenizer_train.py's default --vocab-size
    block_size: int = 1024  # context length (tokens)
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0  # 0 is correct for a single-epoch-scale run; raise if you add more data and start overfitting less
    bias: bool = False  # no bias terms in Linear/Norm layers (LLaMA-style, matches modern practice, fewer params)
    rope_theta: float = 10000.0


@dataclass
class TrainConfig:
    data_dir: Path = PRETRAIN_ROOT / "data"
    out_dir: Path = PRETRAIN_ROOT / "checkpoints"
    tokenizer_path: Path = PRETRAIN_ROOT / "data" / "tokenizer.json"

    # batching
    batch_size: int = 12  # per-step micro-batch; tune down if you hit CUDA OOM on 16GB
    grad_accum_steps: int = 4  # effective batch size = batch_size * grad_accum_steps

    # schedule
    max_iters: int = 60_000
    warmup_iters: int = 1_000
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # logging / eval / checkpointing
    eval_interval: int = 250
    eval_iters: int = 50  # batches averaged per eval, for both train and val
    log_interval: int = 10
    checkpoint_interval: int = 500

    device: str = "cuda"
    dtype: str = "bfloat16"  # Ada Lovelace has native bf16 tensor cores -- no fp16 loss-scaling complexity needed
    compile: bool = True  # torch.compile -- meaningful speedup on Ada, first iter will be slow (graph capture)

    seed: int = 1337
