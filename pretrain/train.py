"""Single-GPU pretraining loop. Run from inside pretrain/:

    python train.py                  # start fresh
    python train.py --resume         # resume from checkpoints/ckpt.pt

Expects data/train.bin + data/val.bin (prepare_data.py) and
data/tokenizer.json (tokenizer_train.py) to already exist.

Logs to stdout and appends to checkpoints/log.csv (iter,loss,val_loss,lr,
tok_per_sec) so you can plot the loss curve afterward -- watching train_loss
keep dropping while val_loss flattens/rises is the single most important
thing to look at with a corpus this small (see README's overfitting note).
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch

from config import GPTConfig, TrainConfig
from model import GPT


def get_batch(data_path: Path, block_size: int, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    # re-opened per call (not cached) -- np.memmap holding a long-lived file
    # handle across a training run of this length has a known tendency to
    # leak/grow in RSS; this keeps memory flat at the cost of a cheap reopen
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    if device == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def lr_at(it: int, cfg: TrainConfig) -> float:
    if it < cfg.warmup_iters:
        return cfg.lr * (it + 1) / cfg.warmup_iters
    if it > cfg.max_iters:
        return cfg.min_lr
    decay_ratio = (it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, cfg: TrainConfig, gcfg: GPTConfig, ctx) -> dict[str, float]:
    model.eval()
    out = {}
    for split in ("train", "val"):
        path = cfg.data_dir / f"{split}.bin"
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_batch(path, gcfg.block_size, cfg.batch_size, cfg.device)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-iters", type=int, default=None, help="override config.TrainConfig.max_iters")
    args = parser.parse_args()

    gcfg = GPTConfig()
    tcfg = TrainConfig()
    if args.max_iters is not None:
        tcfg.max_iters = args.max_iters

    torch.manual_seed(tcfg.seed)
    if not torch.cuda.is_available() and tcfg.device == "cuda":
        print("CUDA not available -- falling back to CPU. This will be extremely slow; expected on this project.")
        tcfg.device = "cpu"

    device_type = "cuda" if "cuda" in tcfg.device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[tcfg.dtype]
    ctx = (
        torch.autocast(device_type=device_type, dtype=ptdtype)
        if device_type == "cuda"
        else torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False)
    )

    if not (tcfg.data_dir / "train.bin").is_file():
        raise SystemExit(f"No data/train.bin at {tcfg.data_dir} -- run tokenizer_train.py then prepare_data.py first.")

    tcfg.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = tcfg.out_dir / "ckpt.pt"
    log_path = tcfg.out_dir / "log.csv"

    model = GPT(gcfg).to(tcfg.device)
    print(f"model: {model.num_params() / 1e6:.1f}M params, block_size={gcfg.block_size}, vocab_size={gcfg.vocab_size}")

    optimizer = model.configure_optimizer(tcfg.weight_decay, tcfg.lr, (tcfg.beta1, tcfg.beta2))

    start_iter = 0
    if args.resume:
        if not ckpt_path.is_file():
            raise SystemExit(f"--resume given but no checkpoint at {ckpt_path}")
        # weights_only=False: our own checkpoint embeds a GPTConfig dataclass
        # instance (not just tensors), which torch>=2.6's safer default
        # rejects. Safe here since this checkpoint is always one we wrote
        # ourselves, never an untrusted download.
        ckpt = torch.load(ckpt_path, map_location=tcfg.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt["iter"] + 1
        print(f"resumed from {ckpt_path} at iter {start_iter}")

    if tcfg.compile:
        print("compiling model (torch.compile) -- first step will be slow ...")
        model = torch.compile(model)

    write_header = not log_path.is_file()
    log_file = log_path.open("a", newline="")
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow(["iter", "train_loss", "val_loss", "lr", "tok_per_sec"])

    train_path = tcfg.data_dir / "train.bin"
    t_last = time.time()
    tokens_since_last_log = 0

    for it in range(start_iter, tcfg.max_iters + 1):
        lr = lr_at(it, tcfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if it % tcfg.eval_interval == 0:
            losses = estimate_loss(model, tcfg, gcfg, ctx)
            elapsed = time.time() - t_last
            tok_per_sec = tokens_since_last_log / elapsed if elapsed > 0 else 0.0
            print(f"iter {it}: train_loss {losses['train']:.4f}, val_loss {losses['val']:.4f}, lr {lr:.2e}, {tok_per_sec:.0f} tok/s")
            log_writer.writerow([it, losses["train"], losses["val"], lr, tok_per_sec])
            log_file.flush()
            t_last = time.time()
            tokens_since_last_log = 0

        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(tcfg.grad_accum_steps):
            x, y = get_batch(train_path, gcfg.block_size, tcfg.batch_size, tcfg.device)
            with ctx:
                _, loss = model(x, y)
                loss = loss / tcfg.grad_accum_steps
            loss.backward()
            tokens_since_last_log += x.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimizer.step()

        if it % tcfg.log_interval == 0 and it % tcfg.eval_interval != 0:
            print(f"iter {it}: loss {loss.item() * tcfg.grad_accum_steps:.4f}, lr {lr:.2e}")

        if it > 0 and it % tcfg.checkpoint_interval == 0:
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model  # unwrap torch.compile wrapper
            torch.save(
                {"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(), "iter": it, "gpt_config": gcfg},
                ckpt_path,
            )
            print(f"  saved checkpoint at iter {it} -> {ckpt_path}")

    log_file.close()
    print("done.")


if __name__ == "__main__":
    main()
