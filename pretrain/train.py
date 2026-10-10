"""Single-GPU pretraining loop for one piece. Run from inside pretrain/:

    python train.py --piece f2fs --preset tiny            # start fresh
    python train.py --piece f2fs --preset tiny --resume   # resume from checkpoints/f2fs/ckpt.pt

Expects data/<piece>/train.bin + val.bin + data_meta.json (prepare_data.py)
and the shared tokenizer (tokenizer_train.py) to already exist.

Writes checkpoints/<piece>/ckpt.pt (latest) and ckpt_best.pt (lowest
val_loss so far -- on a corpus this small the latest checkpoint is usually
the more memorized one, not the better one). Every checkpoint records the
piece, preset, tokenizer sha256 and corpus sha256 that produced it, so
pieces trained in different runs or on different machines can be checked
for compatibility before anything stitches them together.

Logs to stdout and appends to checkpoints/<piece>/log.csv (iter,train_loss,
val_loss,lr,tok_per_sec) so you can plot the loss curve afterward --
watching train_loss keep dropping while val_loss flattens/rises is the
single most important thing to look at with a corpus this small.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from config import PRESETS, GPTConfig, TrainConfig, file_sha256
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--piece", default="f2fs")
    parser.add_argument("--preset", default="tiny", choices=sorted(PRESETS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-iters", type=int, default=None, help="override the preset's max_iters")
    args = parser.parse_args()

    preset_gcfg, train_overrides = PRESETS[args.preset]
    gcfg = copy.deepcopy(preset_gcfg)
    tcfg = TrainConfig(piece=args.piece, **train_overrides)
    if args.max_iters is not None:
        tcfg.max_iters = args.max_iters

    torch.manual_seed(tcfg.seed)
    if not torch.cuda.is_available() and tcfg.device == "cuda":
        print("CUDA not available -- falling back to CPU (no torch.compile). Fine for a smoke test, far too slow for a real run.")
        tcfg.device = "cpu"
        tcfg.compile = False

    device_type = "cuda" if "cuda" in tcfg.device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[tcfg.dtype]
    ctx = (
        torch.autocast(device_type=device_type, dtype=ptdtype)
        if device_type == "cuda"
        else torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False)
    )

    meta_path = tcfg.data_dir / "data_meta.json"
    if not meta_path.is_file():
        raise SystemExit(
            f"No {meta_path} -- run build_corpus.py, tokenizer_train.py, then prepare_data.py --piece {tcfg.piece} first."
        )
    data_meta = json.loads(meta_path.read_text())
    if not tcfg.tokenizer_path.is_file() or file_sha256(tcfg.tokenizer_path) != data_meta["tokenizer_sha256"]:
        raise SystemExit(
            f"{tcfg.tokenizer_path} is missing or isn't the tokenizer data/{tcfg.piece}/*.bin was encoded with -- "
            "re-run prepare_data.py."
        )
    if data_meta["vocab_size"] != gcfg.vocab_size:
        raise SystemExit(f"tokenizer vocab {data_meta['vocab_size']} != GPTConfig.vocab_size {gcfg.vocab_size}")
    for split, n in data_meta["tokens"].items():
        if n <= gcfg.block_size + 1:
            raise SystemExit(f"{split}.bin has {n} tokens, too few for block_size={gcfg.block_size}")
    provenance = {
        "piece": tcfg.piece,
        "preset": args.preset,
        "tokenizer": tcfg.tokenizer_path.name,
        "tokenizer_sha256": data_meta["tokenizer_sha256"],
        "corpus_sha256": data_meta["corpus_sha256"],
    }

    tcfg.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = tcfg.out_dir / "ckpt.pt"
    best_path = tcfg.out_dir / "ckpt_best.pt"
    log_path = tcfg.out_dir / "log.csv"

    model = GPT(gcfg).to(tcfg.device)
    print(f"{tcfg.piece}/{args.preset}: {model.num_params() / 1e6:.1f}M params, block_size={gcfg.block_size}, "
          f"vocab_size={gcfg.vocab_size}, train {data_meta['tokens']['train']:,} / val {data_meta['tokens']['val']:,} tokens")

    optimizer = model.configure_optimizer(tcfg.weight_decay, tcfg.lr, (tcfg.beta1, tcfg.beta2))

    start_iter = 0
    best_val = float("inf")
    if args.resume:
        if not ckpt_path.is_file():
            raise SystemExit(f"--resume given but no checkpoint at {ckpt_path}")
        # weights_only=False: our own checkpoint embeds a GPTConfig dataclass
        # instance (not just tensors), which torch>=2.6's safer default
        # rejects. Safe here since this checkpoint is always one we wrote
        # ourselves, never an untrusted download.
        ckpt = torch.load(ckpt_path, map_location=tcfg.device, weights_only=False)
        mismatched = {k: (ckpt.get(k), v) for k, v in provenance.items() if ckpt.get(k) != v}
        if mismatched:
            raise SystemExit(f"{ckpt_path} was trained with different settings (checkpoint, now): {mismatched}")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt["iter"] + 1
        best_val = ckpt.get("best_val", best_val)
        print(f"resumed from {ckpt_path} at iter {start_iter} (best val_loss so far {best_val:.4f})")

    def save(path: Path, it: int) -> None:
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model  # unwrap torch.compile wrapper
        torch.save(
            {"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(), "iter": it,
             "gpt_config": gcfg, "best_val": best_val, **provenance},
            path,
        )

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
            if it > 0 and losses["val"] < best_val:
                best_val = losses["val"]
                save(best_path, it)
                print(f"  new best val_loss -> {best_path}")

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

        if it > 0 and (it % tcfg.checkpoint_interval == 0 or it == tcfg.max_iters):
            save(ckpt_path, it)
            print(f"  saved checkpoint at iter {it} -> {ckpt_path}")

    log_file.close()
    print("done.")


if __name__ == "__main__":
    main()
