"""Tokenizes the corpus with the trained tokenizer and writes it out as
train.bin/val.bin -- flat uint16 arrays memory-mapped at training time
(nanoGPT's convention), so train.py never holds the whole token stream in
RAM and random-access batch sampling is just an array slice.

Each file is encoded separately and joined with an <|endoftext|> token so
the model learns document boundaries (it should not treat the end of one
struct's file and the start of an unrelated file as continuous text).

Split is 95/5 by *file*, not by token position within a shuffled stream --
splitting mid-file would leak a function's ending into val when its
beginning was in train, silently inflating val performance.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

from tokenizer_train import DEFAULT_CORPUS_DIR, iter_corpus_files

PRETRAIN_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--tokenizer", type=Path, default=PRETRAIN_ROOT / "data" / "tokenizer.json")
    parser.add_argument("--out-dir", type=Path, default=PRETRAIN_ROOT / "data")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if not args.tokenizer.is_file():
        raise SystemExit(f"No tokenizer at {args.tokenizer} -- run tokenizer_train.py first.")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    eot_id = tokenizer.token_to_id("<|endoftext|>")

    files = list(iter_corpus_files(args.corpus_dir))
    if not files:
        raise SystemExit(f"No corpus files found under {args.corpus_dir}.")

    rng = random.Random(args.seed)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * args.val_fraction))
    val_files, train_files = files[:n_val], files[n_val:]

    def encode_split(split_files: list[Path], name: str) -> None:
        ids: list[int] = []
        for path in tqdm(split_files, desc=f"tokenizing {name}"):
            text = path.read_text(errors="ignore")
            ids.extend(tokenizer.encode(text).ids)
            ids.append(eot_id)
        arr = np.array(ids, dtype=np.uint16)
        out_path = args.out_dir / f"{name}.bin"
        arr.tofile(out_path)
        print(f"{name}: {len(split_files)} files, {len(arr):,} tokens -> {out_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    encode_split(train_files, "train")
    encode_split(val_files, "val")


if __name__ == "__main__":
    main()
