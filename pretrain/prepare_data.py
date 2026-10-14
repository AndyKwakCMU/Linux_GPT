"""Tokenizes one piece's corpus (build_corpus.py output) with the shared
tokenizer and writes data/<piece>/train.bin + val.bin -- flat uint16 arrays
memory-mapped at training time (nanoGPT's convention), so train.py never
holds the whole token stream in RAM and batch sampling is an array slice.

    python prepare_data.py --piece f2fs

Layout of each .bin: for every source file, `<|file|>` + "path\\n", then
that file's units for this split in source order, then `<|endoftext|>`.
The path header tells the model which file it's in (segment.c and
super.c read differently); the end-of-text token keeps it from treating
the end of one file and the start of an unrelated one as continuous.

The train/val split itself was decided per top-level unit in
build_corpus.py, not here -- see that file for why.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from config import DEFAULT_TOKENIZER, PRETRAIN_ROOT, PRESETS, file_sha256


FILE, EOT = object(), object()  # placeholders for the special tokens in the segment stream
ENCODE_BATCH = 8192  # segments per encode_batch call (Rust-side parallel; whole-kernel corpus is ~10^6 units)


def split_records(corpus_path: Path, split: str) -> Iterator[dict]:
    # streamed, not loaded: the whole-kernel corpus.jsonl is >1GB
    with corpus_path.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec["split"] == split:
                yield rec


def segments(records: Iterator[dict]) -> Iterator[object]:
    # records are in source order, so consecutive same-path runs are one file
    for path, group in itertools.groupby(records, key=lambda r: r["path"]):
        yield FILE
        yield path + "\n"
        for rec in group:
            yield rec["text"]
        yield EOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--piece", default="f2fs")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    args = parser.parse_args()

    data_dir = PRETRAIN_ROOT / "data" / args.piece
    corpus_path = data_dir / "corpus.jsonl"
    if not corpus_path.is_file():
        raise SystemExit(f"No {corpus_path} -- run `python build_corpus.py --piece {args.piece}` first.")
    if not args.tokenizer.is_file():
        raise SystemExit(f"No tokenizer at {args.tokenizer} -- run tokenizer_train.py first.")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    file_id = tokenizer.token_to_id("<|file|>")
    if eot_id is None or file_id is None:
        raise SystemExit(f"{args.tokenizer} lacks <|endoftext|>/<|file|> special tokens -- retrain with tokenizer_train.py.")

    meta = {"piece": args.piece, "tokenizer": args.tokenizer.name, "tokenizer_sha256": file_sha256(args.tokenizer),
            "vocab_size": tokenizer.get_vocab_size(), "corpus_sha256": file_sha256(corpus_path), "tokens": {}}

    for split in ("train", "val"):
        n_tokens = 0
        n_units = 0
        stream = segments(split_records(corpus_path, split))
        with (data_dir / f"{split}.bin").open("wb") as out:
            while chunk := list(itertools.islice(stream, ENCODE_BATCH)):
                texts = [s for s in chunk if isinstance(s, str)]
                encoded = iter(tokenizer.encode_batch(texts))
                ids: list[int] = []
                for seg in chunk:
                    if seg is FILE:
                        ids.append(file_id)
                    elif seg is EOT:
                        ids.append(eot_id)
                    else:
                        ids.extend(next(encoded).ids)
                n_units += len(texts) - sum(seg is FILE for seg in chunk)  # every FILE is followed by a path-header text
                np.array(ids, dtype=np.uint16).tofile(out)
                n_tokens += len(ids)
        meta["tokens"][split] = n_tokens
        print(f"{split}: {n_units} units, {n_tokens:,} tokens -> {data_dir / f'{split}.bin'}")

    (data_dir / "data_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    max_block = max(cfg.block_size for cfg, _ in PRESETS.values())
    if meta["tokens"]["val"] <= max_block:
        print(f"WARNING: val has only {meta['tokens']['val']} tokens; presets with block_size >= that can't sample from it.")


if __name__ == "__main__":
    main()
