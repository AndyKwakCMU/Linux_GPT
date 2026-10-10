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
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from config import DEFAULT_TOKENIZER, PRETRAIN_ROOT, PRESETS, file_sha256


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

    records = [json.loads(line) for line in corpus_path.open()]
    meta = {"piece": args.piece, "tokenizer": args.tokenizer.name, "tokenizer_sha256": file_sha256(args.tokenizer),
            "vocab_size": tokenizer.get_vocab_size(), "corpus_sha256": file_sha256(corpus_path), "tokens": {}}

    for split in ("train", "val"):
        ids: list[int] = []
        split_records = [r for r in records if r["split"] == split]
        # records are in source order, so consecutive same-path runs are one file
        for path, group in itertools.groupby(split_records, key=lambda r: r["path"]):
            ids.append(file_id)
            ids.extend(tokenizer.encode(path + "\n").ids)
            for rec in group:
                ids.extend(tokenizer.encode(rec["text"]).ids)
            ids.append(eot_id)
        arr = np.array(ids, dtype=np.uint16)
        arr.tofile(data_dir / f"{split}.bin")
        meta["tokens"][split] = len(arr)
        print(f"{split}: {len(split_records)} units, {len(arr):,} tokens -> {data_dir / f'{split}.bin'}")

    (data_dir / "data_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    max_block = max(cfg.block_size for cfg, _ in PRESETS.values())
    if meta["tokens"]["val"] <= max_block:
        print(f"WARNING: val has only {meta['tokens']['val']} tokens; presets with block_size >= that can't sample from it.")


if __name__ == "__main__":
    main()
