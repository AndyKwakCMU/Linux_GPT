"""Trains a byte-level BPE tokenizer on the corpus and saves it to
data/tokenizer.json. Byte-level BPE (GPT-2 style) rather than a
word/whitespace tokenizer because C source is full of tokens a word
tokenizer mangles -- `->`, `__attribute__`, `f2fs_i_size_write`, indentation
-- and byte-level BPE has no out-of-vocabulary problem: any input, including
future files it never saw, encodes losslessly.

vocab_size defaults small (8192) on purpose: the corpus is small (see
README), and a huge vocabulary mostly wastes embedding-table parameters on
subwords that appear a handful of times. Raise it once the corpus grows.

Run once per corpus; prepare_data.py and train.py both load the saved
tokenizer rather than retraining it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

PRETRAIN_ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS_DIR = PRETRAIN_ROOT.parent / ".kernel_src"
CORPUS_EXTENSIONS = {".c", ".h", ".rst", ".txt"}


def iter_corpus_files(corpus_dir: Path):
    for path in sorted(corpus_dir.rglob("*")):
        if path.is_file() and path.suffix in CORPUS_EXTENSIONS:
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--out", type=Path, default=PRETRAIN_ROOT / "data" / "tokenizer.json")
    args = parser.parse_args()

    files = [str(p) for p in iter_corpus_files(args.corpus_dir)]
    if not files:
        raise SystemExit(
            f"No {sorted(CORPUS_EXTENSIONS)} files found under {args.corpus_dir}. "
            "Run `linuxgpt ingest` first if you're pointing this at .kernel_src/, "
            "or pass --corpus-dir at a different text directory."
        )
    print(f"Training BPE tokenizer on {len(files)} files under {args.corpus_dir} ...")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=files,
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=["<|endoftext|>"],
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.out))
    print(f"Saved tokenizer ({tokenizer.get_vocab_size()} tokens) -> {args.out}")


if __name__ == "__main__":
    main()
