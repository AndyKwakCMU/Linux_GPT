"""Trains the ONE byte-level BPE tokenizer every piece shares, on the whole
pinned kernel tree, and saves it to tokenizers/kernel-<version>-<N>k.json.

    python tokenizer_train.py                  # full v6.12 tree
    python tokenizer_train.py --sample 0.25    # deterministic 25% of files, if RAM is tight

Run this once, commit the output, and then treat it as frozen. Pieces
(F2FS now; VFS/block/mm later on rented GPUs) are trained separately, but
they can only be stitched -- routed between, distilled into one model,
merged -- if their token ids mean the same thing. Retraining the tokenizer
silently makes every existing checkpoint incompatible; train.py and
sample.py check the tokenizer's sha256 against each checkpoint for exactly
that reason.

Why the whole kernel and not just F2FS: the vocabulary has to serve pieces
that don't exist yet. Training it on F2FS alone would spend merges on
`f2fs_`/`nat_`/`sit_` and leave mm/ or block/ identifiers fragmented.
Only the tokenizer sees the whole tree; what a piece's *weights* are
trained on is set by its pieces/<name>.yaml.

Why byte-level BPE: C is full of things a word tokenizer mangles -- `->`,
`__attribute__`, tab indentation -- and byte-level BPE has no
out-of-vocabulary problem: any input, including files it never saw,
round-trips losslessly.

Why NOT GPT-2's pre-tokenizer regex: it treats `_` as punctuation and
splits letters from digits, so `f2fs_sb_info` can never become fewer than
7 tokens (' f','2','fs','_','sb','_','info') no matter how often it appears.
C_PRETOKENIZE below keeps a whole C identifier (and a whole numeric literal
like 0x1F or 4096UL) as one pre-token, so BPE is free to learn `f2fs_` or
`_sb_info` -- or the whole identifier -- as a single token.

Files over --max-file-bytes (256KB) are skipped: in v6.12 that's 296
files but 37% of all text bytes, nearly all machine-generated register
headers (drivers/gpu/drm/amd/include/asic_reg/*_sh_mask.h, up to 24MB
each). They'd skew merges toward `mmDCN_..._MASK` names nobody reads, and
their millions of unique identifiers exhausted 16GB of RAM during training.

Why 16384: big enough to give the full kernel's common identifiers and
idioms their own tokens, small enough that a tiny piece (256-dim, tied
embeddings -> 16384*256 = 4.2M params) is still mostly transformer, not
embedding table. Every piece pays for this vocab, so it errs small.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
from tqdm import tqdm

from kernel_tarball import PRETRAIN_ROOT, iter_members, load_kernel_pin, verified_tarball

TOKENIZERS_DIR = PRETRAIN_ROOT / "tokenizers"
SPECIAL_TOKENS = ["<|endoftext|>", "<|file|>"]
TEXT_SUFFIXES = {".c", ".h", ".S", ".rst"}
# identifier | numeric literal | punctuation run | whitespace (GPT-2's
# trailing-whitespace trick kept: a space before a word attaches to it)
C_PRETOKENIZE = r" ?[A-Za-z_][A-Za-z0-9_]*| ?[0-9][A-Za-z0-9_]*| ?[^\sA-Za-z0-9_]+|\s+(?!\S)|\s+"


def new_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(C_PRETOKENIZE), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def is_tokenizer_text(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1]
    return Path(name).suffix in TEXT_SUFFIXES or name.startswith("Kconfig")


def in_sample(rel: str, fraction: float) -> bool:
    if fraction >= 1.0:
        return True
    h = int.from_bytes(hashlib.sha256(rel.encode()).digest()[:8], "big")
    return (h % 10_000) < fraction * 10_000


def default_tokenizer_path(kernel_version: str, vocab_size: int) -> Path:
    return TOKENIZERS_DIR / f"kernel-{kernel_version}-{vocab_size // 1024}k.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--sample", type=float, default=1.0, help="fraction of kernel files to train on (by path hash)")
    parser.add_argument("--max-file-bytes", type=int, default=256 * 1024, help="skip larger (generated) files")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--skip-verify", action="store_true", help="don't check the tarball sha256 against kernel.org (offline)")
    args = parser.parse_args()

    pin = load_kernel_pin()
    out = args.out or default_tokenizer_path(pin["kernel_version"], args.vocab_size)
    tarball, tarball_sha = verified_tarball(pin["tarball_name"], pin["checksums_url"], skip_verify=args.skip_verify)

    stats = {"files": 0, "bytes": 0, "skipped_large": 0}

    def texts():
        for rel, data in tqdm(iter_members(tarball), desc="kernel files", unit="file", mininterval=10):
            if is_tokenizer_text(rel) and in_sample(rel, args.sample):
                if len(data) > args.max_file_bytes:
                    stats["skipped_large"] += 1
                    continue
                stats["files"] += 1
                stats["bytes"] += len(data)
                yield data.decode("utf-8", errors="replace")

    print(f"Training {args.vocab_size}-token BPE on {pin['tarball_name']} (sample={args.sample}) ...")
    tokenizer = new_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(texts(), trainer=trainer)

    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out))
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_suffix(".sha256").write_text(
        f"{sha}  {out.name}\n"
        f"# trained on {stats['files']} files / {stats['bytes']:,} bytes of {pin['tarball_name']} "
        f"(tarball sha256 {tarball_sha}), sample={args.sample}, "
        f"skipped {stats['skipped_large']} files > {args.max_file_bytes} bytes\n"
    )
    print(f"Saved tokenizer ({tokenizer.get_vocab_size()} tokens, {stats['files']} files) -> {out}\nsha256 {sha}")


if __name__ == "__main__":
    main()
