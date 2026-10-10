"""Held-out probes: did the piece learn F2FS's patterns, or only memorize
its training text? Loss curves alone can't tell you what a val_loss of 2.1
*means*; this puts numbers next to baselines.

    python eval_probes.py --piece f2fs
    python eval_probes.py --piece f2fs --checkpoint checkpoints/f2fs/ckpt.pt --show 10

For every held-out (split=val) C function in data/<piece>/corpus.jsonl --
functions the model never trained on -- the model is given the file header
plus everything up to and including the function's opening `{` line (its
leading comment, signature, and `{`), then scored on the body:

- loss / perplexity on the body tokens, teacher-forced
- top-1 next-token accuracy on all body tokens
- top-1 accuracy on identifier tokens only (tokens that start a C name:
  `f2fs_`, `sbi`, `->blkaddr`...). This is the closest cheap proxy for
  "does it know which helper/field comes next", which is the intuition
  the bridge work later wants to use.

Baselines on exactly the same tokens:
- unigram: always predict token frequencies from train.bin
- bigram: predict from the previous token only (interpolated with unigram)
A transformer that doesn't clearly beat bigram has learned surface
statistics, not structure. --show N prints the held-out functions the
model finds most surprising, with path:line, which is often the most
interesting output of the whole script.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from config import DEFAULT_TOKENIZER, PRETRAIN_ROOT, file_sha256
from model import GPT

_IDENT_START = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_]")


def load_model(path: Path, tokenizer_path: Path, device: str) -> tuple[GPT, dict]:
    # weights_only=False: see the matching comment in train.py's resume path
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("tokenizer_sha256") != file_sha256(tokenizer_path):
        raise SystemExit(f"{tokenizer_path} is not the tokenizer {path} was trained with ({ckpt.get('tokenizer')}).")
    model = GPT(ckpt["gpt_config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def heldout_functions(corpus_path: Path) -> list[dict]:
    """Val-split code units with a kernel-style function body: a line that
    is exactly `{`. Prompt = everything through that line; body = the rest."""
    probes = []
    for line in corpus_path.open():
        rec = json.loads(line)
        if rec["split"] != "val" or rec["kind"] != "code":
            continue
        lines = rec["text"].splitlines(keepends=True)
        brace = next((i for i, l in enumerate(lines) if l.rstrip("\n") == "{"), None)
        if brace is None or brace == len(lines) - 1:
            continue
        probes.append({
            "path": rec["path"],
            "line": rec["start_line"],
            "signature": _signature(lines[:brace]),
            "prompt": "".join(lines[: brace + 1]),
            "body": "".join(lines[brace + 1 :]),
        })
    return probes


def _signature(before_brace: list[str]) -> str:
    """The function's first signature line: walking back from the `{`, the
    nearest column-0 line that isn't a comment or preprocessor line
    (continuation lines of a multi-line signature are indented)."""
    for line in reversed(before_brace):
        if line[:1].strip() and not line.startswith(("/*", " *", "*", "#")):
            return line.strip()
    return "?"


class NgramBaseline:
    def __init__(self, train_ids: np.ndarray, vocab_size: int, lam: float = 0.9):
        ids = train_ids.astype(np.int64)
        uni = np.bincount(ids, minlength=vocab_size).astype(np.float64) + 1.0  # add-one
        self.uni = uni / uni.sum()
        self.uni_top = int(np.argmax(uni))
        pairs, counts = np.unique(ids[:-1] * vocab_size + ids[1:], return_counts=True)
        self.bigram = dict(zip(pairs.tolist(), counts.tolist()))
        self.prev_count = np.bincount(ids[:-1], minlength=vocab_size)
        # most likely next token per previous token
        order = np.lexsort((-counts, pairs // vocab_size))
        firsts = pairs[order] // vocab_size
        keep = np.r_[True, firsts[1:] != firsts[:-1]]
        self.bi_top = dict(zip(firsts[keep].tolist(), (pairs[order][keep] % vocab_size).tolist()))
        self.V, self.lam = vocab_size, lam

    def score(self, prev: list[int], tgt: list[int]) -> dict[str, float]:
        uni_nll = bi_nll = uni_hit = bi_hit = 0.0
        for a, b in zip(prev, tgt):
            pu = self.uni[b]
            ca = self.prev_count[a]
            pb = (self.lam * self.bigram.get(a * self.V + b, 0) / ca + (1 - self.lam) * pu) if ca else pu
            uni_nll -= math.log(pu)
            bi_nll -= math.log(pb)
            uni_hit += b == self.uni_top
            bi_hit += b == self.bi_top.get(a, self.uni_top)
        return {"uni_nll": uni_nll, "bi_nll": bi_nll, "uni_hit": uni_hit, "bi_hit": bi_hit}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--piece", default="f2fs")
    parser.add_argument("--checkpoint", type=Path, default=None, help="default: checkpoints/<piece>/ckpt_best.pt")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--show", type=int, default=5, help="print the N most surprising held-out functions")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data_dir = PRETRAIN_ROOT / "data" / args.piece
    ckpt_path = args.checkpoint or PRETRAIN_ROOT / "checkpoints" / args.piece / "ckpt_best.pt"
    if not ckpt_path.is_file():
        raise SystemExit(f"No checkpoint at {ckpt_path} -- train.py hasn't saved one yet.")
    model, ckpt = load_model(ckpt_path, args.tokenizer, args.device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    file_id = tokenizer.token_to_id("<|file|>")
    block = ckpt["gpt_config"].block_size

    probes = heldout_functions(data_dir / "corpus.jsonl")
    if not probes:
        raise SystemExit("No held-out functions found -- check build_corpus.py output.")
    baseline = NgramBaseline(np.fromfile(data_dir / "train.bin", dtype=np.uint16), ckpt["gpt_config"].vocab_size)
    is_ident = np.array([bool(_IDENT_START.match(tokenizer.decode([i]))) for i in range(tokenizer.get_vocab_size())])

    totals = {"n": 0, "nll": 0.0, "hit": 0, "ident_n": 0, "ident_hit": 0, "uni_nll": 0.0, "bi_nll": 0.0, "uni_hit": 0, "bi_hit": 0}
    per_fn = []
    skipped = 0
    for p in probes:
        prompt_ids = [file_id] + tokenizer.encode(p["path"] + "\n").ids + tokenizer.encode(p["prompt"]).ids
        body_ids = tokenizer.encode(p["body"]).ids
        room = block + 1 - len(prompt_ids)
        if room < 2:
            skipped += 1  # comment+signature alone fills the context window
            continue
        ids = prompt_ids + body_ids[:room]
        x = torch.tensor([ids[:-1]], device=args.device)
        with torch.no_grad():
            h = model.final_norm(_forward_hidden(model, x))
            logits = model.lm_head(h)[0]
        start = len(prompt_ids) - 1  # logits[start] predicts the first body token
        tgt = torch.tensor(ids[start + 1 :], device=args.device)
        lg = logits[start:]
        nll = F.cross_entropy(lg.float(), tgt, reduction="sum").item()
        pred = lg.argmax(-1)
        hits = (pred == tgt).cpu().numpy()
        ident_mask = is_ident[tgt.cpu().numpy()]

        totals["n"] += len(tgt)
        totals["nll"] += nll
        totals["hit"] += int(hits.sum())
        totals["ident_n"] += int(ident_mask.sum())
        totals["ident_hit"] += int(hits[ident_mask].sum())
        for k, v in baseline.score(ids[start : -1], ids[start + 1 :]).items():
            totals[k] += v
        per_fn.append((nll / len(tgt), p))

    n = totals["n"]
    print(f"{ckpt['piece']}/{ckpt['preset']} @ iter {ckpt['iter']}: {len(per_fn)} held-out functions, {n:,} body tokens"
          + (f" ({skipped} skipped: prompt longer than block_size)" if skipped else ""))
    print(f"{'':12}{'loss':>8}{'ppl':>10}{'top-1':>9}")
    print(f"{'model':12}{totals['nll'] / n:8.3f}{math.exp(totals['nll'] / n):10.1f}{totals['hit'] / n:9.1%}")
    print(f"{'bigram':12}{totals['bi_nll'] / n:8.3f}{math.exp(totals['bi_nll'] / n):10.1f}{totals['bi_hit'] / n:9.1%}")
    print(f"{'unigram':12}{totals['uni_nll'] / n:8.3f}{math.exp(totals['uni_nll'] / n):10.1f}{totals['uni_hit'] / n:9.1%}")
    print(f"identifier tokens: model top-1 {totals['ident_hit'] / max(1, totals['ident_n']):.1%} over {totals['ident_n']:,}")

    if args.show:
        print(f"\nmost surprising held-out functions (per-token loss):")
        for loss, p in sorted(per_fn, key=lambda t: -t[0])[: args.show]:
            print(f"  {loss:6.3f}  {p['path']}:{p['line']}  {p['signature'][:80]}")


def _forward_hidden(model: GPT, idx: torch.Tensor) -> torch.Tensor:
    """GPT.forward only returns last-position logits at inference; probes
    need every position, so run the blocks directly."""
    x = model.token_emb(idx)
    for block in model.blocks:
        x = block(x)
    return x


if __name__ == "__main__":
    main()
