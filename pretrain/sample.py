"""Generates text from a trained checkpoint, for sanity-checking what the
model has actually learned as training progresses -- run this periodically
against checkpoints/ckpt.pt, not just at the end. Early on expect garbage;
partway through expect plausible-looking C syntax with wrong semantics
(that's the model learning surface structure before content); by the point
it's memorizing the small F2FS corpus (see README) it'll start reproducing
near-verbatim chunks of real source for prompts that match training data.

    python sample.py --prompt "static int f2fs_"
    python sample.py --piece f2fs --checkpoint checkpoints/f2fs/ckpt.pt --prompt "/*" --temperature 1.0

Defaults to the piece's best-val checkpoint. Refuses to decode with a
tokenizer other than the one the checkpoint was trained with -- token ids
from a different vocabulary decode to fluent-looking garbage, not an error.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from config import DEFAULT_TOKENIZER, file_sha256
from model import GPT

PRETRAIN_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piece", default="f2fs")
    parser.add_argument("--checkpoint", type=Path, default=None, help="default: checkpoints/<piece>/ckpt_best.pt")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--prompt", type=str, default="/*")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.checkpoint = args.checkpoint or PRETRAIN_ROOT / "checkpoints" / args.piece / "ckpt_best.pt"

    if not args.checkpoint.is_file():
        raise SystemExit(f"No checkpoint at {args.checkpoint} -- train.py hasn't saved one yet.")

    # weights_only=False: see the matching comment in train.py's resume path
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if ckpt.get("tokenizer_sha256") != file_sha256(args.tokenizer):
        raise SystemExit(f"{args.tokenizer} is not the tokenizer {args.checkpoint} was trained with ({ckpt.get('tokenizer')}).")
    gcfg = ckpt["gpt_config"]
    model = GPT(gcfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {ckpt['piece']}/{ckpt['preset']} checkpoint from iter {ckpt['iter']} "
          f"({model.num_params() / 1e6:.1f}M params, best val_loss {ckpt['best_val']:.4f})")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    ids = tokenizer.encode(args.prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=args.device)

    out = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
