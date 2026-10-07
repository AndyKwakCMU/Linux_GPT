"""Generates text from a trained checkpoint, for sanity-checking what the
model has actually learned as training progresses -- run this periodically
against checkpoints/ckpt.pt, not just at the end. Early on expect garbage;
partway through expect plausible-looking C syntax with wrong semantics
(that's the model learning surface structure before content); by the point
it's memorizing the small F2FS corpus (see README) it'll start reproducing
near-verbatim chunks of real source for prompts that match training data.

    python sample.py --prompt "static int f2fs_"
    python sample.py --prompt "/*" --max-new-tokens 300 --temperature 1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import GPT

PRETRAIN_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PRETRAIN_ROOT / "checkpoints" / "ckpt.pt")
    parser.add_argument("--tokenizer", type=Path, default=PRETRAIN_ROOT / "data" / "tokenizer.json")
    parser.add_argument("--prompt", type=str, default="/*")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"No checkpoint at {args.checkpoint} -- train.py hasn't saved one yet.")

    # weights_only=False: see the matching comment in train.py's resume path
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    gcfg = ckpt["gpt_config"]
    model = GPT(gcfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded checkpoint from iter {ckpt['iter']} ({model.num_params() / 1e6:.1f}M params)")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    ids = tokenizer.encode(args.prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=args.device)

    out = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
