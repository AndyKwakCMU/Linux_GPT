# Handoff — 2026-10-14

State of the `pretrain/` work at the end of the session that trained the
`kernel -> fs -> f2fs` chain. Read this, then `pretrain/README.md`
("Results", "Roadmap"). Delete or rewrite this file once it's stale.

## Where things are

- **Branch `pretrain-fs-piece`**, pushed to origin, ahead of
  `main`, **not merged / no PR yet**. Working tree clean.
- **Machine**: NVIDIA laptop, Gentoo (OpenRC + elogind, no systemd), KDE
  Wayland, **Quadro RTX 5000 Max-Q 16GB (Turing, sm_75)**, Python 3.14,
  62GB RAM. `pretrain/.venv` has torch 2.11+cu128.
- **Data** (gitignored, rebuildable in minutes): `pretrain/data/{f2fs,fs,kernel}/`.
- **Checkpoints** (gitignored, *not* rebuildable cheaply, ~10h of GPU
  time total): `pretrain/checkpoints/<piece>/<run>/{ckpt_best,ckpt}.pt` + `log.csv`.
- **Logs + per-run summaries** (gitignored): `pretrain/logs/`, e.g.
  `kernel_{small,medium,large}_summary.txt`, `overnight_summary.txt`.

## Results in one table

Probe = `eval_probes.py --piece f2fs`, same 123 held-out F2FS functions.

| checkpoint | path | F2FS loss / ppl / top-1 |
|---|---|---|
| **best** | `checkpoints/f2fs/ft_kernel_fs_large/ckpt_best.pt` (59.5M) | 0.854 / 2.3 / 78.8% |
| practical | `checkpoints/f2fs/ft_kernel_fs_medium/ckpt_best.pt` (25.7M) | 0.887 / 2.4 / 77.8% |
| small | `checkpoints/f2fs/ft_kernel_fs/ckpt_best.pt` (10.6M) | 0.939 / 2.6 / 76.6% |
| F2FS-only baseline | `checkpoints/f2fs/small800/ckpt_best.pt` | 1.815 / 6.1 / 65.0% |

Base models: `checkpoints/kernel/{small,medium,large}16000/`; fs
fine-tunes: `checkpoints/fs/ft_kernel{,_medium,_large}/`. Takeaways
(data > size on small corpora; chain steps each help; `large` has about
exhausted v6.12) are written up in `pretrain/README.md` "Results".

## What changed this session (all committed)

- `train.py`: fp16 + GradScaler fallback when the GPU lacks native bf16;
  `--run`, `--init-from`, `--dropout`, `--weight-decay`, `--lr`,
  `--warmup-iters`, `--eval-interval`.
- `config.py`: `medium` / `large` presets.
- `pieces/fs.yaml`, `pieces/kernel.yaml`; optional `max_file_bytes` in
  piece manifests.
- `prepare_data.py`: streaming + `encode_batch` (byte-identical output).
- `pipeline.sh`: the full chain + probes per preset.
- Docs: `pretrain/README.md` Results/Presets/Roadmap, root README, CLAUDE.md.

## Next: bridge stage 1 (not started)

From `pretrain/README.md` Roadmap #2: `pretrain/serve.py` (stdlib
`http.server`, in the pretrain venv) exposing `/embed`, `/score`,
`/complete` for a checkpoint, then measure piece embeddings vs
`nomic-embed-text` as a retriever/reranker on a question ->
expected-symbol set. Verifier untouched by construction.

**Open with the user before building** (the brainstorm was started, not
answered): what concrete failure of `linuxgpt ask` the bridge should fix
— wrong chunk retrieved, abstention despite the right chunk present, or
something else. That decides whether `/embed` (retrieval) or `/score`
(reranking) comes first, and what the eval set looks like. Also open:
which checkpoint to serve (medium is the suggested default).

Deferred, by choice: more pretraining data (git history) — only worth it
for models bigger than `large`.

## Gotchas learned the hard way

- **Long runs**: launch detached so they survive the session and the laptop
  doesn't sleep:
  `setsid nohup elogind-inhibit --what=sleep:idle:handle-lid-switch --why=... --mode=block bash pipeline.sh medium > logs/x.nohup 2>&1 < /dev/null &`
  Then find the real PID with `pgrep -f "^bash pipeline.sh"` — `$!`/a loose
  `pgrep -f` catches the short-lived wrapper shell.
- Use `python -u` when redirecting `train.py` to a file, or the log stays
  empty until exit (stdout block buffering). `log.csv` is always flushed.
- torch.compile prints a harmless gcc `_POSIX_C_SOURCE redefined` warning
  block on first run (Triton + Python 3.14 headers). Don't grep it away
  with a pattern like `from ` — that also hides `initialized from ...`.
- `--init-from` needs the same `--preset` as the source checkpoint.
- Two `fs` files (`fs/btrfs/inode.c`, `block/bfq-iosched.c`) exceed the
  `kernel` piece's 256KB cap, so they're in `fs` but absent from `kernel`
  — expected, not a split leak.
