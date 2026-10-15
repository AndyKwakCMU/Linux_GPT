#!/usr/bin/env bash
# kernel/<preset> base pretraining -> fs -> f2fs and kernel -> f2fs fine-tunes, all probed.
#   bash pipeline.sh medium large      (presets run in order; ~9h for both on a Quadro RTX 5000 Max-Q)
# Results -> logs/kernel_<preset>_summary.txt. A finished base pretrain is skipped on re-run.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs
source .venv/bin/activate

for P in "$@"; do
S=logs/kernel_${P}_summary.txt
echo "started $(date)" > "$S"
stage() {  # label log train.py-args...
    local label=$1 log=$2; shift 2
    echo "[$(date +%H:%M)] $label start" >> "$S"
    python -u train.py "$@" > "$log" 2>&1 || { echo "$label FAILED, see $log" >> "$S"; return 1; }
    echo "[$(date +%H:%M)] $label done, best: $(grep -B1 'new best' "$log" | grep '^iter' | tail -1)" >> "$S"
}
probe() {  # piece ckpt label
    echo "== $3 (probed on $1)" >> "$S"
    python eval_probes.py --piece "$1" --checkpoint "$2" 2>&1 | grep -A6 "held-out functions," >> "$S"
}

K=checkpoints/kernel/${P}16000/ckpt_best.pt
if ! grep -q "^16000," checkpoints/kernel/${P}16000/log.csv 2>/dev/null; then
    stage "kernel/$P pretrain" logs/kernel_${P}16000.log \
        --piece kernel --preset $P --run ${P}16000 --max-iters 16000 --warmup-iters 500 --eval-interval 500 || continue
fi
rm -rf checkpoints/fs/ft_kernel_$P checkpoints/f2fs/ft_kernel_fs_$P checkpoints/f2fs/ft_kernel_$P
stage "kernel/$P -> fs" logs/ft_kernel_fs_$P.log \
    --piece fs --preset $P --run ft_kernel_$P --init-from $K --lr 2e-4 --warmup-iters 50 --max-iters 2000 --eval-interval 100 || continue
stage "kernel/$P -> fs -> f2fs" logs/ft_kernel_fs_f2fs_$P.log \
    --piece f2fs --preset $P --run ft_kernel_fs_$P --init-from checkpoints/fs/ft_kernel_$P/ckpt_best.pt --lr 1e-4 --warmup-iters 20 --max-iters 400 --eval-interval 25 || continue
stage "kernel/$P -> f2fs" logs/ft_kernel_f2fs_$P.log \
    --piece f2fs --preset $P --run ft_kernel_$P --init-from $K --lr 1e-4 --warmup-iters 20 --max-iters 400 --eval-interval 25 || continue

probe fs   $K "kernel/$P, no fine-tune"
probe fs   checkpoints/fs/ft_kernel_$P/ckpt_best.pt "kernel/$P -> fs"
probe f2fs $K "kernel/$P, no fine-tune"
probe f2fs checkpoints/fs/ft_kernel_$P/ckpt_best.pt "kernel/$P -> fs"
probe f2fs checkpoints/f2fs/ft_kernel_fs_$P/ckpt_best.pt "kernel/$P -> fs -> f2fs"
probe f2fs checkpoints/f2fs/ft_kernel_$P/ckpt_best.pt "kernel/$P -> f2fs"
echo "finished $(date)" >> "$S"
done
