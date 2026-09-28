#!/usr/bin/env bash
# run_eval_b.sh — FID at 50k with 3 evaluation seeds, on the val-loss-selected
# and on the final snapshots of both backbones.
set -u
grep -q "matmul_precision" src/evaluate.py || { echo "TF32 line missing in src/evaluate.py"; exit 1; }

DIT_BEST=0098000     # from checkpoint_best_dit.pt
UNET_BEST=0110172    # from checkpoint_best_unet.pt

mkdir -p logs selected/dit selected/unet
cp "checkpoints/long_dit/snapshots/dit/step_${DIT_BEST}.pt"    selected/dit/
cp "checkpoints/long_unet/snapshots/unet/step_${UNET_BEST}.pt" selected/unet/

for seed in 0 1 2; do
  for bb in dit unet; do
    for kind in selected final; do
      python -m src.evaluate --config "configs/runs/long_$bb.yaml" --snapshot-dir "$kind/$bb" \
        --num-samples 50000 --n-steps 50 --guidance 0 --seed "$seed" \
        --output-dir "results/${kind}_$bb" 2>&1 | tee -a "logs/eval_b_${kind}_$bb.log"
    done
  done
done

if grep -l "Traceback" logs/eval_b_*.log; then
  echo "WARNING: the logs listed above contain errors"
else
  echo "No errors in the logs"
fi