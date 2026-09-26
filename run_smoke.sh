#!/usr/bin/env bash
# run_smoke.sh — pre-flight: FID sanity checks, smoke training and smoke evaluation.
set -u
mkdir -p logs

echo "=== FID sanity check: real vs real (expected: low, ~3) ==="
python -m src.evaluate --config configs/runs/smoke_dit.yaml --real-vs-real --seed 0 \
  2>&1 | tee logs/smoke_fid_real_vs_real.log

echo "=== FID sanity check: untrained model (expected: very high) ==="
python -m src.evaluate --config configs/runs/smoke_dit.yaml --untrained \
  --num-samples 1000 --n-steps 10 --guidance 0 --seed 0 \
  2>&1 | tee logs/smoke_fid_untrained.log

for bb in dit unet; do
  echo "=== smoke training: $bb ==="
  python -m src.train --config "configs/runs/smoke_$bb.yaml" 2>&1 | tee "logs/smoke_$bb.log"

  echo "=== throughput: $bb (indicative only, short run) ==="
  python src/throughput.py "checkpoints/smoke_$bb"

  echo "=== smoke evaluation: $bb ==="
  python -m src.evaluate --config "configs/runs/smoke_$bb.yaml" \
    --snapshot-dir "checkpoints/smoke_$bb/snapshots/$bb" \
    --num-samples 1000 --n-steps 10 --guidance 0 --seed 0 \
    --output-dir "results/smoke_$bb" 2>&1 | tee -a "logs/smoke_$bb.log"
done

if grep -l "Traceback" logs/smoke_*.log; then
  echo "WARNING: the logs listed above contain errors"
else
  echo "No errors in the logs"
fi