#!/usr/bin/env bash
# run_sweep.sh — Run all the sweep runs, if one fails
# it passes to the next one.
set -u
mkdir -p logs

for cfg in configs/runs/sweep_*.yaml; do
  name=$(basename "$cfg" .yaml)
  echo "=== $name — start $(date '+%H:%M:%S') ==="
  python -m src.train --config "$cfg" 2>&1 | tee "logs/$name.log"
  echo "=== $name — end $(date '+%H:%M:%S') ==="
done

echo "=== sweep finished $(date '+%H:%M:%S') ==="
if grep -l "Traceback" logs/sweep_*.log; then
  echo "Attention: the log contain errors"
else
  echo "No errors in the logs"
fi