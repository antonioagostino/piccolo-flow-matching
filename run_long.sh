#!/usr/bin/env bash
set -u
mkdir -p logs
for name in long_dit long_unet; do
  echo "=== $name — start $(date '+%H:%M:%S') ==="
  python -m src.train --config "configs/runs/$name.yaml" 2>&1 | tee "logs/$name.log"
  echo "=== $name — end $(date '+%H:%M:%S') ==="
done
if grep -l "Traceback" logs/long_*.log; then
  echo "Attention: logs above contain errors"
else
  echo "No errors in the logs"
fi