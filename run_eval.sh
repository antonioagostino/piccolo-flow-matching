#!/usr/bin/env bash
# run_eval.sh — FID evaluation of the long runs. N_STEPS fixed before seeing the results.
set -u
grep -q "matmul_precision" src/evaluate.py || { echo "TF32 line missing in src/evaluate.py"; exit 1; }
mkdir -p logs final/dit final/unet
N_STEPS=50
SEED=0

ev() {  # ev <backbone> <snapshot_dir> <output_dir> <num_samples> <n_steps> <guidance> <log>
  python -m src.evaluate --config "configs/runs/long_$1.yaml" --snapshot-dir "$2" \
    --num-samples "$4" --n-steps "$5" --guidance "$6" --seed "$SEED" \
    --output-dir "$3" 2>&1 | tee -a "$7"
}

# 1. Training curves: every snapshot, N=10k
for bb in dit unet; do
  ev "$bb" "checkpoints/long_$bb/snapshots/$bb" "results/long_$bb" 10000 "$N_STEPS" 0 "logs/eval_curve_$bb.log"
done

# 2. The three plots
for ax in flops samples time; do
  python -m src.evaluate --plot \
    "results/long_dit/snapshots_dit_n10000_steps${N_STEPS}_w0_seed${SEED}.csv" \
    "results/long_unet/snapshots_unet_n10000_steps${N_STEPS}_w0_seed${SEED}.csv" \
    --x-axis "$ax" --plot-output "results/fid_vs_${ax}.png" 2>&1 | tee -a logs/eval_plots.log
done

# 3. Final snapshots in their own folders
for bb in dit unet; do
  cp "$(ls checkpoints/long_$bb/snapshots/$bb/step_*.pt | sort | tail -n 1)" "final/$bb/"
done

# 4. Final FIDs, N=50k
for bb in dit unet; do
  ev "$bb" "final/$bb" "results/final_$bb" 50000 "$N_STEPS" 0 "logs/eval_final_$bb.log"
done

# 5. NFE sweep, N=10k
for n in 1 2 5 10 25 50 100; do
  for bb in dit unet; do
    ev "$bb" "final/$bb" "results/final_$bb" 10000 "$n" 0 "logs/eval_nfe_$bb.log"
  done
done

# 6. CFG sweep, N=10k (w=0 is already in the NFE sweep)
for w in 0.5 1 2 3; do
  for bb in dit unet; do
    ev "$bb" "final/$bb" "results/final_$bb" 10000 "$N_STEPS" "$w" "logs/eval_cfg_$bb.log"
  done
done

if grep -l "Traceback" logs/eval_*.log; then
  echo "WARNING: the logs listed above contain errors"
else
  echo "No errors in the logs"
fi