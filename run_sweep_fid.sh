for d in checkpoints/sweep_*; do
  name=$(basename "$d"); bb=$(echo "$name" | cut -d_ -f2)
  python -m src.evaluate --config "configs/runs/$name.yaml" \
    --snapshot-dir "$d/snapshots/$bb" \
    --num-samples 5000 --n-steps 25 --guidance 0 --seed 0 \
    --output-dir "results/$name"
done