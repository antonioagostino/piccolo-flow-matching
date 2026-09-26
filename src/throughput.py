# scripts/throughput.py
import glob, re, sys, torch

points = []
for p in sorted(glob.glob(f"{sys.argv[1]}/snapshots/*/step_*.pt")):
    s = torch.load(p, map_location="cpu", weights_only=True)
    step = int(re.search(r"step_(\d+)", p).group(1))
    points.append((step, s["train_samples_seen"], s["training_time_seconds"]))

(st0, n0, t0), (st1, n1, t1) = points[0], points[-1]
print(f"{(st1 - st0) / (t1 - t0):.2f} it/s   {(n1 - n0) / (t1 - t0):.0f} campioni/s   (step {st0} -> {st1})")