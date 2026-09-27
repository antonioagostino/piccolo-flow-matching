import glob, torch

for d in sorted(glob.glob("checkpoints/sweep_*")):
    for p in glob.glob(f"{d}/checkpoint_last_*.pt"):
        c = torch.load(p, map_location="cpu", weights_only=True)
        print(f'{d:42s} step {c["optimizer_steps"]:>6}  val_loss {c["val_loss"]:.5f}')