"""
Denoising score-matching training loop for the 2D toy Denoiser.

Uses the VP forward process (validated in step 2) to build x_t during
training: eps/v-prediction (NOTES.md §5-6) are defined against the
VP-style kernel (mean = alpha(t) x0, std = sqrt(1 - alpha(t)^2)), so VP is
the only forward process compatible with this parametrization.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") --
agnostic so the exact same script runs unmodified on Colab GPU later.

Usage:
    python -m src.training.train_score --distribution eight_gaussians \
        --param eps --steps 3000 --seed 0
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.registry import get_distribution
from src.forward_process import VPSDE
from src.models.denoiser import Denoiser

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--distribution", default="eight_gaussians")
    p.add_argument("--param", choices=["eps", "v"], default="eps")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument(
        "--epochs", type=int, default=None,
        help="if set, overrides --steps as epochs * steps_per_epoch",
    )
    p.add_argument("--steps_per_epoch", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--time_emb_dim", type=int, default=32)
    p.add_argument("--t_min", type=float, default=1e-5)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--output-dir", dest="out_dir", default=str(PROJECT_ROOT / "checkpoints"))
    return p.parse_args()


def build_target(param, x0, noise, alpha, std):
    """eps target: NOTES.md §5. v target: NOTES.md §6 (v = alpha*eps - std*x0)."""
    if param == "eps":
        return noise
    return alpha * noise - std * x0


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    total_steps = args.steps if args.epochs is None else args.epochs * args.steps_per_epoch

    sample_fn = get_distribution(args.distribution)
    process = VPSDE()

    model = Denoiser(
        data_dim=2,
        time_emb_dim=args.time_emb_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"device={DEVICE}  distribution={args.distribution}  param={args.param}  "
          f"steps={total_steps}  seed={args.seed}")

    loss_history = []
    t0 = time.time()

    for step in range(total_steps):
        x0_np = sample_fn(args.batch_size, seed=args.seed * 1_000_000 + step)
        x0 = torch.from_numpy(x0_np).float().to(DEVICE)

        t = torch.rand(args.batch_size, device=DEVICE) * (1.0 - args.t_min) + args.t_min
        noise = torch.randn_like(x0)

        alpha, std = process.marginal_prob(torch.ones_like(x0), t)  # alpha(t), std(t)
        mean, _ = process.marginal_prob(x0, t)
        x_t = mean + std * noise

        target = build_target(args.param, x0, noise, alpha, std)

        pred = model(x_t, t)
        loss = ((pred - target) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if step % args.log_every == 0 or step == total_steps - 1:
            elapsed = time.time() - t0
            print(f"[step {step:>6}/{total_steps}] loss={loss.item():.5f}  elapsed={elapsed:.1f}s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"{args.distribution}_{args.param}_seed{args.seed}"
    ckpt_path = out_dir / f"{ckpt_name}.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": vars(args)}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    log_path = out_dir / f"{ckpt_name}_losses.csv"
    np.savetxt(log_path, np.array(loss_history), delimiter=",", header="loss", comments="")
    print(f"Saved loss log to {log_path}")

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=120)
    ax.plot(loss_history, linewidth=0.7, alpha=0.5, label="raw")
    window = max(1, total_steps // 50)
    if len(loss_history) >= window:
        smoothed = np.convolve(loss_history, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window - 1, len(loss_history)), smoothed, linewidth=1.5,
                 label=f"smoothed (w={window})")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.set_title(f"Training loss — {args.distribution} / {args.param}-pred")
    ax.legend()
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()

    plot_path = PROJECT_ROOT / "outputs" / "sanity_checks" / f"{ckpt_name}_loss_curve.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss curve to {plot_path}")


if __name__ == "__main__":
    main()
