"""
Conditional Flow Matching training loop for 2D toy data.

Convention from references/NOTES.md section 8:
    t=0 is the base distribution N(0,I), t=1 is the data distribution.
    x_t = (1-t) x_base + t x_data
    target velocity = d x_t / dt = x_data - x_base

Usage:
    python -m src.training.train_flow_matching --distribution eight_gaussians --steps 3000
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import torch

from src.data.registry import get_distribution
from src.models.velocity_field import VelocityField
from src.training.common import save_checkpoint_and_plot

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--distribution", default="eight_gaussians")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="if set, overrides --steps as epochs * steps_per_epoch",
    )
    p.add_argument("--steps_per_epoch", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--time_emb_dim", type=int, default=32)
    p.add_argument("--base_std", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--output-dir", dest="out_dir", default=str(PROJECT_ROOT / "checkpoints"))
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    total_steps = args.steps if args.epochs is None else args.epochs * args.steps_per_epoch

    sample_fn = get_distribution(args.distribution)
    model = VelocityField(
        data_dim=2,
        time_emb_dim=args.time_emb_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(
        f"device={DEVICE} distribution={args.distribution} "
        f"flow_matching steps={total_steps} seed={args.seed}"
    )

    loss_history = []
    t0 = time.time()

    for step in range(total_steps):
        x_data_np = sample_fn(args.batch_size, seed=args.seed * 1_000_000 + step)
        x_data = torch.from_numpy(x_data_np).float().to(DEVICE)
        x_base = torch.randn_like(x_data) * args.base_std
        t = torch.rand(args.batch_size, device=DEVICE)
        t_view = t.view(-1, 1)

        # NOTES.md section 8 convention: base at t=0, data at t=1.
        x_t = (1.0 - t_view) * x_base + t_view * x_data
        target_velocity = x_data - x_base

        pred = model(x_t, t)
        loss = ((pred - target_velocity) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if step % args.log_every == 0 or step == total_steps - 1:
            elapsed = time.time() - t0
            print(f"[step {step:>6}/{total_steps}] loss={loss.item():.5f} elapsed={elapsed:.1f}s")

    ckpt_name = f"{args.distribution}_flow_matching_seed{args.seed}"
    save_checkpoint_and_plot(
        model=model,
        config=vars(args),
        loss_history=loss_history,
        ckpt_name=ckpt_name,
        out_dir=Path(args.out_dir),
        project_root=PROJECT_ROOT,
        title=f"Flow Matching loss - {args.distribution}",
    )


if __name__ == "__main__":
    main()
