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
import torch

from src.data.registry import get_distribution
from src.forward_process import VPSDE
from src.models.denoiser import Denoiser
from src.training.common import save_checkpoint_and_plot

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
    p.add_argument(
        "--snr_gamma", type=float, default=None,
        help="if set, apply SNR-based loss reweighting (docs/technical.md §9.4): "
             "weight(t) = min(SNR(t), snr_gamma), SNR(t) = alpha(t)^2/std(t)^2, "
             "which upweights small-t/high-SNR samples (capped for stability) "
             "instead of the standard Hang et al. 2023 min-SNR-gamma direction "
             "(which downweights them) -- see the comment at the loss computation. "
             "None (default) = uniform per-t weighting, unchanged from before.",
    )
    p.add_argument(
        "--tag", default="",
        help="optional suffix appended to the checkpoint name (e.g. 'snr') to avoid "
             "overwriting an existing checkpoint for the same distribution/param/seed",
    )
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
        per_sample_loss = ((pred - target) ** 2).mean(dim=-1)
        if args.snr_gamma is not None:
            # docs/technical.md §9.4's hypothesis: pinwheel's fine angular structure
            # (angular_std=0.04 rad) only survives at small t (high SNR); uniform-t
            # eps-loss gives that regime no more weight than the easy, structure-free
            # large-t region. Upweight small-t/high-SNR samples (capped at snr_gamma
            # for stability, since SNR -> inf as t -> 0) instead of using the
            # Hang et al. 2023 min(SNR,gamma)/SNR form, which weights in the opposite
            # direction (it DOWN-weights small-t) and would fight the stated goal.
            snr = (alpha[:, 0] ** 2) / (std[:, 0] ** 2)
            weight = torch.clamp(snr, max=args.snr_gamma)
            per_sample_loss = per_sample_loss * weight
        loss = per_sample_loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if step % args.log_every == 0 or step == total_steps - 1:
            elapsed = time.time() - t0
            print(f"[step {step:>6}/{total_steps}] loss={loss.item():.5f}  elapsed={elapsed:.1f}s")

    ckpt_name = f"{args.distribution}_{args.param}_seed{args.seed}"
    if args.tag:
        ckpt_name = f"{ckpt_name}_{args.tag}"
    save_checkpoint_and_plot(
        model=model,
        config=vars(args),
        loss_history=loss_history,
        ckpt_name=ckpt_name,
        out_dir=Path(args.out_dir),
        project_root=PROJECT_ROOT,
        title=f"Training loss — {args.distribution} / {args.param}-pred",
    )


if __name__ == "__main__":
    main()
