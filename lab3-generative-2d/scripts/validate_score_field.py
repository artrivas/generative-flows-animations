"""
Qualitative validation: score field visualization.

Loads a trained checkpoint (eps or v parametrization), evaluates the score
at a small t via DenoiserWrapper, and overlays the resulting vector field
on real data samples. If training + the eps/v->score conversion are
correct, the field should point toward the data modes (ascending
log-density) near the data, not in random directions or uniformly toward
the origin.

Usage:
    python scripts/validate_score_field.py --checkpoint checkpoints/eight_gaussians_eps_seed0.pt
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.registry import get_distribution
from src.forward_process import VPSDE
from src.models.denoiser import Denoiser, DenoiserWrapper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--t", type=float, default=0.05)
    p.add_argument("--grid_size", type=int, default=20)
    p.add_argument("--grid_limit", type=float, default=6.0)
    p.add_argument("--n_data", type=int, default=3000)
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    config = ckpt["config"]

    model = Denoiser(
        data_dim=2,
        time_emb_dim=config["time_emb_dim"],
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    process = VPSDE()
    wrapper = DenoiserWrapper(model, process, param=config["param"])

    dist_name = config["distribution"]
    sample_fn = get_distribution(dist_name)
    x0_np = sample_fn(args.n_data, seed=config["seed"])

    xs = np.linspace(-args.grid_limit, args.grid_limit, args.grid_size)
    ys = np.linspace(-args.grid_limit, args.grid_limit, args.grid_size)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.stack([XX.ravel(), YY.ravel()], axis=1)
    grid_t = torch.from_numpy(grid).float().to(DEVICE)
    t = torch.full((grid_t.shape[0],), args.t, device=DEVICE)

    with torch.no_grad():
        score = wrapper.score(grid_t, t).cpu().numpy()

    U = score[:, 0].reshape(XX.shape)
    V = score[:, 1].reshape(XX.shape)
    magnitude = np.sqrt(U ** 2 + V ** 2)

    U_norm = U / (magnitude + 1e-8)
    V_norm = V / (magnitude + 1e-8)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=130)
    ax.scatter(x0_np[:, 0], x0_np[:, 1], s=3, alpha=0.25, color="#4C72B0", label="data ($x_0$)")
    q = ax.quiver(XX, YY, U_norm, V_norm, magnitude, cmap="viridis", scale=25, width=0.003)
    fig.colorbar(q, ax=ax, label="|score|", shrink=0.8)
    ax.set_xlim(-args.grid_limit, args.grid_limit)
    ax.set_ylim(-args.grid_limit, args.grid_limit)
    ax.set_aspect("equal")
    ax.set_title(f"Score field — {dist_name} / {config['param']}-pred, t={args.t}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    out_path = PROJECT_ROOT / "outputs" / "sanity_checks" / f"score_field_{dist_name}_{config['param']}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
